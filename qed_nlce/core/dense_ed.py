"""Exact-diagonalization backend for the NLCE workflow.

Runs a single cluster's ED in-process using :mod:`qed_nlce.ed`, in one
of two tiers:

  * read the per-cluster Hamiltonian (``Trans.dat`` / ``InterAll.dat``),
  * below ``options.oftlm_cutoff`` (raw Hilbert dim): full EXACT
    diagonalization via :func:`qed_nlce.ed.qed_bridge.full_spectrum_qed`,
    which dispatches to ``qed.full_spectrum`` -- U(1) S^z, spatial
    automorphisms (abelian AND non-abelian, via QED's symmetry-adapted
    basis engine), spin-flip Z2, and time-reversal reduction, all
    verified to reproduce the brute-force spectrum
    (``tests/test_qed_bridge_parity.py``);
  * above it: matrix-free OFTLM (:func:`qed_nlce.ed.oftlm_thermodynamics`),
    producing per-cluster thermodynamics directly (no eigenvalue
    spectrum) for clusters too large to diagonalize exactly even with
    full symmetry reduction.

Both tiers persist to ``<output_dir>/output/ed_results.h5`` (the
on-disk contract the NLCE summation kernels read) -- eigenvalues under
``/eigendata/eigenvalues`` for the exact tier, ``/thermodynamics/*``
for both (opt-in for the exact tier via ``options.thermo``, always-on
for OFTLM). Writes are atomic (temp file + ``os.replace``) so a crash
mid-write can never leave a truncated file that the summation kernel
would silently treat as valid.

The pure-Python solver in :mod:`qed_nlce.ed.dense` / ``.symmetry`` is
no longer on this runtime path; it remains as the correctness oracle
for the parity test and existing unit tests.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

import numpy as np

try:  # h5py is the on-disk contract with the summation kernels
    import h5py

    _HAS_H5PY = True
except ImportError:  # pragma: no cover
    _HAS_H5PY = False

from ..ed import solve_spectrum, thermodynamics, oftlm_thermodynamics
from ..ed.qed_bridge import full_spectrum_qed
from ..ed.io import read_operator
from .ed_runner import EDOptions

__all__ = ["can_run_in_process", "run_ed_in_process", "assert_qed_available"]


def assert_qed_available() -> None:
    """Fail fast (before any cluster work) if the ``qed`` backend is missing.

    Both ED tiers (exact ``qed.full_spectrum`` and OFTLM) depend on the
    ``qed`` C++ package. Import it lazily inside a long unattended run
    and a missing/broken install only surfaces after hours of earlier
    cluster work; call this once at workflow start instead.
    """
    try:
        import qed  # noqa: F401
        from qed import _core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'qed' C++ package is required for the QED_NLCE exact-ED "
            "and OFTLM backends (qed_nlce.ed.qed_bridge / qed_nlce.ed.oftlm). "
            "Install it (see the sibling QED repo) before running any "
            "pipeline -- failing now rather than after cluster work has "
            f"already run. Original error: {exc}"
        ) from exc


def can_run_in_process(method: str) -> bool:  # noqa: ARG001 - dense only
    """Every supported method is plain dense diagonalization."""
    return True


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 << 30  # conservative fallback


def _exact_tier_feasible(op, num_sites: int, options: EDOptions,
                         log_tag: str) -> bool:
    """Block-aware feasibility of the EXACT full-spectrum tier.

    The raw Hilbert dimension is the wrong gate: what the dense
    eigensolver actually faces is the largest SYMMETRY BLOCK --
    C(N, N/2) for Sz-conserving models (2^N otherwise), divided by the
    spatial automorphism group, in real arithmetic when time reversal
    holds. A 19-site XXZ tree (2^19 raw) with a reflection is a
    46k-dim real block (~17 GB, very solvable); a 19-site Sz-broken
    cluster is not. Deep NLCE orders are noise-dominated under any
    stochastic solver (weight cancellation ~ (J/T)^n vs flat ~1e-3
    sample error), so every cluster that CAN be exact MUST be -- this
    predicate pushes the exact tier to the true hardware limit instead
    of a fixed 2^N.
    """
    from math import comb

    from ..ed.symmetry import detect_time_reversal, find_automorphisms

    n = int(num_sites)
    if op.conserves_sz():
        sector = comb(n, n // 2)
    elif op.conserves_sz_parity():
        # Sz broken but every term shifts Sz by an even amount (e.g.
        # S+S+ pair terms with Jzpm = 0): up-spin parity halves the
        # space, and the Python solver exploits it (see routing below).
        sector = (1 << n) // 2
    else:
        sector = 1 << n
    n_aut = max(1, len(find_automorphisms(op)))
    block = max(1, sector // n_aut)  # Burnside-average block estimate
    bytes_per = 8 if detect_time_reversal(op) else 16
    # Eigenvalues-only dense solve: matrix + ~25% workspace headroom.
    need = block * block * bytes_per * 1.25
    avail = _mem_available_bytes()
    limit = int(getattr(options, "exact_max_block", 120_000))
    ok = block <= limit and need <= 0.8 * avail
    logging.info(
        "[%s] exact-tier feasibility: N=%d sz_conserved=%s sector=%s "
        "|Aut|=%d block~%s %s need~%.1f GB avail~%.1f GB "
        "exact_max_block=%d -> %s",
        log_tag, n, op.conserves_sz(), f"{sector:,}", n_aut, f"{block:,}",
        "real" if bytes_per == 8 else "complex",
        need / 2**30, avail / 2**30, limit,
        "EXACT" if ok else "OFTLM",
    )
    return ok


def _temperature_grid(options: EDOptions) -> np.ndarray:
    """Log-spaced temperature grid matching the legacy convention."""
    t_min = max(float(options.temp_min), 1e-6)
    t_max = float(options.temp_max)
    bins = int(options.temp_bins)
    if bins <= 1:
        return np.array([t_max], dtype=np.float64)
    return np.logspace(np.log10(t_min), np.log10(t_max), bins)


def _atomic_h5_write(final_path: str, write_fn) -> None:
    """Write an HDF5 file atomically: build it at a per-process temp path,
    then ``os.replace`` onto ``final_path``.

    A crash (or ``kill -9``) mid-write leaves only the orphaned temp file
    behind -- never a truncated ``final_path`` that the summation kernel's
    h5py reader would otherwise silently accept as "valid but empty" and
    cascade-skip every cluster depending on it.
    """
    tmp_path = f"{final_path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        with h5py.File(tmp_path, "w") as f:
            write_fn(f)
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _write_results(
    out_subdir: str,
    eigenvalues: np.ndarray,
    num_sites: int,
    options: EDOptions,
) -> None:
    """Write ``ed_results.h5`` with the eigenvalue spectrum (+thermo)."""
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required to write ed_results.h5")

    eig = np.asarray(sorted(eigenvalues), dtype=np.float64)
    h5_path = os.path.join(out_subdir, "ed_results.h5")

    def _write(f):
        grp = f.create_group("eigendata")
        grp.create_dataset("eigenvalues", data=eig)
        grp.attrs["num_eigenvalues"] = len(eig)
        grp.attrs["hilbert_dim"] = 1 << int(num_sites)
        grp.attrs["num_sites"] = int(num_sites)

        if options.thermo:
            T = _temperature_grid(options)
            res = thermodynamics(eig, T)
            tgrp = f.create_group("thermodynamics")
            tgrp.create_dataset("temperatures", data=np.asarray(res.temperatures))
            tgrp.create_dataset("energy", data=np.asarray(res.energy))
            tgrp.create_dataset("specific_heat", data=np.asarray(res.specific_heat))
            tgrp.create_dataset("entropy", data=np.asarray(res.entropy))
            tgrp.create_dataset("free_energy", data=np.asarray(res.free_energy))

    _atomic_h5_write(h5_path, _write)


def _write_thermo_results(
    out_subdir: str,
    res,
    num_sites: int,
    hilbert_dim: int,
    method: str = "OFTLM",
) -> None:
    """Write ``ed_results.h5`` with ONLY the P(T) thermodynamic curves.

    Used for large clusters solved by OFTLM, which produce thermodynamics
    directly (no eigenvalue spectrum). The ``/thermodynamics`` group matches
    the layout :func:`_write_results` writes for dense clusters, so the NLCE
    summation reads both uniformly.
    """
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required to write ed_results.h5")
    h5_path = os.path.join(out_subdir, "ed_results.h5")

    def _write(f):
        grp = f.create_group("eigendata")
        grp.attrs["num_eigenvalues"] = 0           # spectrum not materialized
        grp.attrs["hilbert_dim"] = int(hilbert_dim)
        grp.attrs["num_sites"] = int(num_sites)
        grp.attrs["method"] = str(method)
        tgrp = f.create_group("thermodynamics")
        tgrp.create_dataset("temperatures", data=np.asarray(res.temperatures))
        tgrp.create_dataset("energy", data=np.asarray(res.energy))
        tgrp.create_dataset("specific_heat", data=np.asarray(res.specific_heat))
        tgrp.create_dataset("entropy", data=np.asarray(res.entropy))
        tgrp.create_dataset("free_energy", data=np.asarray(res.free_energy))
        # Independent-seed resampling errors (stochastic solvers only):
        # the summation kernels use these to decide whether a high-order
        # NLCE weight is signal or amplified sample noise.
        if getattr(res, "std_error", None):
            for prop, arr in res.std_error.items():
                tgrp.create_dataset(f"std_error_{prop}", data=np.asarray(arr))

    _atomic_h5_write(h5_path, _write)


def run_ed_in_process(
    ham_subdir: str,
    output_dir: str,
    num_sites: int,
    options: EDOptions,
    *,
    log_tag: str = "ED-inproc",
    cache: Optional["object"] = None,
    cache_key_extras: Optional[dict] = None,
) -> bool:
    """Run one cluster's full dense ED in-process.

    Returns True on success. When ``cache`` is provided (an
    :class:`qed_nlce.core.cache.EigenvalueCache`) the eigenvalue cache is
    consulted before running and the result persisted on success;
    ``cache_key_extras`` carries the canonicalisation inputs
    (``geometry``, ``cluster_file``).
    """
    # ----- cache lookup -----
    cache_key = None
    if cache is not None and cache_key_extras is not None:
        try:
            cache_key = cache.compute_key(
                geometry=cache_key_extras["geometry"],
                ham_subdir=ham_subdir,
                cluster_file=cache_key_extras.get("cluster_file"),
                options=options,
                num_sites=num_sites,
            )
            if cache.lookup(cache_key, output_dir):
                return True
        except Exception as exc:  # cache must never break correctness
            logging.warning("[%s] eigenvalue cache lookup failed: %s", log_tag, exc)
            cache_key = None

    out_subdir = os.path.join(output_dir, "output")
    os.makedirs(out_subdir, exist_ok=True)

    try:
        op = read_operator(ham_subdir, int(num_sites))
        hilbert_dim = 1 << int(num_sites)
        cutoff = int(getattr(options, "oftlm_cutoff", 1 << 15))
        # Exact whenever possible: below the raw cutoff always; above it,
        # whenever the block-aware feasibility check says the largest
        # symmetry block fits this machine. NLCE weight cancellation makes
        # stochastic solvers unusable at deep orders, so OFTLM is strictly
        # a last resort.
        if hilbert_dim <= cutoff or _exact_tier_feasible(
                op, int(num_sites), options, log_tag):
            # Exact tier. Engine choice:
            #  * Sz-conserving (or small): qed.full_spectrum -- strongest
            #    engine (non-abelian SAB, GPU, scalable rep walk).
            #  * LARGE Sz-broken: the Python solver -- it blocks by
            #    Sz-PARITY (+ spatial + spin-flip + TR reality), which
            #    qed's auto path does not exploit; parity is what makes
            #    e.g. 17-site anisotropic (Jzpm=0) clusters dense-feasible.
            if op.conserves_sz() or hilbert_dim <= (1 << 15):
                evals = full_spectrum_qed(
                    op, device=getattr(options, "device", "cpu"),
                    log_tag=log_tag,
                )
            else:
                import time as _time
                _t0 = _time.monotonic()
                evals = np.asarray(
                    np.sort(solve_spectrum(op, use_symmetry=True)),
                    dtype=np.float64)
                logging.info(
                    "[%s] N=%d exact spectrum via Python parity-blocked "
                    "solver (Sz-broken): %d eigenvalues, %.1fs",
                    log_tag, int(num_sites), evals.shape[0],
                    _time.monotonic() - _t0,
                )
            _write_results(out_subdir, evals, int(num_sites), options)
        else:
            # Large cluster: matrix-free OFTLM (QED). No eigenvalues -- write the
            # per-cluster P(T) directly on the shared temperature grid.
            # Seed is a deterministic function of the cluster's Hamiltonian
            # CONTENT (path-independent, so it is consistent with the
            # content-addressed cache): reproducible run-to-run, but
            # INDEPENDENT across clusters. A single global seed would give
            # same-dimension clusters identical random vectors, making
            # their stochastic errors correlated -- those add coherently
            # in the per-order NLCE sum instead of averaging out.
            import zlib
            term_sig = (repr(sorted(op.single)) + repr(sorted(op.two))).encode()
            seed = (zlib.crc32(term_sig)
                    ^ (int(num_sites) << 16)) & 0x7FFFFFFF or 1
            T = _temperature_grid(options)
            res = oftlm_thermodynamics(
                op, T,
                num_exact=int(getattr(options, "oftlm_num_exact", 16)),
                num_samples=int(getattr(options, "oftlm_num_samples", 20)),
                krylov_dim=int(getattr(options, "oftlm_krylov_dim", 100)),
                random_seed=seed,
                num_seeds=int(getattr(options, "oftlm_num_seeds", 2)),
            )
            _write_thermo_results(out_subdir, res, int(num_sites),
                                  hilbert_dim, method="OFTLM")
    except Exception as exc:
        logging.error(
            "[%s] in-process dense ED failed: %s", log_tag, exc, exc_info=True
        )
        return False

    # ----- cache store -----
    if cache is not None and cache_key is not None:
        try:
            cache.store(cache_key, output_dir)
        except Exception as exc:  # store failures are non-fatal
            logging.warning("[%s] eigenvalue cache store failed: %s", log_tag, exc)
    return True
