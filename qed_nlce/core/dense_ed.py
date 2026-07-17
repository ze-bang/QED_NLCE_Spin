"""Exact-diagonalization backend for the NLCE workflow.

Runs a single cluster's ED in-process, entirely on the QED library via
:mod:`qed_nlce.ed.engine`:

  * read the per-cluster Hamiltonian (``Trans.dat`` / ``InterAll.dat``)
    directly into a native ``qed`` operator
    (:func:`qed_nlce.ed.io.read_qed_operator` -- no Python middleman),
  * resolve the cluster's symmetry ONCE
    (:func:`~qed_nlce.ed.engine.resolve_cluster_symmetry`: spatial
    GeneratorSet with the non-abelian residue, U(1)-Sz / native
    Sz-parity / spin-flip / time-reversal flags),
  * EXACT full diagonalization for EVERY cluster (the default policy)
    via :func:`~qed_nlce.ed.engine.full_spectrum` -- one
    ``qed.full_spectrum`` call carrying ALL of that symmetry. Above
    ``options.oftlm_cutoff`` the block-aware plan
    (:func:`~qed_nlce.ed.engine.plan_exact_solve`) is consulted as an
    ADVISORY: an over-cap cluster logs a loud warning (the job may run
    very long or exhaust memory) but still solves exactly, because NLCE
    weight subtraction amplifies stochastic error by ~(T/J)^-order;
  * only with ``options.oftlm_fallback`` (``--oftlm_fallback``): over-cap
    clusters fall back to matrix-free OFTLM
    (:func:`qed_nlce.ed.oftlm_thermodynamics`), producing per-cluster
    thermodynamics directly (no eigenvalue spectrum) with
    independent-seed error bands.

Both tiers persist to ``<output_dir>/output/ed_results.h5`` (the
on-disk contract the NLCE summation kernels read) -- eigenvalues under
``/eigendata/eigenvalues`` for the exact tier, ``/thermodynamics/*``
for both (opt-in for the exact tier via ``options.thermo``, always-on
for OFTLM). Writes are atomic (temp file + ``os.replace``) so a crash
mid-write can never leave a truncated file that the summation kernel
would silently treat as valid.

The pure-Python solver (``tests/oracle/``) is not on any runtime path;
it remains the correctness oracle for the parity suite.
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

from ..ed import thermodynamics, oftlm_thermodynamics
from ..ed.engine import (full_spectrum, plan_exact_solve,
                         resolve_cluster_symmetry)
from ..ed.io import read_qed_operator
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
            "and OFTLM backends (qed_nlce.ed.engine / qed_nlce.ed.oftlm). "
            "Install it (see the sibling QED repo) before running any "
            "pipeline -- failing now rather than after cluster work has "
            f"already run. Original error: {exc}"
        ) from exc


def can_run_in_process(method: str) -> bool:  # noqa: ARG001 - dense only
    """Every supported method is plain dense diagonalization."""
    return True


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
        qop = read_qed_operator(ham_subdir, int(num_sites))
        hilbert_dim = 1 << int(num_sites)
        cutoff = int(getattr(options, "oftlm_cutoff", 1 << 15))
        # ONE symmetry resolution per cluster: the same ClusterSymmetry
        # feeds the feasibility plan AND the solve, so the router can no
        # longer disagree with what qed does at solve time.
        cs = resolve_cluster_symmetry(qop)
        # EXACT-ONLY policy (default): NLCE weight subtraction amplifies
        # stochastic error by ~(T/J)^-order, so a noisy tier at deep
        # orders is worse than useless. The block-aware plan is consulted
        # above the raw cutoff purely as an ADVISORY: an over-cap cluster
        # gets a loud warning (the job may run very long or exhaust
        # memory) and is solved exactly anyway. --oftlm_fallback restores
        # the stochastic tier for over-cap clusters.
        plan = None
        if hilbert_dim > cutoff:
            plan = plan_exact_solve(qop, cs, options, log_tag=log_tag)
        go_exact = (plan is None or plan.feasible
                    or not bool(getattr(options, "oftlm_fallback", False)))
        if go_exact:
            if plan is not None and not plan.feasible:
                logging.warning(
                    "[%s] EXACT-ONLY: cluster exceeds the exact-tier caps "
                    "(%s). Solving exactly anyway -- this job may run very "
                    "long or exhaust memory; raise --exact_max_block/"
                    "--exact_max_sector deliberately, or pass "
                    "--oftlm_fallback to restore the stochastic tier "
                    "(error bands propagate, but deep-order NLCE weights "
                    "will be noise-dominated).",
                    log_tag, plan.reason,
                )
            evals = full_spectrum(
                qop, cs,
                device=getattr(options, "device", "cpu"),
                point_group=str(getattr(options, "point_group", "auto")),
                log_tag=log_tag,
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
            # (repr-keyed sort: term tuples end in a complex coefficient,
            # which is not orderable when the leading fields tie.)
            import zlib
            term_sig = (
                repr(sorted(qop.iter_one_body_terms(), key=repr))
                + repr(sorted(qop.iter_two_body_terms(), key=repr))
            ).encode()
            seed = (zlib.crc32(term_sig)
                    ^ (int(num_sites) << 16)) & 0x7FFFFFFF or 1
            T = _temperature_grid(options)
            res = oftlm_thermodynamics(
                qop, T,
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
