"""Self-contained dense exact-diagonalization backend for the NLCE
workflow.

This module replaces the historical ``qed_backend`` (which dispatched to
the external ``qed`` C++ package). It runs a single cluster's *full*
dense diagonalization in-process using :mod:`qed_nlce.ed`:

  * read the per-cluster Hamiltonian (``Trans.dat`` / ``InterAll.dat``),
  * diagonalize with the symmetry-adapted dense solver
    (:func:`qed_nlce.ed.solve_spectrum` -- U(1) S^z, spatial
    automorphisms, spin-flip Z2, and real/TR reduction),
  * persist the full spectrum to ``<output_dir>/output/ed_results.h5``
    under ``/eigendata/eigenvalues`` (the on-disk contract the NLCE
    summation kernels read), plus ``/thermodynamics/*`` when requested.

There is only one method -- full dense diagonalization -- so
:func:`can_run_in_process` is always True.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:  # h5py is the on-disk contract with the summation kernels
    import h5py

    _HAS_H5PY = True
except ImportError:  # pragma: no cover
    _HAS_H5PY = False

from ..ed import solve_spectrum, thermodynamics, oftlm_thermodynamics
from ..ed.io import read_operator
from .ed_runner import EDOptions

__all__ = ["can_run_in_process", "run_ed_in_process"]


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
    with h5py.File(h5_path, "w") as f:
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
    with h5py.File(h5_path, "w") as f:
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
        if hilbert_dim <= cutoff:
            # Small cluster: full dense diagonalization (exact spectrum).
            evals = solve_spectrum(op, use_symmetry=bool(options.use_symm))
            _write_results(out_subdir, evals, int(num_sites), options)
        else:
            # Large cluster: matrix-free OFTLM (QED). No eigenvalues -- write the
            # per-cluster P(T) directly on the shared temperature grid.
            T = _temperature_grid(options)
            res = oftlm_thermodynamics(
                op, T,
                num_exact=int(getattr(options, "oftlm_num_exact", 16)),
                num_samples=int(getattr(options, "oftlm_num_samples", 20)),
                krylov_dim=int(getattr(options, "oftlm_krylov_dim", 100)),
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
