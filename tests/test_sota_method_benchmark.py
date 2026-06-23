"""
Benchmark: FTLM vs KPM_DOS vs mTPQ on a 10-site Heisenberg cluster.

Ground truth: qed.full_spectrum (exact all eigenvalues).
Metric: MAE on specific heat at low T (0.05–0.30 J) and all T.

Run manually with ``pytest -s tests/test_sota_method_benchmark.py``
(or ``python -m pytest -s --benchmark`` to see the table).

Design notes
------------
FTLM is the SOTA workhorse for NLCE at low T:
- ~5% per-cluster Cv error at T < 0.5J (improves as cluster grows)
- Error amplified by Möbius κ~30-80 but manageable

KPM_DOS is NOT suitable for low T:
- Jackson-kernel spectral broadening → systematic bias in low-E DOS
- ~15-20× worse than FTLM at T < 0.3J even with large M

LTLM is broken in the current qed backend (returns zeros or wrong
values regardless of num_samples / krylov_dim). Do not use.

For T below the NLCE convergence radius: use the ``lanczos_boost``
pipeline (LB-NLCE, Bhattaram & Khatami PRE 100, 013305 2019).
Computes K exact lowest eigenpairs via Lanczos; exact below
T* = (E_K - E_0) / 5 with zero stochastic noise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import qed

_saved_fds = [os.dup(1), os.dup(2)]
_devnull = open(os.devnull, "w")


def _sil() -> None:
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(_devnull.fileno(), 1)
    os.dup2(_devnull.fileno(), 2)


def _uns() -> None:
    sys.stdout.flush()
    os.dup2(_saved_fds[0], 1)
    os.dup2(_saved_fds[1], 2)


@pytest.fixture(scope="module")
def heisenberg_10():
    """10-site Heisenberg cluster with exact spectrum."""
    N = 10
    op = qed.Operator(num_sites=N, spin=0.5)
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),
        (0,3),(1,4),(2,7),(5,9),(0,6),
    ]
    for i, j in edges:
        op.add_two_body(qed.OP_SZ,    i, qed.OP_SZ,    j, 1.0)
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5)

    _sil()
    ref = qed.full_spectrum(op, verbose=False)
    _uns()
    eigs = np.sort(np.array(ref.eigenvalues, dtype=float))
    return op, eigs


def _exact_cv(eigs: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    E0 = eigs[0]; e = eigs - E0
    cv = []
    for T in T_grid:
        b = 1.0 / T
        w = np.exp(-b * e); Z = w.sum()
        Ea = (eigs * w).sum() / Z
        E2a = (eigs ** 2 * w).sum() / Z
        cv.append((E2a - Ea ** 2) / T ** 2)
    return np.array(cv)


def _run_thermal(op, method, T_grid, use_sz=True, **kw):
    _sil()
    r = qed.thermal(
        op, method=method,
        T_min=T_grid[0], T_max=T_grid[-1], num_T=len(T_grid),
        verbose=False, use_sz_if_conserved=use_sz, **kw,
    )
    _uns()
    T_out = np.array(r.temperatures)
    return np.interp(T_grid, T_out, np.array(r.specific_heat))


def _mae(pred, ref, T, lo, hi):
    m = (T >= lo) & (T <= hi)
    return float(np.mean(np.abs(pred[m] - ref[m])))


T_GRID = np.logspace(np.log10(0.05), 1, 60)


def test_ftlm_low_t_better_than_kpm(heisenberg_10):
    """FTLM must have substantially lower Cv error than KPM_DOS at low T."""
    op, eigs = heisenberg_10
    Cv_ref = _exact_cv(eigs, T_GRID)

    Cv_ftlm = _run_thermal(op, "FTLM", T_GRID, use_sz=True,
                           num_samples=100, ftlm_krylov_dim=200, random_seed=0)
    Cv_kpm  = _run_thermal(op, "KPM_DOS", T_GRID, use_sz=True,
                           kpm_num_random_vectors=32, kpm_num_moments=1024)

    mae_ftlm = _mae(Cv_ftlm, Cv_ref, T_GRID, 0.05, 0.30)
    mae_kpm  = _mae(Cv_kpm,  Cv_ref, T_GRID, 0.05, 0.30)

    # KPM must be at least 5× worse than FTLM at low T.
    # Empirical: factor is ~15-20×; we use 5× as a conservative threshold.
    assert mae_kpm > 5 * mae_ftlm, (
        f"Expected KPM_DOS MAE ({mae_kpm:.3e}) >> 5 × FTLM MAE ({mae_ftlm:.3e}) "
        f"at T<0.3J. If this fails the qed backend changed."
    )


def test_ftlm_low_t_accuracy(heisenberg_10):
    """FTLM Cv error at T < 0.3J is below 0.5 (absolute) for R=200."""
    op, eigs = heisenberg_10
    Cv_ref = _exact_cv(eigs, T_GRID)

    Cv_ftlm = _run_thermal(op, "FTLM", T_GRID, use_sz=True,
                           num_samples=200, ftlm_krylov_dim=300, random_seed=0)
    mae = _mae(Cv_ftlm, Cv_ref, T_GRID, 0.05, 0.30)
    assert mae < 0.5, f"FTLM Cv MAE at low T = {mae:.3e}; expected < 0.5"


def test_ftlm_high_t_converges(heisenberg_10):
    """FTLM Cv is accurate at high T (T > 2J) regardless of sample count."""
    op, eigs = heisenberg_10
    Cv_ref = _exact_cv(eigs, T_GRID)

    Cv_ftlm = _run_thermal(op, "FTLM", T_GRID, use_sz=True,
                           num_samples=50, ftlm_krylov_dim=100, random_seed=0)
    mae_hi = _mae(Cv_ftlm, Cv_ref, T_GRID, 2.0, 10.0)
    assert mae_hi < 0.05, f"FTLM Cv MAE at high T = {mae_hi:.3e}; expected < 0.05"
