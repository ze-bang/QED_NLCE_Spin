"""Large-cluster finite-temperature thermodynamics via QED's OFTLM.

For clusters too large to diagonalize densely, the Orthogonalized
Finite-Temperature Lanczos Method (Morita & Tohyama, PRR 2, 013205 (2020))
gives thermodynamics with a matrix-free Lanczos matvec: the N_V lowest states
are treated exactly (removing FTLM's low-T bias) and the stochastic trace covers
the rest. This module bridges an NLCE :class:`SpinHalfOperator` to a native
(C++ matvec) QED operator and runs OFTLM on the requested temperature grid.

The result is a :class:`ThermoResult` in the SAME shape as :func:`thermodynamics`
(from full-spectrum clusters), so the NLCE weight subtraction is uniform.
"""
from __future__ import annotations

import numpy as np

from .operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from .thermo import ThermoResult


def spinhalf_to_qed(op: SpinHalfOperator):
    """Build a native QED ``Operator`` (C++ matvec) from a SpinHalfOperator."""
    from qed._core import Operator, OP_SPLUS, OP_SMINUS
    from qed._core import OP_SZ as Q_SZ
    code = {OP_SP: OP_SPLUS, OP_SM: OP_SMINUS, OP_SZ: Q_SZ}
    qop = Operator(int(op.num_sites), 0.5)
    for (o, site, c) in op.single:
        qop.add_one_body(code[int(o)], int(site), complex(c))
    for (o1, s1, o2, s2, c) in op.two:
        qop.add_two_body(code[int(o1)], int(s1),
                         code[int(o2)], int(s2), complex(c))
    return qop


def _oftlm_single(qop, T: np.ndarray, num_exact: int, num_samples: int,
                  krylov_dim: int, random_seed: int):
    """One OFTLM run -> (E, C, S) arrays on grid ``T``."""
    from qed import _core

    opts = _core.ThermalOptions()
    opts.method      = _core.ThermalMethod.OFTLM
    opts.num_exact   = int(num_exact)
    opts.num_samples = int(num_samples)
    opts.krylov_dim  = int(krylov_dim)
    # Explicit inverse-temperature grid -> the result temperatures are exactly
    # 1/betas in the same order (the FTLM/OFTLM kernel preserves beta ordering).
    opts.betas       = [1.0 / float(t) for t in T]
    opts.random_seed = int(random_seed)

    res = _core.workflows_thermal(qop, opts)
    td  = res.thermo
    E = np.asarray(td.energy, dtype=np.float64)
    C = np.asarray(td.specific_heat, dtype=np.float64)
    S = np.asarray(td.entropy, dtype=np.float64)
    return E, C, S


def oftlm_thermodynamics(
    op,
    temperatures: np.ndarray,
    *,
    num_exact: int = 8,
    num_samples: int = 20,
    krylov_dim: int = 100,
    random_seed: int = 1,
    num_seeds: int = 2,
) -> ThermoResult:
    """OFTLM thermodynamics on the exact ``temperatures`` grid (full Hilbert
    space -- no Sz decomposition, so it works for Sz-broken clusters too and
    avoids the per-sector recombination).

    Resampled error estimate: the ``num_samples`` random-vector budget is
    SPLIT across ``num_seeds`` independent OFTLM runs (near cost-neutral --
    only the small exact-stage Lanczos repeats); the returned curves are
    the across-seed mean and ``std_error`` holds the standard error of
    that mean per quantity. NLCE weight subtraction amplifies a stochastic
    cluster's ABSOLUTE noise, so downstream kernels need this scale to
    judge whether a high-order weight is signal or noise (standard
    practice: Richter/Steinigeweg PRB 99, 144422; Schnack et al. PRR 2,
    013186 quantify FTLM accuracy via R-scaling).
    """
    T = np.asarray(temperatures, dtype=np.float64)
    # Runtime callers hand a native qed Operator directly (no middleman);
    # the oracle/test path may still pass a SpinHalfOperator.
    qop = spinhalf_to_qed(op) if isinstance(op, SpinHalfOperator) else op

    num_seeds = max(1, int(num_seeds))
    per_seed = max(1, int(num_samples) // num_seeds)

    Es, Cs, Ss = [], [], []
    for k in range(num_seeds):
        # Deterministic, well-separated seed stream.
        seed_k = (int(random_seed) + 0x9E3779B9 * (k + 1)) & 0x7FFFFFFF
        if seed_k == 0:
            seed_k = 1
        E, C, S = _oftlm_single(qop, T, num_exact, per_seed,
                                krylov_dim, seed_k)
        Es.append(E); Cs.append(C); Ss.append(S)

    E = np.mean(Es, axis=0)
    C = np.mean(Cs, axis=0)
    S = np.mean(Ss, axis=0)
    std_error = None
    if num_seeds >= 2:
        f = 1.0 / np.sqrt(num_seeds)
        std_error = {
            "energy":        np.std(Es, axis=0, ddof=1) * f,
            "specific_heat": np.std(Cs, axis=0, ddof=1) * f,
            "entropy":       np.std(Ss, axis=0, ddof=1) * f,
        }

    F = E - T * S
    with np.errstate(divide="ignore", invalid="ignore"):
        logZ = -F / T
    return ThermoResult(T, E, C, S, F, logZ, std_error=std_error)
