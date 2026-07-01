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


def oftlm_thermodynamics(
    op: SpinHalfOperator,
    temperatures: np.ndarray,
    *,
    num_exact: int = 8,
    num_samples: int = 20,
    krylov_dim: int = 100,
    random_seed: int = 1,
) -> ThermoResult:
    """OFTLM thermodynamics on the exact ``temperatures`` grid (full Hilbert
    space -- no Sz decomposition, so it works for Sz-broken clusters too and
    avoids the per-sector recombination).
    """
    from qed import _core

    T = np.asarray(temperatures, dtype=np.float64)
    qop = spinhalf_to_qed(op)

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
    F = (np.asarray(td.free_energy, dtype=np.float64)
         if getattr(td, "free_energy", None) is not None and len(td.free_energy)
         else E - T * S)
    with np.errstate(divide="ignore", invalid="ignore"):
        logZ = -F / T
    return ThermoResult(T, E, C, S, F, logZ)
