"""Exact-diagonalization layer for QED_NLCE.

The RUNTIME path routes through the ``qed`` C++ package:

* :func:`full_spectrum_qed` -- exact full spectrum via
  ``qed.full_spectrum`` (abelian AND non-abelian spatial symmetry,
  U(1) S^z, spin-flip, time-reversal reduction). Primary tier.
* :func:`oftlm_thermodynamics` -- matrix-free stochastic OFTLM for
  clusters above the exact tier's cutoff.
* :func:`thermodynamics` -- energy / specific heat / entropy / free
  energy from a full spectrum (solver-agnostic).

The pure-Python solver is retained as the correctness ORACLE (not on
the runtime path): :func:`solve_spectrum` (numpy/scipy, abelian-only
symmetry) plus the symmetry utilities. ``tests/test_qed_bridge_parity.py``
pins ``full_spectrum_qed == solve_spectrum`` on every model class.
"""

from .operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from .dense import solve_spectrum, SymmetryReport
from .thermo import thermodynamics, ThermoResult
from .oftlm import oftlm_thermodynamics, spinhalf_to_qed
from .qed_bridge import full_spectrum_qed
from .symmetry import (
    find_automorphisms,
    detect_spin_flip,
    build_symmetry_group,
    maximal_abelian_subgroup,
    AbelianGroup,
    permute_state,
    act_state,
)

__all__ = [
    "SpinHalfOperator",
    "OP_SP",
    "OP_SM",
    "OP_SZ",
    "solve_spectrum",
    "SymmetryReport",
    "thermodynamics",
    "ThermoResult",
    "oftlm_thermodynamics",
    "spinhalf_to_qed",
    "full_spectrum_qed",
    "find_automorphisms",
    "detect_spin_flip",
    "build_symmetry_group",
    "maximal_abelian_subgroup",
    "AbelianGroup",
    "permute_state",
    "act_state",
]
