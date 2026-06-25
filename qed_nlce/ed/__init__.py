"""Self-contained dense exact-diagonalization core for QED_NLCE.

Pure numpy/scipy. No external ``qed`` / ``edlib`` dependency. Provides:

* :class:`SpinHalfOperator` -- in-memory spin-1/2 Hamiltonian.
* :func:`solve_spectrum` -- symmetry-adapted dense diagonalization
  (U(1) S^z sectors + spatial automorphism orbit basis + real/TR
  reduction) returning the full eigenvalue spectrum via LAPACK.
* :func:`thermodynamics` -- energy / specific heat / entropy / free
  energy from a spectrum.
* symmetry utilities (:func:`find_automorphisms`,
  :func:`maximal_abelian_subgroup`).
"""

from .operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from .dense import solve_spectrum, SymmetryReport
from .thermo import thermodynamics, ThermoResult
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
    "find_automorphisms",
    "detect_spin_flip",
    "build_symmetry_group",
    "maximal_abelian_subgroup",
    "AbelianGroup",
    "permute_state",
    "act_state",
]
