"""Exact-diagonalization layer for QED_NLCE -- a thin library on the
``qed`` C++ package.

The RUNTIME path (see :mod:`qed_nlce.ed.engine`):

* :func:`resolve_cluster_symmetry` -- ONE symmetry resolution per
  cluster: the spatial GeneratorSet (abelian generators + the
  non-abelian residue as ``star_perms``, clique-budgeted upstream) and
  the conserved-quantity flags (U(1) S^z, native Sz-parity, spin-flip,
  time-reversal). Consumed by both the router and the solver.
* :func:`plan_exact_solve` -- block-aware feasibility of the exact
  tier (Burnside estimate, refined by the engine's actual
  ``plan_only`` star plan when borderline).
* :func:`full_spectrum` -- exact full spectrum via ONE
  ``qed.full_spectrum`` call carrying all of the above.
* :func:`oftlm_thermodynamics` -- matrix-free stochastic OFTLM for
  clusters above the exact tier's reach.
* :func:`thermodynamics` -- energy / specific heat / entropy / free
  energy from a full spectrum (solver-agnostic).
* :func:`read_qed_operator` -- ``Trans.dat`` / ``InterAll.dat`` straight
  into a native ``qed`` operator.

:class:`SpinHalfOperator` (+ the ``io`` writers) remains the in-memory
Hamiltonian builder used by the geometry generators and the test
oracle. The pure-Python solver itself lives in ``tests/oracle/`` --
it is not importable from the installed package and no runtime path
reaches it; ``tests/test_engine_parity.py`` pins the engine
against it to machine precision.
"""

from .operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from .io import read_operator, read_qed_operator, write_operator, write_site_info
from .thermo import thermodynamics, ThermoResult
from .oftlm import oftlm_thermodynamics, spinhalf_to_qed
from .engine import (
    ClusterSymmetry,
    ExactPlan,
    full_spectrum,
    plan_exact_solve,
    resolve_cluster_symmetry,
)

__all__ = [
    "SpinHalfOperator",
    "OP_SP",
    "OP_SM",
    "OP_SZ",
    "read_operator",
    "read_qed_operator",
    "write_operator",
    "write_site_info",
    "thermodynamics",
    "ThermoResult",
    "oftlm_thermodynamics",
    "spinhalf_to_qed",
    "ClusterSymmetry",
    "ExactPlan",
    "full_spectrum",
    "plan_exact_solve",
    "resolve_cluster_symmetry",
]
