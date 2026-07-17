"""The pure-Python correctness ORACLE (tests only -- never runtime).

``dense.solve_spectrum`` (Sz / Sz-parity / spatial-abelian / spin-flip /
time-reversal blocked full spectra in numpy/scipy) and the ``symmetry``
utilities it rests on were the original QED_NLCE production engine; the
runtime now goes entirely through the ``qed`` C++ library
(:mod:`qed_nlce.ed.engine`), and these modules remain solely so the
parity suite can pin that engine against an independent implementation
to machine precision.
"""

from .dense import solve_spectrum, SymmetryReport  # noqa: F401
from .symmetry import (  # noqa: F401
    AbelianGroup,
    act_state,
    build_symmetry_group,
    detect_spin_flip,
    detect_time_reversal,
    find_automorphisms,
    maximal_abelian_subgroup,
    permute_state,
)
