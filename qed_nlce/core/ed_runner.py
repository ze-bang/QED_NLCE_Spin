"""ED option container shared by all NLCE pipelines.

This module used to host the ``./ED`` subprocess bridge
(``build_ed_command``, ``run_ed_subprocess``, binary discovery).
Those have been removed: every NLCE pipeline now runs ED in-process
through :mod:`qed_nlce.core.qed_backend`, which calls the ``qed``
Python package directly. What remains is the :class:`EDOptions`
dataclass that pipelines populate per cluster and that the in-process
backend reads to construct an :class:`qed.EDParameters`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


__all__ = ["EDOptions"]


@dataclass
class EDOptions:
    """Per-cluster ED knobs forwarded into one in-process ``qed`` call.

    Pipeline classes construct an ``EDOptions`` per cluster from the
    user's CLI args; the in-process backend translates it into a
    :class:`qed.EDParameters` and dispatches via
    :func:`qed.exact_diagonalization_from_directory`.
    """

    method: str = "FULL"
    eigenvalues: Optional[str] = "FULL"
    spin_length: float = 0.5
    thermo: bool = False
    temp_min: float = 0.001
    temp_max: float = 20.0
    temp_bins: int = 100
    measure_spin: bool = False
    symmetrized: bool = False
    use_symm: bool = True
    streaming_symmetry: bool = False
    basis_cache_dir: Optional[str] = None
    samples: Optional[int] = None
    krylov_dim: Optional[int] = None
    extra_flags: list[str] = field(default_factory=list)
