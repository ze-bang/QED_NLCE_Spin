"""ED option container shared by all NLCE pipelines.

This module used to host the ``./ED`` subprocess bridge
(``build_ed_command``, ``run_ed_subprocess``, binary discovery).
Those have been removed: every NLCE pipeline now runs ED in-process
through :mod:`qed_nlce.core.dense_ed`, which dispatches each
cluster through the canonical ``qed.solve(H, ...)`` /
``qed.thermal(H, ...)`` Python verbs. What remains is the
:class:`EDOptions` dataclass that pipelines populate per cluster and
that the in-process backend reads when building those calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


__all__ = ["EDOptions"]


@dataclass
class EDOptions:
    """Per-cluster ED knobs forwarded into one in-process ``qed`` call.

    Pipeline classes construct an ``EDOptions`` per cluster from the
    user's CLI args; the in-process backend translates it into kwargs
    for :func:`qed.solve` (ground-state methods) or :func:`qed.thermal`
    (FTLM / LTLM / KPM_DOS / mTPQ / cTPQ) and dispatches in-process.
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
    # Large-cluster OFTLM fallback: clusters whose full Hilbert dimension exceeds
    # ``oftlm_cutoff`` are handled by matrix-free OFTLM (QED) instead of dense ED.
    oftlm_cutoff: int = 1 << 15          # 32768 (~15-site full space)
    oftlm_num_exact: int = 16            # N_V low-lying states treated exactly
    oftlm_num_samples: int = 20          # R random samples
    oftlm_krylov_dim: int = 100          # Lanczos steps per sample
    extra_flags: list[str] = field(default_factory=list)
