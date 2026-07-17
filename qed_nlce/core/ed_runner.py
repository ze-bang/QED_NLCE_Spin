"""ED option container shared by all NLCE pipelines.

This module used to host the ``./ED`` subprocess bridge
(``build_ed_command``, ``run_ed_subprocess``, binary discovery).
Those have been removed: every NLCE pipeline now runs ED in-process
through :mod:`qed_nlce.core.dense_ed`, which dispatches each cluster
to ``qed.full_spectrum`` (exact, symmetry-adapted -- abelian and
non-abelian) below ``oftlm_cutoff``, or to matrix-free OFTLM
(:func:`qed_nlce.ed.oftlm.oftlm_thermodynamics`) above it. What
remains is the :class:`EDOptions` dataclass that pipelines populate
per cluster and that the in-process backend reads when dispatching.
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
    # ``oftlm_cutoff`` are handled by matrix-free OFTLM (QED) instead of the
    # exact qed.full_spectrum tier (which applies abelian + non-abelian
    # symmetry reduction, so it scales well past the raw-dimension naive limit).
    oftlm_cutoff: int = 1 << 18          # 262144 (~18-site full space)
    oftlm_num_exact: int = 16            # N_V low-lying states treated exactly
    oftlm_num_samples: int = 20          # R random samples (split across seeds)
    oftlm_krylov_dim: int = 100          # Lanczos steps per sample
    oftlm_num_seeds: int = 2             # independent seeds -> std_error bands
    # Block-aware exact-tier gate: above oftlm_cutoff a cluster still
    # solves EXACTLY when its largest symmetry block (Sz sector / |G_ab|,
    # real when TR holds) is below this AND fits available RAM.
    exact_max_block: int = 120_000
    # ... AND its largest raw Sz(-parity) sector is below this. The old
    # serial column-crawl scaled with the sector (6.5 h+ at C(22,11)=705k
    # vs 147 s at 92k); the QED rep-walk assembler (6699a42) retired that
    # cliff, and a 22-site order-7 cluster (sector 705432) now solves
    # exactly in 3.65 h (verify_order7_exact.py, 2026-07-06). Cap raised
    # 200k -> 800k to admit order 7; order 8 (C(25,12)=5.2M) still excluded.
    exact_max_sector: int = 800_000
    # EXACT-ONLY routing (the default): every cluster runs the exact
    # full-spectrum tier. NLCE weight subtraction amplifies any stochastic
    # error by ~(T/J)^-order, so OFTLM results at deep orders are noise --
    # when a cluster exceeds the exact-tier caps the runner now WARNS
    # (job may run very long / exhaust memory) and solves exactly anyway.
    # Set True to restore the stochastic OFTLM fallback for over-cap
    # clusters (error bands are still propagated when it runs).
    oftlm_fallback: bool = False
    device: str = "cpu"                  # qed.full_spectrum backend device
    # qed point-group routing for the exact tier: "auto" (default)
    # projects through the factorized little-group lane where it accepts
    # and falls back to the abelian rep lane with star/TR/flip folds;
    # "off" keeps the abelian lane; "full" requires projection (raises
    # with the decline reason). Forwarded to qed.full_spectrum.
    point_group: str = "auto"
    extra_flags: list[str] = field(default_factory=list)
