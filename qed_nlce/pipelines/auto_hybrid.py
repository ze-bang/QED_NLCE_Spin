"""Auto-hybrid NLCE pipeline.

A drop-in default for users who don't want to think about FULL vs.
FTLM. The crossover is keyed on the *full* Hilbert-space dimension
(``2 ** num_sites``) rather than a site count, which is the right
quantity for memory and runtime planning:

* ``2 ** num_sites <= --auto_full_hilbert`` (default 2**16 = 65536)
  -> FULL diagonalization with all eigenvalues. Noise-free, exact.
* otherwise -> FTLM with adaptively tuned ``samples`` and
  ``krylov_dim``: more samples for the smallest "large" clusters
  (where statistical noise dominates) and a Krylov dimension scaled
  to the spectral spread.

Summation routes through ``NLC_sum_ftlm.py``, which already handles
mixed FULL + FTLM input (the FULL eigenvalue files are written in the
FTLM-compatible HDF5 schema by the qed backend).

Inherits :class:`qed_nlce.pipelines.ftlm.FTLMPipeline` so the
NLC-summation command, robust-pipeline switch, GPU support, and
SI-unit conversion all come along for free.
"""

from __future__ import annotations

import argparse

from ..core import EDOptions, register_pipeline
from .ftlm import FTLMPipeline


@register_pipeline
class AutoHybridPipeline(FTLMPipeline):
    name = "auto"
    description = (
        "Auto-hybrid: FULL ED for small Hilbert spaces, FTLM for large, "
        "with Krylov dim and sample count auto-tuned to cluster size."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Inherit FTLM's flags first, then layer the auto-specific ones.
        super().add_arguments(parser)
        g = parser.add_argument_group("auto pipeline")
        g.add_argument(
            "--auto_full_hilbert", type=int, default=1 << 16,
            help="FULL ED ceiling on full Hilbert-space dim (default 65536 = 2**16)",
        )
        g.add_argument(
            "--auto_min_samples", type=int, default=40,
            help="Minimum FTLM samples for the largest clusters (default 40)",
        )
        g.add_argument(
            "--auto_max_samples", type=int, default=200,
            help="Maximum FTLM samples for the smallest 'large' clusters (default 200)",
        )

    def make_ed_options(self, args: argparse.Namespace, num_sites: int) -> EDOptions:
        hilbert_dim = 2 ** num_sites
        full_ceiling = getattr(args, "auto_full_hilbert", 1 << 16)

        # Below the crossover: FULL ED (exact).
        if hilbert_dim <= full_ceiling:
            return EDOptions(
                method="FULL",
                eigenvalues="FULL",
                thermo=True,  # NLC_sum_ftlm needs the thermo block
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                temp_bins=args.temp_bins,
                symmetrized=getattr(args, "symmetrized", False),
                use_symm=False,
            )

        # Above the crossover: FTLM with adaptively tuned parameters.
        # Krylov dim ~ log2(hilbert) * 20, capped by hilbert_dim itself
        # and by the user's --krylov_dim ceiling.
        krylov_ceiling = getattr(args, "krylov_dim", 300)
        adaptive_krylov = min(
            hilbert_dim,
            max(int(num_sites * 20), krylov_ceiling),
        )

        # Sample count: more for the smallest "large" clusters (where
        # ratio of FTLM samples to full Hilbert dim matters most),
        # ramping down to --auto_min_samples for the largest.
        min_s = getattr(args, "auto_min_samples", 40)
        max_s = getattr(args, "auto_max_samples", 200)
        # Decay: each doubling of hilbert_dim past the ceiling halves
        # the excess sample budget.
        if hilbert_dim <= 2 * full_ceiling:
            adaptive_samples = max_s
        else:
            import math
            decay = math.log2(hilbert_dim / full_ceiling)
            adaptive_samples = int(max(min_s, max_s / max(decay, 1.0)))

        return EDOptions(
            method="FTLM_GPU" if getattr(args, "use_gpu", False) else "FTLM",
            eigenvalues=None,
            thermo=True,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_bins=args.temp_bins,
            symmetrized=getattr(args, "symmetrized", False),
            use_symm=False,
            samples=adaptive_samples,
            krylov_dim=adaptive_krylov,
        )
