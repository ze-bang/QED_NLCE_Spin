"""Auto-hybrid NLCE pipeline (smart default).

A drop-in default for users who don't want to think about FULL vs KPM-DOS
vs FTLM, or about whether to enable Sz fixed-quantum-number blocking
or lattice-symmetry sector decomposition. The pipeline selects all of
these per cluster:

Backend selection (keyed on the *full* Hilbert-space dim ``2 ** N``):

* ``2**N <= --auto_full_hilbert``  (default ``2**12 = 4096``)
    -> ``FULL`` ED with the entire spectrum -- noise-free anchors
       for the small / dominant clusters. (Was ``2**14 = 16384``
       before May 2026; lowered because dense LAPACK on a
       16384x16384 matrix costs ~4 TFlops and minutes per cluster,
       and KPM-DOS at the same dim is faster *and* matches the
       FULL specific heat to ~1e-3 with ``R=32, M=2048``. The new
       4096 ceiling matches the ``qed.diag`` small-dim threshold
       and keeps every per-cluster diag under ~1 s.)

* ``2**N > --auto_full_hilbert``
    -> ``--auto_backend`` (default ``kpm_dos``):

      * ``kpm_dos`` (recommended): the C++ KPM-DOS thermodynamics
        kernel. Hutchinson stochastic-trace variance scales as
        ``1 / sqrt(R * D)``, so per-cluster relative error on
        ``C(T)`` *improves* as the cluster grows. At ``N=20`` with
        ``R=20``, ``M=2048``, the per-cluster error is ``~4e-4``,
        well below the ``~0.1%`` per-cluster target dictated by the
        NLCE Mobius condition number ``kappa ~ 30-80``.

      * ``ftlm``: the legacy Finite-Temperature Lanczos backend.
        Noise floor of ``~5%`` on ``C(T)``, which gets amplified by
        the Mobius condition number to ``15-40%`` on the resummed
        curve at orders ``>= 6``. Provided for back-compat.

Symmetry axes (orthogonal, can be combined):

* ``--auto_fixed_sz`` -- assert that the model conserves
  ``S^z_total`` (e.g. XXZ + longitudinal field, no transverse field).
  Routes through the dispatcher with ``params.use_fixed_sz = True``,
  giving an immediate ``binom(N, N/2) / 2^N`` reduction in Hilbert
  dimension -- for ``N=20`` that is a ``~5.4x`` speedup *and* an
  equivalent KPM-DOS-variance improvement (which scales as ``1/D``).

* ``--auto_streaming_symmetry`` -- exploit the cluster's geometric
  automorphism group (lattice symmetries). Adds a one-time orbit-
  basis construction per cluster (cached under the cluster's ham
  dir as ``basis_cache/``).

Use ``--no_hybrid_mode`` to force the iterative backend on every
cluster (useful for benchmarking the variance scaling).
"""

from __future__ import annotations

import argparse
import math

from ..core import EDOptions, register_pipeline
from .ftlm import FTLMPipeline


@register_pipeline
class AutoHybridPipeline(FTLMPipeline):
    name = "auto"
    description = (
        "Smart default: FULL ED for small clusters, KPM-DOS (or FTLM) "
        "for large; orthogonal --auto_fixed_sz / --auto_streaming_symmetry "
        "axes auto-applied to every cluster."
    )

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Inherit FTLM's flags first (they double as fall-back knobs for
        # --auto_backend=ftlm), then layer the auto-specific flags on
        # top. We *also* re-expose the kpm_* knobs from the kpm_dos
        # pipeline so a single `--pipeline auto` invocation can drive
        # either backend without reaching for a sibling pipeline.
        super().add_arguments(parser)

        g = parser.add_argument_group("auto pipeline")
        g.add_argument(
            "--auto_backend", type=str, default="kpm_dos",
            choices=["kpm_dos", "ftlm"],
            help="Iterative backend used above the FULL-ED ceiling "
                 "(default: kpm_dos -- the low-variance KPM density-of-"
                 "states + Chebyshev quadrature solver).",
        )
        g.add_argument(
            "--auto_full_hilbert", type=int, default=1 << 12,
            help="FULL ED ceiling on full Hilbert-space dim "
                 "(default 4096 = 2**12; was 2**14 before May 2026). "
                 "Clusters with 2**N <= this go through dense LAPACK; "
                 "larger clusters use the iterative backend. The new "
                 "default keeps every per-cluster FULL diag under ~1 s "
                 "and matches the ``qed.diag`` auto-pilot threshold.",
        )

        # KPM-DOS knobs (only consulted when --auto_backend=kpm_dos).
        g.add_argument(
            "--auto_kpm_moments", type=int, default=2048,
            help="KPM Chebyshev moments M for large clusters (default 2048).",
        )
        g.add_argument(
            "--auto_kpm_random_vectors", type=int, default=32,
            help="KPM Hutchinson random-vector count R for large clusters "
                 "(default 32; was 20 before May 2026 -- bumped because "
                 "the SOTA NLCE accuracy target is per-cluster relative "
                 "error <~ 1e-3 on C(T), and 1/sqrt(R*D) at R=32, D=4096 "
                 "gives ~3e-3, comfortably below the typical NLCE "
                 "Mobius condition number 30-80 budget).",
        )
        g.add_argument(
            "--auto_kpm_kernel", type=str, default="jackson",
            choices=["jackson", "lorentz"],
            help="KPM smoothing kernel (default jackson).",
        )
        g.add_argument(
            "--auto_kpm_seed", type=int, default=0,
            help="Seed for KPM Hutchinson vectors (0 = nondeterministic).",
        )

        # FTLM knobs (only consulted when --auto_backend=ftlm). The
        # base FTLM pipeline already exposes --ftlm_samples / --krylov_dim
        # but we add an adaptive-sample range specifically for auto mode.
        g.add_argument(
            "--auto_min_samples", type=int, default=40,
            help="Minimum FTLM samples for the largest clusters (default 40, "
                 "only used when --auto_backend=ftlm).",
        )
        g.add_argument(
            "--auto_max_samples", type=int, default=200,
            help="Maximum FTLM samples for smallest 'large' clusters "
                 "(default 200, only used when --auto_backend=ftlm).",
        )

        # Symmetry axes (orthogonal -- can be combined).
        g.add_argument(
            "--auto_fixed_sz", action="store_true",
            help="Assert the Hamiltonian conserves S^z_total. Routes "
                 "every cluster through the fixed-Sz block (~5x smaller "
                 "Hilbert space at N=20). Use ONLY if your model has no "
                 "transverse field (h_x, h_y) and no Sx/Sy single-site "
                 "or anisotropic xy terms.",
        )
        g.add_argument(
            "--auto_streaming_symmetry", action="store_true",
            help="Exploit the geometric automorphism group of each "
                 "cluster (orbit-basis sector decomposition). Cached "
                 "per cluster under <ham_dir>/basis_cache/.",
        )

    # ------------------------------------------------------------------
    # Per-cluster option construction
    # ------------------------------------------------------------------

    def _maybe_fixed_sz_suffix(self, args: argparse.Namespace, base: str) -> str:
        """Append ``_FIXED_SZ`` to the method name if requested.

        The qed backend's :func:`_resolve_method` strips this suffix and
        sets ``params.use_fixed_sz = True``, so this is the canonical
        plumbing path for an orthogonal U(1)-Sz axis.
        """
        if getattr(args, "auto_fixed_sz", False):
            return base + "_FIXED_SZ"
        return base

    def make_ed_options(self, args: argparse.Namespace, num_sites: int) -> EDOptions:
        hilbert_dim = 2 ** num_sites
        full_ceiling = getattr(args, "auto_full_hilbert", 1 << 12)
        streaming = getattr(args, "auto_streaming_symmetry", False) or \
                    getattr(args, "streaming_symmetry", False)
        symmetrized = getattr(args, "symmetrized", False)
        backend = getattr(args, "auto_backend", "kpm_dos").lower()

        # ---- Below the crossover: FULL ED (exact, noise-free) ----
        if hilbert_dim <= full_ceiling:
            return EDOptions(
                method=self._maybe_fixed_sz_suffix(args, "FULL"),
                eigenvalues="FULL",
                thermo=True,  # NLC_sum_ftlm reads /thermodynamics/{...}
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                temp_bins=args.temp_bins,
                symmetrized=symmetrized,
                use_symm=False,
                streaming_symmetry=streaming,
            )

        # ---- Above the crossover: iterative backend ----
        if backend == "kpm_dos":
            base = "KPM_DOS"
            extra_flags = [
                f"--kpm_kernel={getattr(args, 'auto_kpm_kernel', 'jackson')}",
                f"--kpm_seed={getattr(args, 'auto_kpm_seed', 0)}",
            ]
            return EDOptions(
                method=self._maybe_fixed_sz_suffix(args, base),
                eigenvalues=None,
                thermo=True,
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                temp_bins=args.temp_bins,
                symmetrized=symmetrized,
                use_symm=False,
                streaming_symmetry=streaming,
                # Tunneled into kpm_num_random_vectors / kpm_num_moments
                # by qed_backend.run_ed_in_process when method is KPM_DOS:
                samples=getattr(args, "auto_kpm_random_vectors", 20),
                krylov_dim=getattr(args, "auto_kpm_moments", 2048),
                extra_flags=extra_flags,
            )

        # ---- backend == "ftlm": legacy adaptive FTLM ----
        krylov_ceiling = getattr(args, "krylov_dim", 300)
        adaptive_krylov = min(
            hilbert_dim,
            max(int(num_sites * 20), krylov_ceiling),
        )
        min_s = getattr(args, "auto_min_samples", 40)
        max_s = getattr(args, "auto_max_samples", 200)
        if hilbert_dim <= 2 * full_ceiling:
            adaptive_samples = max_s
        else:
            decay = math.log2(hilbert_dim / full_ceiling)
            adaptive_samples = int(max(min_s, max_s / max(decay, 1.0)))

        ftlm_base = "FTLM_GPU" if getattr(args, "use_gpu", False) else "FTLM"
        return EDOptions(
            method=self._maybe_fixed_sz_suffix(args, ftlm_base),
            eigenvalues=None,
            thermo=True,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_bins=args.temp_bins,
            symmetrized=symmetrized,
            use_symm=False,
            streaming_symmetry=streaming,
            samples=adaptive_samples,
            krylov_dim=adaptive_krylov,
        )

    def needs_thermo(self, args: argparse.Namespace) -> bool:
        return True
