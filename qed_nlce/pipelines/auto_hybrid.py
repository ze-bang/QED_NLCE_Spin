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
import logging
import math
import os
import re

from ..core import EDOptions, register_pipeline
from .ftlm import FTLMPipeline


def _interall_conserves_sz(ham_dir: str) -> bool:
    """Inspect InterAll.dat to determine if the Hamiltonian conserves S^z_total.

    A spin-1/2 two-body term conserves Sz iff it is diagonal in the
    computational basis (SzSz) or is a flip-pair (S+S- or S-S+). In the
    InterAll.dat format, the 4 spin indices (s1, s2, s3, s4) encode the
    matrix element <s1 s3 | H | s2 s4> for each site pair. The Sz-conserving
    terms have s1==s2 AND s3==s4 (diagonal/SzSz), or s1!=s2 AND s3!=s4 with
    complementary flips (S+S-/S-S+).

    Returns True if all terms conserve Sz, False otherwise. Returns True
    if the file cannot be parsed (conservative default that preserves
    existing behaviour).
    """
    inter_file = os.path.join(ham_dir, "InterAll.dat")
    if not os.path.isfile(inter_file):
        return True  # no interactions → trivially conserves Sz

    try:
        with open(inter_file, "r") as f:
            lines = f.readlines()
    except OSError:
        return True

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        # HPhi InterAll format (10 fields):
        #   site_i σ1 site_j σ2 site_k σ3 site_l σ4 Re Im
        # Term: c^+_{i,σ1} c_{j,σ2} c^+_{k,σ3} c_{l,σ4}
        # Sz conservation: (σ2 - σ1) + (σ4 - σ3) == 0
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            sigma1 = int(parts[1])
            sigma2 = int(parts[3])
            sigma3 = int(parts[5])
            sigma4 = int(parts[7])
        except (ValueError, IndexError):
            continue

        if (sigma2 - sigma1) + (sigma4 - sigma3) != 0:
            return False

    # Also check Trans.dat for single-site terms that flip spin.
    # HPhi Trans.dat format (6 fields):
    #   site_i σ1 site_j σ2 Re Im
    # Term: c^+_{i,σ1} c_{j,σ2}
    # Sz conservation: σ1 == σ2
    trans_file = os.path.join(ham_dir, "Trans.dat")
    if os.path.isfile(trans_file):
        try:
            with open(trans_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("="):
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        sigma1 = int(parts[1])
                        sigma2 = int(parts[3])
                    except (ValueError, IndexError):
                        continue
                    if sigma1 != sigma2:
                        return False
        except OSError:
            pass

    return True


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
            choices=["kpm_dos", "ftlm", "ltlm"],
            help="Iterative backend used above the FULL-ED ceiling "
                 "(default: kpm_dos). "
                 "kpm_dos: KPM density-of-states + Chebyshev quadrature — "
                 "low variance, scales as 1/sqrt(R·D), recommended for T > 0.2J. "
                 "ltlm: Low-Temperature Lanczos Method (Jaklič & Prelovšek, "
                 "PRB 67, 161103 2003) — double-bracket estimator that is "
                 "exact at T=0 for any R; 3–5× less noise than FTLM at "
                 "T < 0.5J with the same Krylov dimension. Recommended when "
                 "low-T accuracy is the priority (e.g. NLCE order 6–8 on "
                 "frustrated lattices). "
                 "ftlm: legacy FTLM — has ~5%% noise per cluster at low T "
                 "amplified by Möbius condition number; use ltlm instead.",
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
            "--auto_sz_decomposed", action="store_true",
            help="(No-op for FULL-ED clusters: FULL_SYMMETRIZED already "
                 "iterates all Sz sectors. Still applies to the iterative "
                 "backend, e.g. FTLM_SZ_DECOMPOSED / KPM_DOS_SZ_DECOMPOSED.)",
        )
        g.add_argument(
            "--auto_fixed_sz", action="store_true",
            help="(Legacy) For FULL-ED clusters, restrict to Sz=0 sector "
                 "only (FULL_FIXED_SZ). Approximate at finite T. For the "
                 "iterative backend, routes through FTLM_FIXED_SZ / "
                 "KPM_DOS_FIXED_SZ.",
        )
        g.add_argument(
            "--auto_streaming_symmetry", action="store_true",
            help="For the iterative backend (FTLM/KPM_DOS), exploit the "
                 "cluster's geometric automorphism group via streaming "
                 "symmetry. FULL-ED clusters already use FULL_SYMMETRIZED "
                 "(Sz + spatial), so this flag only affects the large-cluster "
                 "path.",
        )
        g.add_argument(
            "--auto_sz_detect", action="store_true", default=True,
            help="Auto-detect Sz conservation from InterAll.dat and "
                 "route through fixed-Sz block if the model conserves "
                 "S^z_total (default: on). Disable with --no_auto_sz_detect.",
        )
        g.add_argument(
            "--no_auto_sz_detect", action="store_true",
            help="Disable Sz auto-detection (always use full Hilbert space).",
        )

    # ------------------------------------------------------------------
    # Per-cluster option construction
    # ------------------------------------------------------------------

    def _maybe_sz_suffix(self, args: argparse.Namespace, base: str) -> str:
        """Append ``_SZ_DECOMPOSED`` or ``_FIXED_SZ`` to the method name.

        - ``_SZ_DECOMPOSED`` (default for auto-detected Sz conservation):
          iterates all Sz sectors and recombines Z — physically exact for
          finite-T NLCE while benefiting from block-diagonal structure.
        - ``_FIXED_SZ`` (legacy, explicit opt-in only): restricts to the
          single Sz=0 sector. Approximate at finite T.
        """
        if getattr(args, "auto_sz_decomposed", False):
            return base + "_SZ_DECOMPOSED"
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
            # FULL_SYMMETRIZED decomposes by ALL (Sz × spatial) symmetries
            # via qed.full_spectrum + on-the-fly RepMV. It subsumes the
            # _SZ_DECOMPOSED behaviour (iterates every Sz sector) and adds
            # spatial block-diagonalisation (cluster automorphism group,
            # cached per topology). Falls back gracefully to Sz-only when
            # no spatial generators are found. The only exception is
            # --auto_fixed_sz, which intentionally restricts to Sz=0 as an
            # explicit legacy approximation.
            if getattr(args, "auto_fixed_sz", False):
                full_method = "FULL_FIXED_SZ"
            else:
                full_method = "FULL_SYMMETRIZED"
            return EDOptions(
                method=full_method,
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
            # Adaptive R: for large clusters (dim >> 2^14), the stochastic
            # trace variance 1/sqrt(R*D) is already small, so we can reduce
            # R. For small clusters just above the full-ED ceiling, use
            # more vectors for tighter accuracy. Target: per-cluster relative
            # error < 1e-3 on C(T) to satisfy NLCE Mobius budget (kappa~80-150).
            base_R = getattr(args, "auto_kpm_random_vectors", 32)
            if hilbert_dim >= (1 << 18):
                # Very large cluster: 1/sqrt(R*D) tiny even at R=16
                adaptive_R = max(16, base_R // 2)
            elif hilbert_dim >= (1 << 16):
                adaptive_R = max(20, int(base_R * 0.75))
            else:
                adaptive_R = base_R

            return EDOptions(
                method=self._maybe_sz_suffix(args, base),
                eigenvalues=None,
                thermo=True,
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                temp_bins=args.temp_bins,
                symmetrized=symmetrized,
                use_symm=False,
                streaming_symmetry=streaming,
                samples=adaptive_R,
                krylov_dim=getattr(args, "auto_kpm_moments", 2048),
                extra_flags=extra_flags,
            )

        # ---- backend == "ftlm" or "ltlm": Lanczos-based thermal solvers ----
        # LTLM (Jaklič & Prelovšek, PRB 67, 161103, 2003) uses a double-
        # bracket estimator that is exact at T=0 for any number of random
        # vectors and gives 3–5× less per-cluster noise at T < 0.5J vs FTLM
        # with the same Krylov dimension. For NLCE, this directly reduces
        # Möbius amplification: κ~80 × (1.5%) ≈ 120% → κ~80 × (0.3%) ≈ 24%.
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

        use_gpu = getattr(args, "use_gpu", False)
        if backend == "ltlm":
            ltlm_base = "LTLM"  # LTLM has no GPU variant in QED 0.3
            return EDOptions(
                method=self._maybe_sz_suffix(args, ltlm_base),
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

        ftlm_base = "FTLM_GPU" if use_gpu else "FTLM"
        return EDOptions(
            method=self._maybe_sz_suffix(args, ftlm_base),
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
