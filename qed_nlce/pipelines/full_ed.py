"""Full dense ED pipeline.

For each cluster, computes the full eigenvalue spectrum in-process via
the self-contained symmetry-adapted dense solver
(:func:`qed_nlce.ed.solve_spectrum`).

Summation kernel:

* Pyrochlore -> ``NLC_sum.py``
* Triangular -> ``NLC_sum_triangular.py`` (with the triangular-specific
  ``--temp_points_file`` / ``--resummation`` knobs forwarded).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core import EDOptions, Pipeline, register_pipeline
from ..geometries import _paths


@register_pipeline
class FullEDPipeline(Pipeline):
    name = "full_ed"
    description = "Full dense ED via the in-process symmetry-adapted solver."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("full_ed pipeline")
        g.add_argument("--method", type=str, default="FULL",
                       help=argparse.SUPPRESS)
        # Back-compat: legacy ScaLAPACK knobs are no longer used; kept
        # as silent argparse aliases so existing CLI lines do not break.
        g.add_argument("--scalapack_threshold", type=int, default=16,
                       help=argparse.SUPPRESS)
        g.add_argument("--no_scalapack", action="store_true",
                       help=argparse.SUPPRESS)
        g.add_argument("--symmetrized", action="store_true",
                       help="Use --symmetrized (stronger guarantee than --symm)")
        g.add_argument("--measure_spin", action="store_true",
                       help="Forward --measure_spin to ED")
        g.add_argument("--SI_units", action="store_true",
                       help="Convert NLCE-summation output to SI units (J/(mol·K) etc.)")
        g.add_argument("--resummation", type=str, default=None,
                       help="Resummation method for the triangular summation kernel "
                            "(default: 'auto' for pyrochlore, 'euler' for triangular)")
        g.add_argument("--temp_points_file", type=str, default=None,
                       help="File of explicit temperature points (triangular summation only)")
        g.add_argument("--oftlm_cutoff", type=int, default=1 << 18,
                       help="Full Hilbert-dim (2**num_sites, BEFORE symmetry "
                            "reduction) above which a cluster is routed to "
                            "stochastic OFTLM instead of the exact "
                            "qed.full_spectrum backend (default 262144 = 2^18). "
                            "The exact backend applies abelian AND non-abelian "
                            "spatial symmetry, U(1)-Sz, spin-flip, and "
                            "time-reversal reduction, so many clusters well "
                            "above this raw threshold still solve quickly -- "
                            "watch the per-cluster 'qed-bridge' log lines "
                            "(eigenvalue count + wall time) to judge whether "
                            "raising this further is safe for your clusters. "
                            "A cluster with a near-trivial automorphism group "
                            "gets little reduction, so raising this too far "
                            "risks an effectively-brute-force dense diagonalization.")
        g.add_argument("--oftlm_num_exact", type=int, default=16,
                       help="OFTLM: number of lowest eigenstates treated exactly. "
                            "Must be large enough to cover the thermally-relevant "
                            "low-energy states at the lowest T, or NLCE weights on "
                            "large clusters are corrupted (default 16).")
        g.add_argument("--oftlm_num_samples", type=int, default=20,
                       help="OFTLM: number of random vectors (FTLM trace estimate); "
                            "statistical error ~1/sqrt(R) (default 20).")
        g.add_argument("--oftlm_krylov_dim", type=int, default=100,
                       help="OFTLM: Lanczos/Krylov dimension per random vector "
                            "(default 100).")
        g.add_argument("--exact_max_block", type=int, default=120_000,
                       help="Largest symmetry-block dimension the exact tier "
                            "will dense-diagonalize (default 120000). Above "
                            "--oftlm_cutoff a cluster is STILL solved exactly "
                            "when its largest block (Sz sector / |Aut|, real "
                            "arithmetic when time-reversal holds) is below "
                            "this and fits in RAM -- e.g. 19-site XXZ "
                            "pyrochlore order-6 trees. OFTLM is a last "
                            "resort: NLCE weight cancellation at deep orders "
                            "amplifies any stochastic noise.")
        g.add_argument("--exact_max_sector", type=int, default=800_000,
                       help="Largest raw Sz(-parity) sector the exact tier "
                            "accepts (default 800000, admits order 7). The "
                            "upstream rep-walk dense assembler (QED 6699a42) "
                            "removed the old serial-column-build cliff: a "
                            "22-site order-7 cluster (C(22,11)=705432) was "
                            "verified to solve exactly in 3.65 h on a cluster "
                            "node (scripts/verify_order7_exact.py, 2026-07-06; "
                            "4,194,304 = 2^22 eigenvalues, E0=-7.487). Order 8 "
                            "(C(25,12)=5.2M) stays excluded pending its own "
                            "timing.")
        g.add_argument("--oftlm_num_seeds", type=int, default=2,
                       help="OFTLM: number of independent seeds the sample "
                            "budget is split across (default 2). >= 2 yields "
                            "per-cluster std_error bands that the summation "
                            "propagates into NLCE error estimates; 1 disables "
                            "error tracking (not recommended for order >= 6).")

    # ------------------------------------------------------------ ED config

    def make_ed_options(self, args: argparse.Namespace, num_sites: int) -> EDOptions:
        # symm-threshold is a triangular concept; on pyrochlore it's just always-on.
        symm_threshold = getattr(args, "symm_threshold", -1)
        use_symm = (num_sites > symm_threshold) if symm_threshold > 0 else True
        streaming_symmetry = getattr(args, "streaming_symmetry", False)
        basis_cache_dir = None
        if streaming_symmetry:
            import os
            basis_cache_dir = os.path.join(
                args.base_dir,
                f"hamiltonians_order_{args.max_order}",
                "_unused_placeholder",  # workflow rewrites via ham_subdir
            )
            # Real value is computed below when build_ed_command sees ham_subdir;
            # to keep things simple, we rely on the per-cluster ham_subdir convention
            # used by NLCEWorkflow + the build_ed_command basis_cache_dir guard.
            basis_cache_dir = None  # let workflow plumb it in via extra_flags if needed

        return EDOptions(
            method=getattr(args, "method", "FULL"),
            eigenvalues="FULL",
            thermo=getattr(args, "thermo", False),
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_bins=args.temp_bins,
            measure_spin=getattr(args, "measure_spin", False),
            symmetrized=getattr(args, "symmetrized", False),
            use_symm=use_symm and not getattr(args, "symmetrized", False),
            streaming_symmetry=streaming_symmetry,
            basis_cache_dir=None,
            oftlm_cutoff=getattr(args, "oftlm_cutoff", 1 << 18),
            oftlm_num_exact=getattr(args, "oftlm_num_exact", 16),
            oftlm_num_samples=getattr(args, "oftlm_num_samples", 20),
            oftlm_krylov_dim=getattr(args, "oftlm_krylov_dim", 100),
            oftlm_num_seeds=getattr(args, "oftlm_num_seeds", 2),
            exact_max_block=getattr(args, "exact_max_block", 120_000),
            exact_max_sector=getattr(args, "exact_max_sector", 800_000),
            device=getattr(args, "device", "cpu"),
        )

    def needs_thermo(self, args: argparse.Namespace) -> bool:
        return False  # thermo is opt-in via the user --thermo

    def extra_env(self, args, num_sites):
        # Triangular geometries thrash on small clusters with many threads.
        if (
            getattr(args, "geometry", "").startswith("triangular")
            and num_sites <= 8
        ):
            return {"OMP_NUM_THREADS": "1"}
        return None

    # --------------------------------------------------------- summation

    def summation_command(
        self,
        args: argparse.Namespace,
        cluster_info_dir: str,
        ed_dir: str,
        nlc_dir: str,
        order_cutoff: int,
    ) -> Optional[list[str]]:
        geometry_name = getattr(args, "geometry", "")

        if geometry_name.startswith("triangular"):
            resummation = args.resummation or "euler"
            cmd = [
                sys.executable,
                _paths.NLC_SUM_TRIANGULAR,
                f"--cluster_dir={cluster_info_dir}",
                f"--eigenvalue_dir={ed_dir}",
                f"--output_dir={nlc_dir}",
                f"--max_order={order_cutoff}",
                f"--resummation={resummation}",
            ]
            if getattr(args, "temp_points_file", None):
                cmd.append(f"--temp_points_file={args.temp_points_file}")
            else:
                cmd += [
                    f"--temp_min={args.temp_min}",
                    f"--temp_max={args.temp_max}",
                    f"--temp_bins={args.temp_bins}",
                ]
            if getattr(args, "measure_spin", False):
                cmd.append("--measure_spin")
            if getattr(args, "SI_units", False):
                cmd.append("--SI_units")
            return cmd

        # default: pyrochlore (NLC_sum.py)
        resummation = args.resummation or "auto"
        cmd = [
            sys.executable,
            _paths.NLC_SUM_FULL,
            f"--cluster_dir={cluster_info_dir}",
            f"--eigenvalue_dir={ed_dir}",
            f"--output_dir={nlc_dir}",
            "--plot",
            f"--temp_min={args.temp_min}",
            f"--temp_max={args.temp_max}",
            f"--temp_bins={args.temp_bins}",
            f"--resummation_method={resummation}",
        ]
        if getattr(args, "SI_units", False):
            cmd.append("--SI_units")
        if order_cutoff and order_cutoff != args.max_order:
            cmd.append(f"--order_cutoff={order_cutoff}")
        if getattr(args, "measure_spin", False):
            cmd.append("--measure_spin")
        return cmd
