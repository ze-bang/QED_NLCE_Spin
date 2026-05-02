"""Full / dense ED pipeline.

For each cluster, runs ``--method=FULL`` (or ``FULL_GPU`` if the user
asks for it) in-process via the qed Python bindings.

Summation kernel:

* Pyrochlore -> ``NLC_sum.py``
* Triangular -> ``NLC_sum_triangular.py`` (with the triangular-specific
  ``--temp_points_file`` / ``--resummation`` knobs forwarded).

Note: the legacy ``SCALAPACK_MIXED`` auto-promotion has been removed.
MPI-only methods cannot run in the in-process qed backend (a Python
interpreter cannot call ``MPI_Init``); use the standalone
``ed_distributed_main`` binary directly for those.
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
    description = "Full dense ED via the in-process qed backend."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("full_ed pipeline")
        g.add_argument("--method", type=str, default="FULL",
                       help="ED method (FULL, FULL_GPU, LANCZOS, ...). "
                            "MPI-only methods (SCALAPACK*, mTPQ_MPI) are "
                            "rejected by the in-process backend.")
        # Back-compat: legacy ScaLAPACK auto-promotion knobs are no
        # longer used (the in-process backend cannot host MPI). Kept
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
