"""KPM-DOS thermodynamics pipeline.

For each cluster, runs the C++ KPM-DOS kernel
(:cpp:func:`ed::kpm_dos::compute_kpm_dos`) which computes the density
of states via Chebyshev-moment expansion + Hutchinson stochastic trace,
then evaluates Z(beta), <H>(beta), C(beta), S(beta), F(beta) on the
requested temperature grid.

Why KPM-DOS for NLCE
--------------------
The Hutchinson stochastic-trace variance is

    sigma_C/C ~ 1/sqrt(R * D)

where R is the number of random vectors and D is the Hilbert space
dimension. So per-cluster accuracy *improves* as the cluster grows.
By contrast FTLM has a constant ~5% noise floor in C(T) that gets
amplified by the NLCE Mobius condition number (kappa ~ 30-80) into
~15-40% error on the resummed curve at orders >= 6. KPM-DOS at
N=20, R=20, M=2048 is predicted to give ~4e-4 per-cluster error
on C(T), well below the 0.1% per-cluster target.

Cost
----
R * M sparse matrix-vector applies, plus a one-time spectral-bound
Lanczos (~150 iterations). For the same wallclock as FTLM at
``--ftlm_samples=80 --krylov_dim=300``, KPM-DOS gives 2-3 orders of
magnitude lower variance.

For tiny clusters (``num_sites <= --hybrid_threshold``, default 10)
it uses full ED for noise-free anchors, identical to the FTLM hybrid
mode. The summation step reuses ``NLC_sum_ftlm.py`` since both
pipelines write the same ``/thermodynamics/{...}`` HDF5 schema.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core import EDOptions, Pipeline, register_pipeline
from ..geometries import _paths


@register_pipeline
class KPMDOSPipeline(Pipeline):
    name = "kpm_dos"
    description = (
        "Kernel Polynomial Method density-of-states + thermodynamics "
        "(low-variance NLCE backend, recommended for orders >= 6)."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("kpm_dos pipeline")
        g.add_argument("--kpm_moments", type=int, default=2048,
                       help="Chebyshev moments M (default 2048)")
        g.add_argument("--kpm_random_vectors", type=int, default=20,
                       help="Hutchinson random vectors R (default 20)")
        g.add_argument("--kpm_quadrature_nodes", type=int, default=0,
                       help="Chebyshev-Gauss quadrature nodes (0 = auto = 2*M)")
        g.add_argument("--kpm_kernel", type=str, default="jackson",
                       choices=["jackson", "lorentz"],
                       help="Smoothing kernel (default jackson)")
        g.add_argument("--kpm_lorentz_lambda", type=float, default=4.0,
                       help="Lorentz kernel lambda (only if --kpm_kernel=lorentz)")
        g.add_argument("--kpm_seed", type=int, default=0,
                       help="Random seed for Hutchinson vectors (0 = nondeterministic)")
        g.add_argument("--hybrid_mode", action="store_true", default=True,
                       help="Use full ED for small clusters (default True). "
                            "Pass --no_hybrid_mode to disable.")
        g.add_argument("--no_hybrid_mode", action="store_true",
                       help="Disable hybrid mode (KPM-DOS for all clusters)")
        g.add_argument("--hybrid_threshold", type=int, default=10,
                       help="Max sites for full ED in hybrid mode (default 10)")
        g.add_argument("--symmetrized", action="store_true",
                       help="Use --symmetrized in the per-cluster ED step")
        g.add_argument("--robust_pipeline", action="store_true",
                       help="Use robust two-pipeline cross-validation for C(T)")
        g.add_argument("--n_spins_per_unit", type=int, default=4,
                       help="Spins per expansion unit (default 4 for pyrochlore tetrahedra)")
        g.add_argument("--SI_units", action="store_true",
                       help="Convert NLCE-summation output to SI units")
        g.add_argument("--resummation", type=str, default="auto",
                       choices=["auto", "direct", "euler", "wynn", "theta", "robust"],
                       help="Resummation method for the summation kernel")
        g.add_argument("--quiet", "-q", action="store_true",
                       help="Disable verbose per-order output during summation")
        g.add_argument("--verbose_plot", action="store_true",
                       help="Generate comprehensive verbose summation plots")

    def make_ed_options(self, args: argparse.Namespace, num_sites: int) -> EDOptions:
        threshold = getattr(args, "hybrid_threshold", 10)
        hybrid = getattr(args, "hybrid_mode", True) and not getattr(
            args, "no_hybrid_mode", False
        )

        if hybrid and num_sites <= threshold:
            return EDOptions(
                method="FULL",
                eigenvalues="FULL",
                thermo=True,
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                temp_bins=args.temp_bins,
                symmetrized=getattr(args, "symmetrized", False),
                use_symm=False,
            )

        # KPM-DOS path. We tunnel the KPM knobs through the existing
        # `samples` (-> kpm_num_random_vectors) and `krylov_dim`
        # (-> kpm_num_moments) fields the qed_backend already reads,
        # plus pass the kernel choice via extra_flags so the backend
        # can map it onto EDParameters.kpm_use_jackson_kernel.
        extra: list[str] = [
            f"--kpm_kernel={args.kpm_kernel}",
            f"--kpm_lorentz_lambda={args.kpm_lorentz_lambda}",
            f"--kpm_quadrature_nodes={args.kpm_quadrature_nodes}",
            f"--kpm_seed={args.kpm_seed}",
        ]

        return EDOptions(
            method="KPM_DOS",
            eigenvalues=None,
            thermo=True,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_bins=args.temp_bins,
            symmetrized=getattr(args, "symmetrized", False),
            use_symm=False,
            samples=getattr(args, "kpm_random_vectors", 20),
            krylov_dim=getattr(args, "kpm_moments", 2048),
            extra_flags=extra,
        )

    def needs_thermo(self, args: argparse.Namespace) -> bool:
        return True

    def summation_command(
        self,
        args: argparse.Namespace,
        cluster_info_dir: str,
        ed_dir: str,
        nlc_dir: str,
        order_cutoff: int,
    ) -> Optional[list[str]]:
        # KPM-DOS writes the same /thermodynamics/{...} HDF5 schema as
        # FTLM, so the NLC_sum_ftlm summation kernel handles both.
        cmd = [
            sys.executable,
            _paths.NLC_SUM_FTLM,
            f"--cluster_dir={cluster_info_dir}",
            f"--ftlm_dir={ed_dir}",
            f"--output_dir={nlc_dir}",
            "--plot",
            f"--temp_min={args.temp_min}",
            f"--temp_max={args.temp_max}",
            f"--temp_bins={args.temp_bins}",
            f"--resummation={args.resummation}",
        ]
        if getattr(args, "SI_units", False):
            cmd.append("--SI_units")
        if order_cutoff and order_cutoff != args.max_order:
            cmd.append(f"--order_cutoff={order_cutoff}")
        if getattr(args, "robust_pipeline", False):
            cmd += [
                "--robust_pipeline",
                f"--n_spins_per_unit={args.n_spins_per_unit}",
            ]
        if getattr(args, "quiet", False):
            cmd.append("--quiet")
        else:
            cmd.append("--verbose")
        if getattr(args, "verbose_plot", False):
            cmd.append("--verbose_plot")
        return cmd
