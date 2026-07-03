"""Triangular-lattice NLCE geometry, *site-based* expansion.

Order = number of *sites*. This is the conservative expansion (more
clusters per order, slower convergence) but works for any model
formulated on the triangular lattice.

Wraps:

* ``qed_nlce/prep/generate_triangular_clusters.py``
* ``python/edlib/helper_cluster_triangular.py``

Models supported:

* ``xxz_j1j2``    -- J1-J2 XXZ
* ``kitaev``      -- J-K-Γ-Γ' Kitaev-Heisenberg
* ``anisotropic`` -- YbMgGaO4-type anisotropic exchange
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from ..core import Geometry, register_geometry
from ..ed.io import write_operator, write_site_info
from ..hamiltonians import build_triangular_operator, read_cluster_file
from . import _paths


def _add_triangular_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared arg group for the two triangular geometries."""
    g = parser.add_argument_group("triangular model parameters")
    g.add_argument("--J1", type=float, default=1.0,
                   help="Nearest-neighbour exchange (or J for kitaev model)")
    g.add_argument("--J2", type=float, default=0.0,
                   help="Next-nearest-neighbour exchange (or K for kitaev model)")
    g.add_argument("--Jz_ratio", type=float, default=1.0,
                   help="Jz/Jxy ratio for XXZ model")
    g.add_argument("--h", type=float, default=0.0, help="Magnetic field strength")
    g.add_argument("--field_dir", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   help="Field direction; default is out-of-plane (z)")
    g.add_argument("--model", type=str, default="xxz_j1j2",
                   choices=["xxz_j1j2", "kitaev", "anisotropic"],
                   help="Spin model")
    # Anisotropic model
    g.add_argument("--Jzz", type=float, default=None, help="J_zz (anisotropic)")
    g.add_argument("--Jpm", type=float, default=None, help="J_± (anisotropic)")
    g.add_argument("--Jpmpm", type=float, default=None, help="J_±± (anisotropic)")
    g.add_argument("--Jzpm", type=float, default=None, help="J_z± (anisotropic)")
    # Kitaev
    g.add_argument("--Gamma", type=float, default=None, help="Γ (kitaev)")
    g.add_argument("--Gamma_prime", type=float, default=None, help="Γ' (kitaev)")
    # g-tensor
    g.add_argument("--g_ab", type=float, default=2.0, help="In-plane g-factor")
    g.add_argument("--g_c", type=float, default=2.0, help="Out-of-plane g-factor")
    # Cluster-generator visualization
    g.add_argument("--visualize", action="store_true",
                   help="Emit visualizations during cluster generation.")
    # Triangular-specific NLCE knobs
    g.add_argument("--symm_threshold", type=int, default=13,
                   help="Site threshold for using --symm at the ED step (default 13)")
    g.add_argument("--streaming-symmetry", dest="streaming_symmetry",
                   action="store_true",
                   help="Use --streaming-symmetry kernel with cached orbit basis.")
    g.add_argument("--skip_basis_precompute", action="store_true",
                   help="Skip the orbit-basis precompute step (assumes cache exists). "
                        "Only meaningful with --streaming-symmetry.")


def _run_triangular_helper(args: argparse.Namespace, cluster_file_path: str,
                           ham_subdir: str, cluster_id: int) -> bool:
    """Build the per-cluster triangular Hamiltonian in-memory and persist
    it as ``Trans.dat`` / ``InterAll.dat`` (+ a site-info stub)."""
    try:
        cluster = read_cluster_file(cluster_file_path)
        op = build_triangular_operator(
            cluster,
            J1=args.J1,
            J2=args.J2,
            Jz_ratio=args.Jz_ratio,
            h=args.h,
            field_dir=tuple(args.field_dir),
            model=args.model,
            Jzz=args.Jzz,
            Jpm=args.Jpm,
            Jpmpm=args.Jpmpm,
            Jzpm=args.Jzpm,
            Gamma=args.Gamma,
            Gamma_prime=args.Gamma_prime,
            g_ab=args.g_ab,
            g_c=args.g_c,
        )
        write_operator(op, ham_subdir)
        order = getattr(cluster, "order", 0)
        write_site_info(ham_subdir, cluster_id, order, cluster.n_sites)
        return True
    except Exception as e:
        logging.error(
            "Hamiltonian build failed for triangular cluster %d: %s",
            cluster_id, e, exc_info=True,
        )
        return False


def _precompute_basis_for_cluster(
    args: argparse.Namespace, cluster_id: int, order: int, ham_subdir: str,
) -> bool:
    """No-op retained for API compatibility.

    The self-contained dense solver discovers and exploits all spatial
    automorphisms (plus U(1) S^z and spin-flip Z2) internally at solve
    time, so there is no separate orbit-basis precompute step.
    """
    return True


@register_geometry
class TriangularSite(Geometry):
    name = "triangular_site"
    description = "Triangular lattice; site-based NLCE expansion (order = sites)."

    default_temp_min = 0.1
    default_temp_max = 10.0
    default_temp_bins = 100
    default_min_order = 1
    default_max_order = 8

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_triangular_arguments(parser)

    def generate_clusters(
        self, args: argparse.Namespace, order: int, cluster_dir: str
    ) -> bool:
        cmd = [
            sys.executable,
            _paths.TRIANGULAR_SITE_GENERATOR,
            f"--max_order={order}",
            f"--output_dir={cluster_dir}",
        ]
        if getattr(args, "model", "xxz_j1j2") in ("kitaev", "anisotropic"):
            # Direction-dependent couplings: topologically isomorphic
            # embeddings are NOT isospectral (straight vs bent chains
            # differ), so dedup must be by bond-colored isomorphism.
            cmd.append("--bond_colored")
        if getattr(args, "visualize", False):
            cmd.append("--visualize")
        logging.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error("Triangular (site) cluster generation failed: %s", e)
            return False

    def prepare_hamiltonian(
        self, args, cluster_id, order, cluster_file_path, ham_subdir,
    ) -> bool:
        return _run_triangular_helper(args, cluster_file_path, ham_subdir, cluster_id)

    def precompute_basis(self, args, cluster_id, order, ham_subdir) -> bool:
        return _precompute_basis_for_cluster(args, cluster_id, order, ham_subdir)
