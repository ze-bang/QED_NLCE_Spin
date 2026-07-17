"""NLCE subtraction sanity checks.

Verifies that the inclusion-exclusion weight computation
W(c) = P(c) - sum_s Y(c,s) * W(s)
produces correct results for known cases:

1. Single-site: W = P (trivial, no subclusters)
2. Bond: W = P(bond) - 2*W(site) (Y(bond,site)=2)
3. High-T limit: the NLCE sum per site converges to the correct
   single-site partition function Z_0 = 2 (for S=1/2), giving
   entropy per site S = ln(2) at beta→0.
4. Free-energy extensivity: for trivial P(c) = Z_0^n (free-spin limit),
   the NLCE sum gives exactly ln(2).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.site_generator import (
    create_triangular_lattice,
    generate_clusters,
    identify_subclusters,
    compute_automorphism_count,
)


@pytest.fixture(scope="module")
def nlce_data():
    """Generate clusters up to order 5 and compute subclusters."""
    G, pos = create_triangular_lattice(10, periodic=True)
    clusters, mults, details = generate_clusters(G, 5)
    subclusters = identify_subclusters(clusters, G)
    return G, clusters, mults, subclusters


class TestNLCEWeightAlgebra:
    """Test the NLCE weight calculation W(c) = P(c) - sum_s Y(c,s)*W(s)."""

    def test_single_site_weight(self, nlce_data):
        """Single-site cluster: W(c) = P(c), no subtraction."""
        _, clusters, _, subclusters = nlce_data
        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        assert subclusters[site_idx] == []

    def test_bond_has_two_site_subclusters(self, nlce_data):
        """Bond cluster: Y(bond, site) = 2."""
        _, clusters, _, subclusters = nlce_data
        bond_idx = next(i for i, c in enumerate(clusters) if len(c) == 2)
        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        sub_dict = dict(subclusters[bond_idx])
        assert site_idx in sub_dict
        assert sub_dict[site_idx] == 2

    def test_triangle_subclusters(self, nlce_data):
        """Triangle cluster: Y(triangle, site)=3, Y(triangle, bond)=3."""
        G, clusters, _, subclusters = nlce_data
        # Find the triangle (order-3 cluster with 3 edges)
        triangle_idx = None
        for i, c in enumerate(clusters):
            if len(c) == 3:
                sub = G.subgraph(c)
                if sub.number_of_edges() == 3:
                    triangle_idx = i
                    break
        assert triangle_idx is not None

        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        bond_idx = next(i for i, c in enumerate(clusters) if len(c) == 2)

        sub_dict = dict(subclusters[triangle_idx])
        assert sub_dict[site_idx] == 3  # 3 sites in the triangle
        assert sub_dict[bond_idx] == 3  # 3 bonds in the triangle

    def test_order3_path_subclusters(self, nlce_data):
        """Path-3 cluster: Y(path, site)=3, Y(path, bond)=2."""
        G, clusters, _, subclusters = nlce_data
        # Path of 3 nodes: order-3 cluster with 2 edges
        path_idx = None
        for i, c in enumerate(clusters):
            if len(c) == 3:
                sub = G.subgraph(c)
                if sub.number_of_edges() == 2:
                    path_idx = i
                    break
        assert path_idx is not None

        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        bond_idx = next(i for i, c in enumerate(clusters) if len(c) == 2)

        sub_dict = dict(subclusters[path_idx])
        assert sub_dict[site_idx] == 3
        assert sub_dict[bond_idx] == 2


class TestHighTemperatureLimit:
    """At beta→0, the NLCE sum should give free-spin entropy ln(2) per site.

    In the free-spin limit (P(c) = Z_0^n = 2^n for a cluster of n sites),
    the NLCE weight W(c) for order n>1 should contribute such that the
    total sum: sum_c L(c)*W(c) = ln(2) per site (free energy).

    Specifically, for free spins: the exact extensive free energy per site is
    -T*ln(2). The NLCE should reproduce this.
    """

    def test_free_spin_nlce_sum(self, nlce_data):
        """Verify NLCE sum with P(c)=2^n gives ln(2) (free-spin entropy)."""
        _, clusters, mults, subclusters = nlce_data

        # Compute W(c) for free spins: P(c) = ln(Z_0^n) = n*ln(2)
        # Using logarithmic quantities (f = -T*ln(Z)/N_sites):
        # P(c) = n*ln(2) for free spins at any beta→0
        # This is the entropy contribution, S = ln(2) per site

        weights = {}
        for i, cluster in enumerate(clusters):
            n = len(cluster)
            P_c = n * np.log(2)  # ln(Z) for n free spins
            W_c = P_c
            for sub_idx, Y_cs in subclusters[i]:
                W_c -= Y_cs * weights[sub_idx]
            weights[i] = W_c

        # NLCE sum: S/N_site = sum_c L(c) * W(c)
        nlce_sum = sum(mults[i] * weights[i] for i in range(len(clusters)))

        # Should equal ln(2) per site
        assert nlce_sum == pytest.approx(np.log(2), rel=1e-10), (
            f"NLCE free-spin sum = {nlce_sum}, expected ln(2) = {np.log(2)}"
        )

    def test_weights_vanish_for_free_spins(self, nlce_data):
        """For free spins, W(c) = 0 for all clusters of order > 1.

        Since ln(Z) is extensive (= n*ln(2)), the NLCE subtraction should
        isolate W to be 0 for all but order 1, where W(site) = ln(2).
        This is the key property that makes NLCE work: extensive quantities
        have W=0 for all higher-order clusters.
        """
        _, clusters, mults, subclusters = nlce_data

        weights = {}
        for i, cluster in enumerate(clusters):
            n = len(cluster)
            P_c = n * np.log(2)
            W_c = P_c
            for sub_idx, Y_cs in subclusters[i]:
                W_c -= Y_cs * weights[sub_idx]
            weights[i] = W_c

        # W(site) = ln(2)
        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        assert weights[site_idx] == pytest.approx(np.log(2), rel=1e-10)

        # All higher-order W must vanish
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:
                assert weights[i] == pytest.approx(0.0, abs=1e-10), (
                    f"W(cluster {i}, order {len(cluster)}) = {weights[i]} "
                    f"!= 0 for free spins"
                )


class TestSubclusterConsistency:
    """Structural consistency of the subcluster table."""

    def test_no_self_subclusters(self, nlce_data):
        """No cluster lists itself as a subcluster."""
        _, clusters, _, subclusters = nlce_data
        for i in range(len(clusters)):
            sub_indices = [s for s, _ in subclusters[i]]
            assert i not in sub_indices

    def test_subcluster_order_strictly_smaller(self, nlce_data):
        """All subclusters have strictly smaller order."""
        _, clusters, _, subclusters = nlce_data
        for i, cluster in enumerate(clusters):
            for sub_idx, _ in subclusters[i]:
                assert len(clusters[sub_idx]) < len(cluster)

    def test_positive_multiplicities(self, nlce_data):
        """All subcluster multiplicities Y_cs are positive integers."""
        _, _, _, subclusters = nlce_data
        for i in subclusters:
            for _, count in subclusters[i]:
                assert isinstance(count, int)
                assert count > 0

    def test_site_subcluster_count_equals_order(self, nlce_data):
        """Every cluster of order n has exactly n occurrences of the single-site
        subcluster (since every connected subgraph of size 1 is a single vertex).
        """
        _, clusters, _, subclusters = nlce_data
        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                continue
            sub_dict = dict(subclusters[i])
            assert sub_dict.get(site_idx, 0) == len(cluster), (
                f"Cluster {i} (order {len(cluster)}): "
                f"Y(c, site) = {sub_dict.get(site_idx, 0)}, expected {len(cluster)}"
            )
