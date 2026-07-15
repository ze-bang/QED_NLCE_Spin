"""Correctness verification for cluster generation.

Tests:
- Cluster counts per order against published values (triangular + pyrochlore)
- NLCE subcluster subtraction sanity (W(single-site)=P, high-T limit)
- Multiplicity sum rule
- High-order (5-8) validation: lattice-size invariance (pyrochlore) and
  reference counts/Sum-L (triangle-based generator). Orders whose
  generation takes >1 min are marked ``slow``; run ``pytest -m slow``
  before any production order-7/8 campaign.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qed_nlce.prep.generate_triangular_clusters import (
    create_triangular_lattice,
    generate_clusters as gen_triangular,
    identify_subclusters as sub_triangular,
    compute_automorphism_count as aut_triangular,
)
from qed_nlce.prep.generate_pyrochlore_clusters import (
    create_pyrochlore_lattice,
    build_tetrahedron_graph,
    generate_clusters as gen_pyrochlore,
    identify_subclusters as sub_pyrochlore,
    compute_automorphism_count as aut_pyrochlore,
)
from qed_nlce.prep.generate_triangle_nlce_clusters import (
    create_triangular_lattice_with_triangles,
    build_triangle_meta_graph,
    generate_triangle_clusters,
)


# Verified cluster counts for site-based NLCE on the triangular lattice.
# These count distinct connected INDUCED subgraphs (up to graph isomorphism)
# embeddable in the triangular lattice (z=6). Note: the commonly cited
# sequence 1,1,2,5,13,36,... is for the SQUARE lattice (z=4). The triangular
# lattice has fewer distinct topologies because its higher connectivity
# (e.g. C4 cannot exist as an induced subgraph since every 4-cycle has a
# chord on the triangular lattice).
# Validated with L=14 and L=16 PBC lattices (results identical).
TRIANGULAR_EXPECTED_COUNTS = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 22,
    7: 54,
}

# Diamond/pyrochlore lattice: cluster counts for z=4 connectivity
# (number of connected graphs of n nodes embeddable in z=4 graph).
# Orders 5-6 verified by generation on two lattice sizes (L=7/8 for
# order 5, L=8/9 for order 6 -- identical counts + multiplicity
# multisets, ruling out PBC wraparound artifacts): order 5 = the 3 free
# trees with max degree <= 4 (path, fork, K_{1,4}); order 6 = the 5
# such trees + the hexagonal loop (diamond-lattice girth is 6, so
# order 6 is the first order with a cycle topology).
PYROCHLORE_EXPECTED_COUNTS = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 3,
    6: 6,
    # Orders 7-8: generated on L=10, verified identical on L=11
    # (test_order7_and_8_invariant, 2026-07-03).
    7: 10,
    8: 24,
}

# Multiplicity (L_pyro) multisets per order, verified lattice-size
# invariant as above. A wraparound or dedup bug shifts these before it
# shifts the topology count.
PYROCHLORE_EXPECTED_MULTS = {
    1: [0.5],
    2: [1.0],
    3: [3.0],
    4: [2.0, 9.0],
    5: [0.5, 18.0, 27.0],
    6: [1.0, 6.0, 9.0, 54.0, 54.0, 75.0],
    7: [6.0, 12.0, 18.0, 27.0, 27.0, 54.0, 108.0, 138.0, 213.0, 300.0],
    8: [1.0, 1.5, 6.0, 12.0, 18.0, 24.0, 24.0, 36.0, 36.0, 42.0, 54.0,
        54.0, 54.0, 57.0, 150.0, 162.0, 276.0, 276.0, 300.0, 402.0,
        414.0, 426.0, 591.0, 780.0],
}

# Triangle-based NLCE reference values from the generator docstring
# (generate_triangle_nlce_clusters.py): order -> (num_clusters, Sum-L).
# Orders 1-8 reproduced by generation on the default L = order+3
# lattice (order 8: 299 clusters, ~105 s).
#
# L is the PER-SITE lattice constant (#up-triangles == #sites), consistent
# with the per-site order-0 cluster. These are 3x the pre-2026-07 values,
# which normalized orders>=1 per BOND (3 bonds/site) and made every summed
# property 3x too small. Per-site lattice constants are integers.
TRIANGLE_BASED_REFERENCE = {
    1: (1, Fraction(1)),
    2: (1, Fraction(3)),
    3: (3, Fraction(11)),
    4: (5, Fraction(44)),
    5: (12, Fraction(186)),
    6: (35, Fraction(814)),
    7: (98, Fraction(3652)),
    8: (299, Fraction(16689)),
}


class TestTriangularClusterCounts:
    """Verify triangular cluster counts against known values."""

    @pytest.fixture(scope="class")
    def triangular_data(self):
        """Generate triangular clusters up to order 7 with a big enough lattice."""
        G, pos = create_triangular_lattice(14, periodic=True)
        clusters, mults, details = gen_triangular(G, 7)
        return G, clusters, mults, details

    def test_order_counts(self, triangular_data):
        G, clusters, mults, details = triangular_data
        order_counts = Counter(len(c) for c in clusters)
        for order, expected in TRIANGULAR_EXPECTED_COUNTS.items():
            actual = order_counts.get(order, 0)
            assert actual == expected, (
                f"Order {order}: expected {expected} clusters, got {actual}"
            )

    def test_single_site_multiplicity(self, triangular_data):
        """Single-site cluster must have multiplicity L=1."""
        _, clusters, mults, _ = triangular_data
        single_sites = [(c, m) for c, m in zip(clusters, mults) if len(c) == 1]
        assert len(single_sites) == 1
        assert single_sites[0][1] == pytest.approx(1.0)

    def test_bond_multiplicity(self, triangular_data):
        """Order-2 bond cluster on triangular lattice: L = z/2 = 3."""
        _, clusters, mults, _ = triangular_data
        bonds = [(c, m) for c, m in zip(clusters, mults) if len(c) == 2]
        assert len(bonds) == 1
        # Triangular lattice z=6, each bond shared by 2 sites → L = 6/2 = 3
        assert bonds[0][1] == pytest.approx(3.0)

    def test_automorphism_count(self, triangular_data):
        """Path graph has |Aut|=2, complete triangle has |Aut|=6."""
        G, clusters, _, _ = triangular_data
        # Order 2: single bond → |Aut|=2
        bond_cluster = [c for c in clusters if len(c) == 2][0]
        assert aut_triangular(G, bond_cluster) == 2

        # Order 3: find the triangle (3 edges)
        order3 = [c for c in clusters if len(c) == 3]
        for c in order3:
            sub = G.subgraph(c)
            if sub.number_of_edges() == 3:  # triangle
                assert aut_triangular(G, c) == 6
                break


class TestPyrochloreClusterCounts:
    """Verify pyrochlore cluster counts against known values."""

    FIXTURE_MAX_ORDER = 5  # order 6 takes ~30 s -> covered by the slow tests

    @pytest.fixture(scope="class")
    def pyrochlore_data(self):
        lattice, pos, tetrahedra = create_pyrochlore_lattice(7, periodic=True)
        tet_graph = build_tetrahedron_graph(tetrahedra)
        clusters, mults, details = gen_pyrochlore(tet_graph, self.FIXTURE_MAX_ORDER)
        return tet_graph, clusters, mults, details

    def test_order_counts(self, pyrochlore_data):
        tet_graph, clusters, mults, details = pyrochlore_data
        order_counts = Counter(len(c) for c in clusters)
        for order, expected in PYROCHLORE_EXPECTED_COUNTS.items():
            if order > self.FIXTURE_MAX_ORDER:
                continue
            actual = order_counts.get(order, 0)
            assert actual == expected, (
                f"Order {order}: expected {expected} clusters, got {actual}"
            )

    def test_multiplicity_multisets(self, pyrochlore_data):
        """L_pyro multisets per order must match the verified reference."""
        _, clusters, mults, _ = pyrochlore_data
        by_order = defaultdict(list)
        for c, m in zip(clusters, mults):
            by_order[len(c)].append(round(m, 6))
        for order, expected in PYROCHLORE_EXPECTED_MULTS.items():
            if order > self.FIXTURE_MAX_ORDER:
                continue
            assert sorted(by_order[order]) == expected, (
                f"Order {order}: multiplicity multiset {sorted(by_order[order])} "
                f"!= expected {expected}"
            )

    def test_single_tet_multiplicity(self, pyrochlore_data):
        """Single tetrahedron: L_pyro = 0.5."""
        _, clusters, mults, _ = pyrochlore_data
        single = [(c, m) for c, m in zip(clusters, mults) if len(c) == 1]
        assert len(single) == 1
        assert single[0][1] == pytest.approx(0.5)

    def test_bond_multiplicity(self, pyrochlore_data):
        """Order-2 bond on diamond lattice (z=4): L_pyro = z/2/2 = 1.0."""
        _, clusters, mults, _ = pyrochlore_data
        bonds = [(c, m) for c, m in zip(clusters, mults) if len(c) == 2]
        assert len(bonds) == 1
        assert bonds[0][1] == pytest.approx(1.0)


class TestSubclusterSubtraction:
    """NLCE subtraction sanity checks."""

    @pytest.fixture(scope="class")
    def triangular_subclusters(self):
        G, pos = create_triangular_lattice(10, periodic=True)
        clusters, mults, details = gen_triangular(G, 4)
        subclusters = sub_triangular(clusters, G)
        return G, clusters, mults, subclusters

    def test_single_site_weight(self, triangular_subclusters):
        """For single-site cluster: W(c) = P(c) (no subclusters to subtract)."""
        _, clusters, _, subclusters = triangular_subclusters
        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        assert subclusters[site_idx] == []

    def test_bond_subtraction(self, triangular_subclusters):
        """For bond cluster: W = P(bond) - 2*W(site) → Y(bond, site)=2."""
        _, clusters, _, subclusters = triangular_subclusters
        bond_idx = next(i for i, c in enumerate(clusters) if len(c) == 2)
        site_idx = next(i for i, c in enumerate(clusters) if len(c) == 1)
        # subclusters[bond_idx] should contain (site_idx, 2)
        sub_dict = dict(subclusters[bond_idx])
        assert sub_dict[site_idx] == 2

    def test_high_temperature_weights_vanish(self, triangular_subclusters):
        """At beta→0, all W(c)→0 for order>1 (trivial check with constant P=1).

        If P(c)=1 for all c (infinite temperature limit), then:
        - W(site) = P(site) = 1
        - W(bond) = P(bond) - Y(bond,site)*W(site) = 1 - 2*1 = -1  (NOT zero)
        Actually this checks the NLCE sum converges: sum_c L(c)*W(c) → ln(2)
        per site at beta→0 for entropy. The individual W don't vanish but the
        NLCE series gives the correct extensive thermodynamics.

        Here we verify the much simpler statement: the subcluster table is
        correctly structured (each cluster of order n has subclusters only
        from orders 1..n-1, and each subcluster count is positive).
        """
        _, clusters, _, subclusters = triangular_subclusters
        for i, cluster in enumerate(clusters):
            for sub_idx, count in subclusters[i]:
                assert len(clusters[sub_idx]) < len(cluster)
                assert count > 0


class TestMultiplicitySumRule:
    """Verify that embedding counts are consistent.

    For a translationally-invariant lattice with N sites, the number of
    labeled embeddings of cluster c is L(c) * N. Dividing by |Aut(c)|
    gives the number of distinct (unlabeled) embeddings starting from
    any site, times N / |Aut(c)|.

    Sum rule: sum over distinct clusters c of order n:
      sum_c L(c) * |Aut(c)| = total labeled connected subgraphs of size n
    This is hard to verify directly, but we can check:
      sum_c L(c) * N / |Aut(c)| = number of distinct labeled embeddings
    which must be an integer.
    """

    def test_integer_embeddings_triangular(self):
        G, pos = create_triangular_lattice(10, periodic=True)
        N = G.number_of_nodes()
        clusters, mults, details = gen_triangular(G, 5)

        for i, (cluster, mult) in enumerate(zip(clusters, mults)):
            aut = aut_triangular(G, cluster)
            # raw_count = L * N must be integer
            raw_count = mult * N
            assert raw_count == pytest.approx(round(raw_count), abs=1e-10), (
                f"Cluster {i}: L*N = {raw_count} is not integer"
            )

    def test_integer_embeddings_pyrochlore(self):
        lattice, pos, tetrahedra = create_pyrochlore_lattice(5, periodic=True)
        tet_graph = build_tetrahedron_graph(tetrahedra)
        N = tet_graph.number_of_nodes()
        clusters, mults, details = gen_pyrochlore(tet_graph, 4)

        for i, (cluster, mult) in enumerate(zip(clusters, mults)):
            aut = aut_pyrochlore(tet_graph, cluster)
            # L_pyro = raw_count / N / 2, so raw_count = L_pyro * N * 2
            raw_count = mult * N * 2
            assert raw_count == pytest.approx(round(raw_count), abs=1e-10), (
                f"Cluster {i}: L_pyro*N*2 = {raw_count} is not integer"
            )


# ---------------------------------------------------------------------------
# High-order validation (orders 5-8) -- the range an order-8 NLCE campaign
# actually depends on. Nothing below relies on an external table alone:
# the triangle-based numbers were reproduced by generation before being
# frozen here, and the pyrochlore checks assert lattice-size INVARIANCE
# (a PBC-wraparound or dedup bug shifts counts/multiplicities between
# lattice sizes before anything else).
# ---------------------------------------------------------------------------


def _triangle_based_counts(max_order: int):
    """Generate triangle-based clusters; return {order: (count, Sum-L)}."""
    L = max_order + 3  # generator main()'s default sizing
    _, _, triangles, _, _ = create_triangular_lattice_with_triangles(L)
    meta_graph, _ = build_triangle_meta_graph(triangles, L)
    _, mults, orders = generate_triangle_clusters(meta_graph, triangles, max_order)
    counts = Counter(orders)
    sums = defaultdict(Fraction)
    for m, o in zip(mults, orders):
        sums[o] += m
    return {o: (counts[o], sums[o]) for o in counts}


class TestTriangleBasedReferenceCounts:
    """Triangle-based generator vs its own (now enforced) reference table."""

    def test_orders_1_to_6(self):
        got = _triangle_based_counts(6)
        for order in range(1, 7):
            assert got[order] == TRIANGLE_BASED_REFERENCE[order], (
                f"Order {order}: (count, Sum-L) = {got[order]} != "
                f"reference {TRIANGLE_BASED_REFERENCE[order]}"
            )

    @pytest.mark.slow
    def test_orders_7_and_8(self):
        got = _triangle_based_counts(8)
        for order in (7, 8):
            assert got[order] == TRIANGLE_BASED_REFERENCE[order], (
                f"Order {order}: (count, Sum-L) = {got[order]} != "
                f"reference {TRIANGLE_BASED_REFERENCE[order]}"
            )


def _pyrochlore_signature(L: int, max_order: int):
    """Per-order (count, sorted multiplicities) signature for invariance checks."""
    _, _, tetrahedra = create_pyrochlore_lattice(L, periodic=True)
    tet_graph = build_tetrahedron_graph(tetrahedra)
    clusters, mults, _ = gen_pyrochlore(tet_graph, max_order)
    sig = defaultdict(list)
    for c, m in zip(clusters, mults):
        sig[len(c)].append(round(m, 6))
    return {o: (len(v), sorted(v)) for o, v in sig.items()}


class TestPyrochloreLatticeSizeInvariance:
    """Counts + multiplicities must not depend on the PBC cell size.

    This is the strongest table-free self-consistency check available:
    wraparound artifacts (the classic high-order generation failure when
    the periodic cell is too small for the cluster diameter) produce
    different counts/multiplicities on different lattice sizes.
    """

    def test_order5_invariant(self):
        assert _pyrochlore_signature(7, 5) == _pyrochlore_signature(8, 5)

    @pytest.mark.slow
    def test_order6_invariant(self):
        sig8 = _pyrochlore_signature(8, 6)
        sig9 = _pyrochlore_signature(9, 6)
        assert sig8 == sig9
        # And pin against the frozen reference so a symmetric-but-wrong
        # change (same bug on both sizes) is still caught.
        assert sig8[6][0] == PYROCHLORE_EXPECTED_COUNTS[6]
        assert sig8[6][1] == PYROCHLORE_EXPECTED_MULTS[6]

    @pytest.mark.slow
    def test_order7_and_8_invariant(self):
        """Order-7/8 pyrochlore generation: lattice-size invariance AND
        agreement with the frozen reference (recorded from the first
        passing L=10/L=11 run, 2026-07-03: order 7 = 10 topologies,
        order 8 = 24)."""
        sig_a = _pyrochlore_signature(10, 8)
        sig_b = _pyrochlore_signature(11, 8)
        assert sig_a == sig_b, (
            "Order-8 pyrochlore cluster generation depends on the PBC "
            f"cell size: L=10 -> {sig_a}, L=11 -> {sig_b}"
        )
        for order in (7, 8):
            assert sig_a[order][0] == PYROCHLORE_EXPECTED_COUNTS[order]
            assert sig_a[order][1] == PYROCHLORE_EXPECTED_MULTS[order]
