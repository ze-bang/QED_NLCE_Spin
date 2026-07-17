"""Bond-colored cluster dedup: correctness for direction-dependent models.

Background (the bug this guards against): the triangular generators
historically deduplicated clusters by *uncolored* graph isomorphism, but
the kitaev / anisotropic Hamiltonians assign couplings from each bond's
lattice direction -- so topologically isomorphic embeddings with
different bond-direction content (straight vs 120-degree-bent chains)
are NOT isospectral. The single stored representative then stood in,
with the combined embedding count, for genuinely inequivalent clusters,
corrupting every direction-dependent NLCE sum from order 3 up.

The fix: bond-colored canonical certificates (colors = the three bond
directions, canonical modulo the S3 color action induced by the lattice
point group -- an exact covariance of both models), enabled via
``--bond_colored`` / ``bond_colored=True`` and auto-selected by the
geometries when the model is kitaev / anisotropic.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oracle.dense import solve_spectrum
from qed_nlce.hamiltonians.cluster import ClusterData
from qed_nlce.hamiltonians.triangular import build_triangular_operator
from oracle.site_generator import (
    _cluster_certificate,
    create_triangular_lattice,
    extract_cluster_info,
    generate_clusters as gen_site,
)
from qed_nlce.prep.generate_triangle_nlce_clusters import (
    build_triangle_meta_graph,
    create_triangular_lattice_with_triangles,
    generate_triangle_clusters,
)

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])

KITAEV_KW = dict(J1=1.0, J2=0.8, Gamma=0.3, Gamma_prime=0.15, model="kitaev")
ANISO_KW = dict(J1=1.0, Jzz=1.0, Jpm=0.35, Jpmpm=0.27, Jzpm=0.19,
                model="anisotropic")


def _spectrum_from_info(info, model_kw):
    """Spectrum of a cluster built exactly the production way: from the
    positions/edges the generator would write into the .dat file."""
    nodes = sorted(info["vertices"])
    remap = {v: k for k, v in enumerate(nodes)}
    cd = ClusterData(cluster_id=0, order=len(nodes), multiplicity=1.0,
                     n_sites=len(nodes))
    cd.edges = [(remap[a], remap[b]) for a, b in info["edges"]]
    cd.positions = {
        remap[v]: (info["vertex_positions"][v][0],
                   info["vertex_positions"][v][1], 0.0)
        for v in nodes
    }
    op = build_triangular_operator(cd, **model_kw)
    return np.sort(solve_spectrum(op, use_symmetry=False))


def _hand_built_chain(positions, model_kw):
    cd = ClusterData(cluster_id=0, order=len(positions), multiplicity=1.0,
                     n_sites=len(positions))
    cd.edges = [(k, k + 1) for k in range(len(positions) - 1)]
    cd.positions = {k: (p[0], p[1], 0.0) for k, p in enumerate(positions)}
    op = build_triangular_operator(cd, **model_kw)
    return np.sort(solve_spectrum(op, use_symmetry=False))


class TestCounterexamplePinned:
    """The concrete case that makes topological dedup wrong."""

    def test_straight_vs_bent_chain_differ_for_colored_models(self):
        straight = [np.zeros(2), A1, 2 * A1]
        bent = [np.zeros(2), A1, A1 + A2]
        for kw in (KITAEV_KW, ANISO_KW):
            s = _hand_built_chain(straight, kw)
            b = _hand_built_chain(bent, kw)
            assert not np.allclose(s, b, atol=1e-9), (
                f"straight and bent 3-chains must differ for {kw['model']} "
                "-- if this ever passes, the colored dedup machinery is "
                "unnecessary and should be revisited"
            )

    def test_straight_vs_bent_chain_equal_for_xxz(self):
        straight = [np.zeros(2), A1, 2 * A1]
        bent = [np.zeros(2), A1, A1 + A2]
        s = _hand_built_chain(straight, dict(J1=1.0, model="xxz_j1j2"))
        b = _hand_built_chain(bent, dict(J1=1.0, model="xxz_j1j2"))
        np.testing.assert_allclose(s, b, atol=1e-9)


@pytest.fixture(scope="module")
def site_lattice():
    G, pos = create_triangular_lattice(9, periodic=True)
    return G, pos


class TestSiteBasedColoredDedup:
    def test_topological_counts_unchanged(self, site_lattice):
        """Default (uncolored) mode must keep the historical counts."""
        G, _ = site_lattice
        clusters, _, _ = gen_site(G, 5, bond_colored=False)
        cnt = Counter(len(c) for c in clusters)
        assert dict(cnt) == {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}

    def test_colored_counts(self, site_lattice):
        G, _ = site_lattice
        clusters, _, _ = gen_site(G, 5, bond_colored=True)
        cnt = Counter(len(c) for c in clusters)
        # Colored refinement: order 3 splits into triangle / straight /
        # bent chain, etc. Frozen from the first validated run.
        assert dict(cnt) == {1: 1, 2: 1, 3: 3, 4: 7, 5: 22}

    def test_colored_multiplicities_refine_topological(self, site_lattice):
        """Per-order Sum-L is identical in both modes (colored classes
        partition exactly the same embedding sets)."""
        G, _ = site_lattice
        topo_c, topo_m, _ = gen_site(G, 5, bond_colored=False)
        col_c, col_m, _ = gen_site(G, 5, bond_colored=True)
        topo_sums = defaultdict(float)
        col_sums = defaultdict(float)
        for c, m in zip(topo_c, topo_m):
            topo_sums[len(c)] += m
        for c, m in zip(col_c, col_m):
            col_sums[len(c)] += m
        for order in topo_sums:
            assert col_sums[order] == pytest.approx(topo_sums[order])

    @pytest.mark.parametrize("model_kw", [KITAEV_KW, ANISO_KW],
                             ids=["kitaev", "anisotropic"])
    def test_within_class_isospectrality(self, site_lattice, model_kw):
        """THE production invariant: every embedding in a colored class,
        built through the actual .dat pipeline (extract_cluster_info ->
        Hamiltonian builder), is isospectral to its class representative.

        Includes PBC-boundary-crossing embeddings, which also validates
        the unwrapped-position fix (raw wrapped positions misclassify
        bond directions)."""
        G, pos = site_lattice
        order = 4

        # Enumerate every connected 4-site subset once, bucket by the
        # colored certificate.
        by_cert = defaultdict(list)
        seen = set()
        for anchor in G.nodes():
            stack = [(frozenset([anchor]),
                      frozenset(x for x in G.neighbors(anchor) if x > anchor))]
            while stack:
                cur, fr = stack.pop()
                if len(cur) == order:
                    if cur not in seen:
                        seen.add(cur)
                        by_cert[_cluster_certificate(G, cur, True)].append(cur)
                    continue
                for nxt in fr:
                    new = cur | {nxt}
                    nf = (fr | frozenset(G.neighbors(nxt))) - new
                    nf = frozenset(x for x in nf if x > anchor)
                    stack.append((new, nf))

        assert len(by_cert) == 7  # matches test_colored_counts order 4

        for cert, embeds in by_cert.items():
            ref = _spectrum_from_info(
                extract_cluster_info(G, pos, sorted(embeds[0])), model_kw)
            step = max(1, len(embeds) // 8)
            for e in embeds[::step]:
                spec = _spectrum_from_info(
                    extract_cluster_info(G, pos, sorted(e)), model_kw)
                np.testing.assert_allclose(
                    spec, ref, atol=1e-9,
                    err_msg="embeddings within one colored class must be "
                            "isospectral (S3 covariance violated or "
                            "position unwrap broken)",
                )

    def test_colored_counts_lattice_size_invariant(self):
        sigs = []
        for L in (8, 10):
            G, _ = create_triangular_lattice(L, periodic=True)
            clusters, mults, _ = gen_site(G, 4, bond_colored=True)
            sig = defaultdict(list)
            for c, m in zip(clusters, mults):
                sig[len(c)].append(round(m, 6))
            sigs.append({o: sorted(v) for o, v in sig.items()})
        assert sigs[0] == sigs[1]

    def test_unwrapped_positions_are_nn_consistent(self, site_lattice):
        """Every written edge must be exactly one lattice spacing long --
        the property raw PBC positions violate for boundary-crossing
        representatives (and which _bond_type needs to classify bonds)."""
        G, pos = site_lattice
        clusters, _, _ = gen_site(G, 5, bond_colored=True)
        for c in clusters:
            info = extract_cluster_info(G, pos, c)
            for u, v in info["edges"]:
                d = (np.asarray(info["vertex_positions"][v])
                     - np.asarray(info["vertex_positions"][u]))
                assert abs(np.linalg.norm(d) - 1.0) < 1e-9


class TestTriangleBasedColoredDedup:
    def test_colored_refines_but_preserves_sum_L(self):
        L = 7
        lat, _, tris, _, _ = create_triangular_lattice_with_triangles(L)
        meta, _ = build_triangle_meta_graph(tris, L)
        ij_of = {v: lat.nodes[v]["ij"] for v in lat.nodes()}

        _, m_topo, o_topo = generate_triangle_clusters(meta, tris, 4)
        _, m_col, o_col = generate_triangle_clusters(
            meta, tris, 4, colored_ctx=(ij_of, L))

        topo_cnt, col_cnt = Counter(o_topo), Counter(o_col)
        # Topological counts = the published reference (1, 1, 3, 5).
        assert [topo_cnt[k] for k in (1, 2, 3, 4)] == [1, 1, 3, 5]
        # Colored refinement (frozen from the first validated run).
        assert [col_cnt[k] for k in (1, 2, 3, 4)] == [1, 1, 4, 10]

        topo_sums = defaultdict(Fraction)
        col_sums = defaultdict(Fraction)
        for m, o in zip(m_topo, o_topo):
            topo_sums[o] += m
        for m, o in zip(m_col, o_col):
            col_sums[o] += m
        assert topo_sums == col_sums
