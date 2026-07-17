"""Guard the J2 (next-nearest-neighbour) bond set.

J2 couples the NNN shell of the triangular lattice: distance sqrt(3)*a,
i.e. the six vectors +-(a1+a2), +-(2a1-a2), +-(2a2-a1). Anything else is a
different interaction.

Regression: `_nnn_pairs` used to walk two edges of the cluster's own NN
graph and keep whatever was not a direct neighbour. On 40 order-6 clusters
that returned 188 pairs at d=a (geometric nearest neighbours whose bond is
simply absent from the triangle cluster), 209 at d=2a (two collinear hops
land on the THIRD shell), and missed 111 of 508 true NNN (the connecting
site lies outside the cluster). A pure-J2 model came out 1.97x its exact
high-T limit. These tests pin the geometry so that cannot return.
"""
from __future__ import annotations

import numpy as np
import pytest

from qed_nlce.hamiltonians.cluster import ClusterData
from qed_nlce.hamiltonians.triangular import _A1, _A2, _nnn_pairs

SQRT3 = np.sqrt(3.0)


def _patch(n1: int, n2: int):
    """A parallelogram patch of the triangular lattice: sites at i*a1 + j*a2."""
    pos, idx = {}, {}
    k = 0
    for i in range(n1):
        for j in range(n2):
            r = i * _A1 + j * _A2
            pos[k] = (float(r[0]), float(r[1]), 0.0)
            idx[(i, j)] = k
            k += 1
    # NN bonds: the three primitive directions a1, a2, a2-a1
    edges = []
    for (i, j), s in idx.items():
        for di, dj in ((1, 0), (0, 1), (-1, 1)):
            t = idx.get((i + di, j + dj))
            if t is not None:
                edges.append((s, t))
    return ClusterData(cluster_id=0, order=0, n_sites=len(pos),
                       positions=pos, edges=edges, multiplicity=1.0,
                       sublattice={}, tetrahedra=[], orig_id=0)


def _dist(c, i, j):
    a = np.asarray(c.positions[i][:2]); b = np.asarray(c.positions[j][:2])
    return float(np.linalg.norm(b - a))


def test_nnn_pairs_are_exactly_the_sqrt3_shell():
    """Every returned pair sits at sqrt(3)*a -- no NN (a), no 3rd shell (2a)."""
    c = _patch(4, 4)
    pairs = _nnn_pairs(c)
    assert pairs, "no NNN pairs found in a 4x4 patch"
    for i, j in pairs:
        d = _dist(c, i, j)
        assert abs(d - SQRT3) < 1e-6, (
            f"pair ({i},{j}) at d={d:.4f} is not the NNN shell (sqrt3={SQRT3:.4f}); "
            "d=1 is a nearest neighbour, d=2 is the third shell")


def test_nnn_pairs_finds_every_true_nnn():
    """No true NNN pair is missed (the old 2-hop walk missed 22% of them)."""
    c = _patch(4, 4)
    got = {(min(i, j), max(i, j)) for i, j in _nnn_pairs(c)}
    ids = sorted(c.positions)
    true = {(i, j) for ii, i in enumerate(ids) for j in ids[ii + 1:]
            if abs(_dist(c, i, j) - SQRT3) < 1e-6}
    assert got == true, f"missed {len(true - got)}, spurious {len(got - true)}"


def test_nnn_pairs_ignores_missing_nn_bonds():
    """A geometric NN pair whose BOND is absent must not be called NNN.

    This is the exact failure of the 2-hop walk: the triangle expansion only
    carries the bonds of its chosen triangles, so adjacent sites can lack an
    edge -- they are then 2 hops apart and used to be counted as J2 partners.
    """
    c = _patch(3, 3)
    c.edges = [(a, b) for a, b in c.edges
               if not (abs(_dist(c, a, b) - 1.0) < 1e-6 and a == 0)]
    for i, j in _nnn_pairs(c):
        assert abs(_dist(c, i, j) - SQRT3) < 1e-6, (
            "a pair with a missing NN bond leaked into the J2 set")
