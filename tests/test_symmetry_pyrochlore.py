"""End-to-end symmetry-vs-reference verification on real pyrochlore clusters.

The dense ED README promises that the spatial-automorphism (+ spin-flip +
reality) blocking yields a spectrum identical *as a multiset* to brute-force
dense ED. The unit tests in ``test_symmetry_dense.py`` check this on Heisenberg
rings; this module checks it on genuine NLCE clusters of the pyrochlore lattice,
built exactly as the workflow builds them (XXZ on every intra-tetrahedron bond).

Two tiers:

* order-3 cluster (10 sites): literal ``use_symmetry=False`` brute-force dense
  spectrum vs the fully symmetry-blocked spectrum. Cheap (2**10).
* order-4 cluster (13 sites): an independent U(1)-S^z-sector-only reference
  (no spatial symmetry) vs the full spatial+flip blocking. 2**13 is too large
  for an unsymmetrised dense matrix to be cheap, so the Sz grading -- itself
  exact and uncontroversial -- is the reference. This is the size at which the
  spatial machinery genuinely matters.

The order-5 (16-site) check lives in ``scripts``/manual runs: at 2**16 even the
Sz reference is minutes of compute, too slow for the default suite.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qed_nlce.prep.generate_pyrochlore_clusters import (
    create_pyrochlore_lattice,
    build_tetrahedron_graph,
    generate_clusters,
)
from qed_nlce.ed.operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from qed_nlce.ed.dense import solve_spectrum, _eigh_block


def _build_xxz_op(cluster_tets, tetrahedra, Jxx=1.0, Jzz=1.4):
    """XXZ Heisenberg on every intra-tetrahedron (K4) bond of the cluster."""
    sites = sorted({s for t in cluster_tets for s in tetrahedra[t]})
    remap = {s: i for i, s in enumerate(sites)}
    op = SpinHalfOperator(len(sites))
    bonds = set()
    for t in cluster_tets:
        verts = [remap[s] for s in tetrahedra[t]]
        for a, b in itertools.combinations(verts, 2):
            bonds.add((min(a, b), max(a, b)))
    for a, b in sorted(bonds):
        op.add_two(OP_SZ, a, OP_SZ, b, Jzz)
        op.add_two(OP_SP, a, OP_SM, b, 0.5 * Jxx)
        op.add_two(OP_SM, a, OP_SP, b, 0.5 * Jxx)
    return op, len(sites)


def _sz_sector_reference(op):
    """Exact spectrum by U(1)-S^z blocking only (NO spatial symmetry)."""
    n = op.num_sites
    buckets = [[] for _ in range(n + 1)]
    for s in range(1 << n):
        buckets[bin(s).count("1")].append(s)
    evals = []
    for b in buckets:
        states = np.array(b, dtype=np.int64)
        index_of = {int(s): i for i, s in enumerate(states)}
        evals.append(_eigh_block(op.matrix_on_basis(states, index_of)))
    return np.sort(np.concatenate(evals))


@pytest.fixture(scope="module")
def pyro_clusters():
    lattice, _pos, tetrahedra = create_pyrochlore_lattice(3, periodic=True)
    tet_graph = build_tetrahedron_graph(tetrahedra)
    clusters, _mults, _ = generate_clusters(tet_graph, 4)
    by_order = {}
    for c in clusters:
        by_order.setdefault(len(c), []).append(c)
    return tetrahedra, by_order


def test_order3_bruteforce_equals_symmetry(pyro_clusters):
    """10-site cluster: literal full-dense (no symmetry) == symmetry-blocked."""
    tetrahedra, by_order = pyro_clusters
    op, n = _build_xxz_op(by_order[3][0], tetrahedra)
    assert n == 10

    ref = solve_spectrum(op, use_symmetry=False)
    test, rep = solve_spectrum(op, use_symmetry=True, return_report=True)

    assert ref.shape == test.shape == (1 << n,)
    np.testing.assert_allclose(ref, test, atol=1e-9, rtol=0)
    # The spatial group must actually have reduced the problem, otherwise this
    # test silently degenerates into "dense == dense".
    assert rep.largest_block < (1 << n)


def test_order4_sz_reference_equals_full_symmetry(pyro_clusters):
    """13-site cluster: Sz-only reference == full spatial+flip blocking.

    This is the regime where spatial symmetry matters: the largest Sz sector is
    C(13,6)=1716, which the spatial group shrinks substantially.
    """
    tetrahedra, by_order = pyro_clusters
    op, n = _build_xxz_op(by_order[4][0], tetrahedra)
    assert n == 13

    ref = _sz_sector_reference(op)
    test, rep = solve_spectrum(op, use_symmetry=True, return_report=True)

    assert ref.shape == test.shape == (1 << n,)
    np.testing.assert_allclose(ref, test, atol=1e-9, rtol=0)
    # Largest symmetry block must be smaller than the largest Sz sector C(13,6).
    from math import comb
    assert rep.largest_block < comb(13, 6)
