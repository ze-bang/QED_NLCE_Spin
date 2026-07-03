"""Parity check: qed.full_spectrum (Phase 1 runtime path) must reproduce
the pure-Python solve_spectrum oracle to machine precision.

This is the regression guard for routing the dense-ED tier through
QED's C++ symmetry engine instead of the homegrown Python solver. If
this test ever fails, do NOT trust any NLCE numbers produced via
``qed_nlce.ed.qed_bridge.full_spectrum_qed`` until it's fixed.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

pytest.importorskip("qed", reason="qed C++ package not installed")

from qed_nlce.ed import OP_SM, OP_SP, OP_SZ, SpinHalfOperator, solve_spectrum
from qed_nlce.ed.qed_bridge import full_spectrum_qed
from qed_nlce.prep.generate_pyrochlore_clusters import (
    build_tetrahedron_graph,
    create_pyrochlore_lattice,
    generate_clusters,
)


def _heisenberg_ring(n: int, jz: float = 1.0, h: float = 0.0) -> SpinHalfOperator:
    op = SpinHalfOperator(n)
    for i in range(n):
        j = (i + 1) % n
        op.add_two(OP_SZ, i, OP_SZ, j, jz)
        op.add_two(OP_SP, i, OP_SM, j, 0.5)
        op.add_two(OP_SM, i, OP_SP, j, 0.5)
    if h:
        for i in range(n):
            op.add_single(OP_SZ, i, -h)
    return op


def _pmpm_ring(n: int) -> SpinHalfOperator:
    op = SpinHalfOperator(n)
    for i in range(n):
        j = (i + 1) % n
        op.add_two(OP_SZ, i, OP_SZ, j, 1.0)
        op.add_two(OP_SP, i, OP_SP, j, 0.3)
        op.add_two(OP_SM, i, OP_SM, j, 0.3)
    return op


def _build_xxz_pyrochlore_op(cluster_tets, tetrahedra, Jxx=1.0, Jzz=1.4):
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


def _assert_parity(op: SpinHalfOperator) -> None:
    ref = np.sort(solve_spectrum(op, use_symmetry=True))
    test = np.sort(full_spectrum_qed(op))
    assert ref.shape == test.shape
    np.testing.assert_allclose(ref, test, atol=1e-8, rtol=0)


@pytest.mark.parametrize("n", [6, 8, 10])
def test_heisenberg_ring_parity(n):
    _assert_parity(_heisenberg_ring(n))


@pytest.mark.parametrize("n", [8, 10])
def test_xxz_ring_parity(n):
    _assert_parity(_heisenberg_ring(n, jz=0.7))


def test_field_broken_spin_flip_parity():
    _assert_parity(_heisenberg_ring(8, h=0.4))


def test_pmpm_ring_sz_broken_parity():
    _assert_parity(_pmpm_ring(8))


@pytest.fixture(scope="module")
def pyro_clusters():
    lattice, _pos, tetrahedra = create_pyrochlore_lattice(3, periodic=True)
    tet_graph = build_tetrahedron_graph(tetrahedra)
    clusters, _mults, _ = generate_clusters(tet_graph, 4)
    by_order = {}
    for c in clusters:
        by_order.setdefault(len(c), []).append(c)
    return tetrahedra, by_order


def test_pyrochlore_order3_parity(pyro_clusters):
    """10-site cluster with the non-abelian tetrahedral point group intact --
    exactly the case the qed non-abelian SAB engine exists for."""
    tetrahedra, by_order = pyro_clusters
    op, n = _build_xxz_pyrochlore_op(by_order[3][0], tetrahedra)
    assert n == 10
    _assert_parity(op)


def test_pyrochlore_order4_parity(pyro_clusters):
    tetrahedra, by_order = pyro_clusters
    op, n = _build_xxz_pyrochlore_op(by_order[4][0], tetrahedra)
    assert n == 13
    _assert_parity(op)
