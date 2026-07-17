"""Sz-parity Z2 and antiunitary time-reversal reductions.

These cover the regime where full U(1)-S^z, the unitary spin flip, and
z-basis reality are all *broken* by a genuinely complex pair term
``gamma S^+ S^+ + conj(gamma) S^- S^-`` -- i.e. (non-)Kramers quantum
spin ice. In that regime ``solve_spectrum`` must still equal brute-force
dense ED while exploiting:

* the residual up-spin-parity Z2 (even/odd popcount sectors), and
* antiunitary ``T = U_F K`` (``T^2 = +1``): real-basis rotation when ``T``
  fixes a sector (N even), or isospectral sector pairing when it swaps two
  (N odd).
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qed_nlce.ed.operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from oracle.symmetry import detect_spin_flip, detect_time_reversal
from oracle.dense import solve_spectrum


def _complex_pair_chain(n: int) -> SpinHalfOperator:
    """Open chain with XXZ + a complex ``S^+S^+`` pair term whose phase
    varies per bond, so that:

    * full U(1) is broken (S^+S^+ changes S^z by 2) but parity survives,
    * the unitary spin flip is broken (complex gamma != conj(gamma)),
    * z-basis reality is broken,
    * antiunitary T survives, and
    * the per-bond phases differ so no spatial automorphism survives.
    """
    op = SpinHalfOperator(n)
    for b in range(n - 1):
        i, j = b, b + 1
        gamma = np.exp(1j * (0.7 + 1.3 * b))  # distinct phase per bond
        op.add_two(OP_SZ, i, OP_SZ, j, 1.4)
        op.add_two(OP_SP, i, OP_SM, j, 0.5)
        op.add_two(OP_SM, i, OP_SP, j, 0.5)
        op.add_two(OP_SP, i, OP_SP, j, np.conj(gamma) * 0.3)
        op.add_two(OP_SM, i, OP_SM, j, gamma * 0.3)
    return op


def test_symmetry_detection_on_complex_pair_chain():
    op = _complex_pair_chain(6)
    assert not op.conserves_sz()           # S+S+ breaks U(1)
    assert op.conserves_sz_parity()        # ...but parity survives
    assert not op.is_real_in_z_basis()     # complex gamma
    assert not detect_spin_flip(op)        # unitary flip broken
    assert detect_time_reversal(op)        # antiunitary T survives


def test_realify_path_even_n_matches_bruteforce():
    """N even: T fixes each parity sector -> real-basis rotation."""
    n = 6
    op = _complex_pair_chain(n)
    ref = solve_spectrum(op, use_symmetry=False)
    test, rep = solve_spectrum(op, use_symmetry=True, return_report=True)

    np.testing.assert_allclose(ref, test, atol=1e-9, rtol=0)
    assert rep.sz_parity and rep.time_reversal
    # trivial spatial group + parity halving -> largest block = 2^(n-1),
    # produced through the real LAPACK path.
    assert rep.abelian_order == 1
    assert rep.largest_block == (1 << (n - 1))
    assert rep.num_real_blocks >= 1


def test_realify_disabled_recovers_same_spectrum():
    """Turning T off must not change the spectrum (only the path)."""
    op = _complex_pair_chain(6)
    with_t = solve_spectrum(op, use_symmetry=True, use_time_reversal=True)
    without_t = solve_spectrum(op, use_symmetry=True, use_time_reversal=False)
    np.testing.assert_allclose(with_t, without_t, atol=1e-9, rtol=0)


def test_tpair_path_odd_n_matches_bruteforce():
    """N odd: T swaps the two equal-size parity sectors -> diagonalise one."""
    n = 5
    op = _complex_pair_chain(n)
    ref = solve_spectrum(op, use_symmetry=False)
    test, rep = solve_spectrum(op, use_symmetry=True, return_report=True)

    np.testing.assert_allclose(ref, test, atol=1e-9, rtol=0)
    assert rep.sz_parity and rep.time_reversal
    assert rep.abelian_order == 1
    # only one parity sector is diagonalised (a single 2^(n-1) block);
    # the other is its T-image.
    assert rep.num_blocks == 1
    assert rep.largest_block == (1 << (n - 1))


# --- integrated check on a genuine non-Kramers pyrochlore cluster ----------

@pytest.fixture(scope="module")
def pyro_nonkramers():
    from qed_nlce.prep.generate_pyrochlore_clusters import (
        create_pyrochlore_lattice, build_tetrahedron_graph, generate_clusters)
    from qed_nlce.hamiltonians.cluster import ClusterData
    from qed_nlce.hamiltonians.pyrochlore import build_pyrochlore_operator

    lat, _pos, tets = create_pyrochlore_lattice(3, periodic=True)
    tg = build_tetrahedron_graph(tets)
    clusters, _m, _d = generate_clusters(tg, 3)
    by_order = defaultdict(list)
    for c in clusters:
        by_order[len(c)].append(c)

    def make(cl):
        sites = sorted({s for t in cl for s in tets[t]})
        rm = {s: i for i, s in enumerate(sites)}
        bonds = set()
        for t in cl:
            for a, b in itertools.combinations([rm[s] for s in tets[t]], 2):
                bonds.add((min(a, b), max(a, b)))
        cd = ClusterData(0, len(cl), 1.0, len(sites),
                         edges=sorted(bonds),
                         sublattice={rm[s]: s % 4 for s in sites})
        return build_pyrochlore_operator(cd, Jxx=1.3, Jyy=0.7, Jzz=1.0), len(sites)

    return {o: make(by_order[o][0]) for o in (2, 3)}


@pytest.mark.parametrize("order,expect_n", [(2, 7), (3, 10)])
def test_non_kramers_pyrochlore_matches_bruteforce(pyro_nonkramers, order, expect_n):
    op, n = pyro_nonkramers[order]
    assert n == expect_n
    assert detect_time_reversal(op) and not detect_spin_flip(op)

    ref = solve_spectrum(op, use_symmetry=False)
    test, rep = solve_spectrum(op, use_symmetry=True, return_report=True)
    np.testing.assert_allclose(ref, test, atol=1e-9, rtol=0)
    assert rep.sz_parity and rep.time_reversal
    assert rep.largest_block <= (1 << (n - 1))  # at least the parity halving
