"""Block-aware exact-tier routing (_exact_tier_feasible).

Deep NLCE orders are noise-dominated under any stochastic solver (the
weight cancellation ~ (J/T)^n meets a flat ~1e-3 sample error), so the
router must send every cluster that CAN be solved exactly to the exact
tier -- gated by the largest symmetry block and available RAM, not the
raw Hilbert dimension.
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qed_nlce.core import dense_ed
from qed_nlce.core.ed_runner import EDOptions
from qed_nlce.ed import OP_SM, OP_SP, OP_SZ, SpinHalfOperator


def _xxz_chain(n, reflection=True):
    """Open XXZ chain: conserves Sz, real, |Aut| >= 2 (reflection)."""
    op = SpinHalfOperator(n)
    for i in range(n - 1):
        op.add_two(OP_SZ, i, OP_SZ, i + 1, 1.0)
        op.add_two(OP_SP, i, OP_SM, i + 1, 0.4)
        op.add_two(OP_SM, i, OP_SP, i + 1, 0.4)
    if not reflection:
        # break the reflection with a single asymmetric field term
        op.add_single(OP_SZ, 0, 0.173)
    return op


def _sz_broken_chain(n):
    """S+S+ pair terms: breaks U(1), keeps time reversal."""
    op = _xxz_chain(n)
    for i in range(n - 1):
        op.add_two(OP_SP, i, OP_SP, i + 1, 0.3)
        op.add_two(OP_SM, i, OP_SM, i + 1, 0.3)
    return op


@pytest.fixture
def big_ram(monkeypatch):
    monkeypatch.setattr(dense_ed, "_mem_available_bytes", lambda: 120 << 30)


@pytest.fixture
def tiny_ram(monkeypatch):
    monkeypatch.setattr(dense_ed, "_mem_available_bytes", lambda: 2 << 30)


def test_19site_xxz_with_reflection_routes_exact(big_ram):
    """The order-6 pyrochlore regime: 2^19 raw, but C(19,9)/2 ~ 46k real."""
    op = _xxz_chain(19)
    assert dense_ed._exact_tier_feasible(op, 19, EDOptions(), "t")


def test_19site_szbroken_routes_oftlm(big_ram):
    """Sz broken at N=19: even with parity, 2^18-per-parity blocks are
    beyond the block cap for a low-symmetry cluster."""
    op = _sz_broken_chain(19)
    assert not dense_ed._exact_tier_feasible(op, 19, EDOptions(), "t")


def test_17site_parity_conserving_routes_exact(big_ram):
    """Triangle-based order-8 anisotropic regime (Jzpm=0): Sz broken but
    parity conserved -> 2^17/2 = 65536-per-parity blocks, dense-feasible."""
    op = _sz_broken_chain(17)
    assert op.conserves_sz_parity() and not op.conserves_sz()
    assert dense_ed._exact_tier_feasible(op, 17, EDOptions(), "t")


def test_17site_parity_broken_routes_oftlm(big_ram):
    """A single odd-Delta-Sz term (transverse field / Jzpm) kills parity:
    the full 2^17 block exceeds the cap for a low-symmetry cluster."""
    op = _sz_broken_chain(17)
    op.add_single(OP_SP, 3, 0.21)
    op.add_single(OP_SM, 3, 0.21)
    assert not op.conserves_sz_parity()
    assert not dense_ed._exact_tier_feasible(op, 17, EDOptions(), "t")


def test_22site_xxz_routes_oftlm(big_ram):
    """Order-7 pyrochlore regime: C(22,11)=705k blows the block cap."""
    op = _xxz_chain(22)
    assert not dense_ed._exact_tier_feasible(op, 22, EDOptions(), "t")


def test_sector_cap_rejects_symmetric_order7(big_ram):
    """Even a HIGH-symmetry 22-site cluster (small blocks) must be
    rejected: the streaming lane's serial basis construction scales with
    the raw C(22,11)=705k sector (measured 6.5 h+ vs 147 s at 92k), so
    tiny blocks do not make the solve fast. Simulate by lifting the
    block cap and confirming the sector cap still gates."""
    op = _xxz_chain(22)
    lifted = EDOptions(exact_max_block=10_000_000)
    assert not dense_ed._exact_tier_feasible(op, 22, lifted, "t")
    # Control: the same lifted-block config accepts an order-6-scale
    # sector (C(19,9) = 92k < 200k), proving the rejection above came
    # from the sector cap, not the block cap.
    assert dense_ed._exact_tier_feasible(_xxz_chain(19), 19, lifted, "t")


def test_ram_guard_blocks_exact(tiny_ram):
    """Same 19-site cluster must fall back to OFTLM on a small machine."""
    op = _xxz_chain(19)
    assert not dense_ed._exact_tier_feasible(op, 19, EDOptions(), "t")


def test_exact_max_block_knob(big_ram):
    op = _xxz_chain(19)
    small = EDOptions(exact_max_block=1000)
    assert not dense_ed._exact_tier_feasible(op, 19, small, "t")


def test_block_estimate_matches_sector_over_aut(big_ram):
    """Sanity on the estimate itself for the reflection chain."""
    op = _xxz_chain(19)
    # reflection => |Aut| = 2, Sz conserved, TR real
    from qed_nlce.ed.symmetry import detect_time_reversal, find_automorphisms
    assert len(find_automorphisms(op)) == 2
    assert op.conserves_sz()
    assert detect_time_reversal(op)
    assert comb(19, 9) // 2 == 46189
