"""Block-aware exact-tier routing (qed_nlce.ed.engine.plan_exact_solve).

Deep NLCE orders are noise-dominated under any stochastic solver (the
weight cancellation ~ (J/T)^n meets a flat ~1e-3 sample error), so the
router must send every cluster that CAN be solved exactly to the exact
tier -- gated by the largest symmetry block and available RAM, not the
raw Hilbert dimension. The plan and the solver consume the SAME
resolved ClusterSymmetry, and a borderline Burnside estimate is refined
by the engine's actual plan_only star plan.
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("qed", reason="qed C++ package not installed")

from qed_nlce.ed import (  # noqa: E402
    OP_SM, OP_SP, OP_SZ, SpinHalfOperator, spinhalf_to_qed,
)
from qed_nlce.ed import engine  # noqa: E402
from qed_nlce.core.ed_runner import EDOptions  # noqa: E402


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
    """S+S+ pair terms: breaks U(1), keeps parity + time reversal."""
    op = _xxz_chain(n)
    for i in range(n - 1):
        op.add_two(OP_SP, i, OP_SP, i + 1, 0.3)
        op.add_two(OP_SM, i, OP_SM, i + 1, 0.3)
    return op


def _plan(op, n, options=None):
    qop = spinhalf_to_qed(op)
    cs = engine.resolve_cluster_symmetry(qop)
    return cs, engine.plan_exact_solve(qop, cs, options or EDOptions(),
                                       log_tag="t")


@pytest.fixture
def big_ram(monkeypatch):
    monkeypatch.setattr(engine, "_mem_available_bytes", lambda: 120 << 30)


@pytest.fixture
def tiny_ram(monkeypatch):
    monkeypatch.setattr(engine, "_mem_available_bytes", lambda: 2 << 30)


def test_19site_xxz_with_reflection_routes_exact(big_ram):
    """The order-6 pyrochlore regime: 2^19 raw, but C(19,9)/2 ~ 46k real.
    The 46k estimate is within 4x of the 120k cap, so this also
    exercises the plan_only refinement (block_exact)."""
    cs, plan = _plan(_xxz_chain(19), 19)
    assert cs.sz_conserved and cs.time_reversal
    assert plan.feasible
    assert plan.block_exact, "borderline estimate must be plan-refined"
    assert plan.max_block <= 120_000


def test_19site_szbroken_routes_oftlm(big_ram):
    """Sz broken at N=19: even with parity, 2^18-per-parity blocks are
    beyond the block cap for a low-symmetry cluster."""
    _cs, plan = _plan(_sz_broken_chain(19), 19)
    assert not plan.feasible


def test_17site_parity_conserving_routes_exact(big_ram):
    """Triangle-based order-8 anisotropic regime (Jzpm=0): Sz broken but
    parity conserved -> 2^17/2 = 65536-per-parity blocks, dense-feasible."""
    cs, plan = _plan(_sz_broken_chain(17), 17)
    assert cs.sz_parity and not cs.sz_conserved
    assert plan.sector == (1 << 17) // 2
    assert plan.feasible


def test_17site_parity_broken_routes_oftlm(big_ram):
    """A single odd-Delta-Sz term (transverse field / Jzpm) kills parity:
    the full 2^17 block exceeds the cap for a low-symmetry cluster."""
    op = _sz_broken_chain(17)
    op.add_single(OP_SP, 3, 0.21)
    op.add_single(OP_SM, 3, 0.21)
    cs, plan = _plan(op, 17)
    assert not cs.sz_parity and not cs.sz_conserved
    assert not plan.feasible


def test_22site_xxz_routes_oftlm(big_ram):
    """Order-7 pyrochlore regime on a LOW-symmetry cluster:
    C(22,11)/2 = 352k blows the 120k block cap. The over-limit Burnside
    estimate is a certain rejection (max >= average), so the plan must
    NOT pay for a plan_only refinement here."""
    _cs, plan = _plan(_xxz_chain(22), 22)
    assert not plan.feasible
    assert not plan.block_exact


def test_sector_cap_rejects_symmetric_order7(big_ram):
    """Even with the block cap lifted, the raw-sector cap still gates
    (its default 800k admits order 7 = C(22,11) = 705432 but nothing
    larger); and the same lifted config accepts an order-6-scale
    sector, proving which cap fired."""
    lifted = EDOptions(exact_max_block=10_000_000,
                       exact_max_sector=500_000)
    _cs, plan = _plan(_xxz_chain(22), 22, lifted)
    assert plan.sector == comb(22, 11)
    assert not plan.feasible
    _cs, plan = _plan(_xxz_chain(19), 19, lifted)
    assert plan.feasible


def test_ram_guard_blocks_exact(tiny_ram):
    """Same 19-site cluster must fall back to OFTLM on a small machine."""
    _cs, plan = _plan(_xxz_chain(19), 19)
    assert not plan.feasible


def test_exact_max_block_knob(big_ram):
    _cs, plan = _plan(_xxz_chain(19), 19, EDOptions(exact_max_block=1000))
    assert not plan.feasible
    # certain rejection: estimate 46k >> 1k cap, no plan_only spent
    assert not plan.block_exact


def test_resolved_symmetry_matches_expectations(big_ram):
    """Sanity on the single detection point for the reflection chain."""
    op = _xxz_chain(19)
    qop = spinhalf_to_qed(op)
    cs = engine.resolve_cluster_symmetry(qop)
    # reflection => |A| = 2, Sz conserved, TR real
    assert cs.abelian_size == 2
    assert cs.sz_conserved and cs.time_reversal
    assert comb(19, 9) // 2 == 46189
