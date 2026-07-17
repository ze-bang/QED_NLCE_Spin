"""EXACT-ONLY routing policy (run_ed_in_process).

Default: every cluster solves exactly. An over-cap cluster logs a loud
warning and STILL solves exactly -- NLCE weight subtraction amplifies
stochastic error by ~(T/J)^-order, so the stochastic tier is opt-in
(``oftlm_fallback=True``) and never a silent default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("h5py")

from qed_nlce.core import dense_ed  # noqa: E402
from qed_nlce.core.ed_runner import EDOptions  # noqa: E402
from qed_nlce.ed.engine import ClusterSymmetry, ExactPlan  # noqa: E402

N_SITES = 20  # 2^20 > default oftlm_cutoff (2^18): the plan gets consulted


def _wire(monkeypatch, feasible):
    """Stub the engine seams so routing (not solving) is what's tested."""
    calls = {"exact": 0, "oftlm": 0}

    class _FakeOp:
        def iter_one_body_terms(self):
            return [(2, 0, 0.5 + 0j)]

        def iter_two_body_terms(self):
            return [(2, 0, 2, 1, 1.0 + 0j)]

    monkeypatch.setattr(dense_ed, "read_qed_operator",
                        lambda d, n: _FakeOp())
    monkeypatch.setattr(
        dense_ed, "resolve_cluster_symmetry",
        lambda qop, **kw: ClusterSymmetry(
            gens=None, sz_conserved=True, sz_parity=True, spin_flip=True,
            time_reversal=True, abelian_size=1, num_star_perms=0))
    monkeypatch.setattr(
        dense_ed, "plan_exact_solve",
        lambda qop, cs, options, **kw: ExactPlan(
            feasible=feasible, sector=1 << N_SITES, max_block=1 << 19,
            block_exact=False, is_real=True, need_bytes=1 << 40,
            reason="stubbed plan"))

    def _fake_exact(qop, cs, **kw):
        calls["exact"] += 1
        return np.zeros(1 << N_SITES)

    class _FakeThermo:
        temperatures = np.array([1.0])
        energy = np.array([0.0])
        specific_heat = np.array([0.0])
        entropy = np.array([0.0])
        free_energy = np.array([0.0])
        std_error = None

    def _fake_oftlm(qop, T, **kw):
        calls["oftlm"] += 1
        return _FakeThermo()

    monkeypatch.setattr(dense_ed, "full_spectrum", _fake_exact)
    monkeypatch.setattr(dense_ed, "oftlm_thermodynamics", _fake_oftlm)
    return calls


def test_overcap_cluster_warns_and_solves_exactly(monkeypatch, tmp_path,
                                                  caplog):
    calls = _wire(monkeypatch, feasible=False)
    with caplog.at_level("WARNING"):
        ok = dense_ed.run_ed_in_process(
            str(tmp_path), str(tmp_path), N_SITES, EDOptions(), log_tag="t")
    assert ok
    assert calls == {"exact": 1, "oftlm": 0}
    assert any("EXACT-ONLY" in r.message for r in caplog.records)


def test_feasible_cluster_solves_exactly_without_warning(monkeypatch,
                                                         tmp_path, caplog):
    calls = _wire(monkeypatch, feasible=True)
    with caplog.at_level("WARNING"):
        ok = dense_ed.run_ed_in_process(
            str(tmp_path), str(tmp_path), N_SITES, EDOptions(), log_tag="t")
    assert ok
    assert calls == {"exact": 1, "oftlm": 0}
    assert not any("EXACT-ONLY" in r.message for r in caplog.records)


def test_optin_fallback_routes_overcap_to_oftlm(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, feasible=False)
    ok = dense_ed.run_ed_in_process(
        str(tmp_path), str(tmp_path), N_SITES,
        EDOptions(oftlm_fallback=True), log_tag="t")
    assert ok
    assert calls == {"exact": 0, "oftlm": 1}


def test_cache_key_carries_routing_policy(tmp_path):
    from qed_nlce.core.cache import EigenvalueCache

    cache = EigenvalueCache(str(tmp_path))
    ham = tmp_path / "ham"
    ham.mkdir()
    (ham / "Trans.dat").write_text("num 0\n")
    k_off = cache.compute_key(geometry="g", ham_subdir=str(ham),
                              cluster_file=None, options=EDOptions(),
                              num_sites=10)
    k_on = cache.compute_key(geometry="g", ham_subdir=str(ham),
                             cluster_file=None,
                             options=EDOptions(oftlm_fallback=True),
                             num_sites=10)
    assert k_off.digest() != k_on.digest()
