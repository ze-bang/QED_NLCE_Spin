"""Tests for the auto-hybrid pipeline."""
from __future__ import annotations

import argparse

import pytest

from qed_nlce.core import get_pipeline, list_pipelines


def _build_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        temp_min=0.1,
        temp_max=10.0,
        temp_bins=50,
        ftlm_samples=80,
        krylov_dim=300,
        hybrid_mode=True,
        no_hybrid_mode=False,
        hybrid_threshold=10,
        use_gpu=False,
        symmetrized=False,
        robust_pipeline=False,
        n_spins_per_unit=4,
        SI_units=False,
        resummation="auto",
        quiet=False,
        verbose_plot=False,
        auto_full_hilbert=1 << 16,
        auto_min_samples=40,
        auto_max_samples=200,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_auto_pipeline_is_registered():
    assert "auto" in list_pipelines()


def test_auto_picks_full_for_small_clusters():
    pipe = get_pipeline("auto")
    args = _build_args()
    opts = pipe.make_ed_options(args, num_sites=8)  # 2^8 = 256
    assert opts.method == "FULL"
    assert opts.eigenvalues == "FULL"
    assert opts.thermo is True


def test_auto_picks_ftlm_above_crossover():
    pipe = get_pipeline("auto")
    args = _build_args(auto_full_hilbert=64)  # force FTLM at 8 sites
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.method == "FTLM"
    assert opts.samples is not None and opts.samples >= args.auto_min_samples
    assert opts.krylov_dim is not None


def test_auto_sample_count_decays_with_size():
    pipe = get_pipeline("auto")
    args = _build_args(auto_full_hilbert=16, auto_min_samples=40, auto_max_samples=200)
    s_small = pipe.make_ed_options(args, num_sites=6).samples   # 2^6 = 64
    s_large = pipe.make_ed_options(args, num_sites=14).samples  # 2^14 = 16384
    assert s_small >= s_large
    assert s_large >= args.auto_min_samples
