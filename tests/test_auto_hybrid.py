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
        auto_full_hilbert=1 << 12,   # SOTA default (May 2026)
        auto_min_samples=40,
        auto_max_samples=200,
        auto_backend="kpm_dos",      # SOTA default (May 2026)
        auto_kpm_kernel="jackson",
        auto_kpm_seed=0,
        auto_kpm_moments=2048,
        auto_kpm_random_vectors=32,  # SOTA default (May 2026)
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_auto_pipeline_is_registered():
    assert "auto" in list_pipelines()


def test_auto_picks_full_symmetrized_for_small_clusters():
    """Small clusters (below the FULL-ED ceiling) use FULL_SYMMETRIZED:
    Sz + spatial symmetry decomposition via qed.full_spectrum."""
    pipe = get_pipeline("auto")
    args = _build_args()
    opts = pipe.make_ed_options(args, num_sites=8)  # 2^8 = 256
    assert opts.method == "FULL_SYMMETRIZED"
    assert opts.eigenvalues == "FULL"
    assert opts.thermo is True


def test_auto_fixed_sz_overrides_full_symmetrized():
    """--auto_fixed_sz still routes to FULL_FIXED_SZ (legacy Sz=0 approx)."""
    pipe = get_pipeline("auto")
    args = _build_args(auto_fixed_sz=True)
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.method == "FULL_FIXED_SZ"


def test_auto_picks_iterative_above_crossover_kpm_default():
    """With the SOTA default backend (kpm_dos), large clusters get KPM_DOS."""
    pipe = get_pipeline("auto")
    args = _build_args(auto_full_hilbert=64)  # force iterative at 8 sites
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.method == "KPM_DOS"
    # KPM-DOS knobs are tunneled through samples / krylov_dim.
    assert opts.samples == args.auto_kpm_random_vectors
    assert opts.krylov_dim == args.auto_kpm_moments


def test_auto_picks_ftlm_above_crossover_when_backend_overridden():
    """Explicit `--auto_backend=ftlm` still routes to legacy FTLM."""
    pipe = get_pipeline("auto")
    args = _build_args(auto_full_hilbert=64, auto_backend="ftlm")
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.method == "FTLM"
    assert opts.samples is not None and opts.samples >= args.auto_min_samples
    assert opts.krylov_dim is not None


def test_auto_sample_count_decays_with_size():
    """Sample-count decay is an FTLM-backend property; pin the backend."""
    pipe = get_pipeline("auto")
    args = _build_args(auto_full_hilbert=16, auto_min_samples=40,
                       auto_max_samples=200, auto_backend="ftlm")
    s_small = pipe.make_ed_options(args, num_sites=6).samples   # 2^6 = 64
    s_large = pipe.make_ed_options(args, num_sites=14).samples  # 2^14 = 16384
    assert s_small >= s_large
    assert s_large >= args.auto_min_samples
