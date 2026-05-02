"""Tests for the on-disk eigenvalue / subcluster cache."""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from qed_nlce.core import (
    EDOptions,
    EigenvalueCache,
    SubclusterCache,
    canonical_cluster_hash,
    default_cache_dir,
)


# ----- fixtures ------------------------------------------------------------


@pytest.fixture
def tmp_cluster_file(tmp_path):
    """A minimal triangular-cluster .dat with a 3-site triangle adjacency."""
    p = tmp_path / "cluster_42.dat"
    p.write_text(
        "# Cluster ID: 42\n"
        "# Order (number of sites): 3\n"
        "# Multiplicity: 2.0\n"
        "0 0.0 0.0 0.0\n"
        "1 1.0 0.0 0.0\n"
        "2 0.5 0.866 0.0\n"
        "# Edges:\n"
        "0 1\n0 2\n1 2\n"
        "# Adjacency Matrix:\n"
        "0 1 1\n"
        "1 0 1\n"
        "1 1 0\n"
        "# Node Mapping:\n"
        "0 0\n1 1\n2 2\n"
    )
    return str(p)


@pytest.fixture
def tmp_cluster_file_relabeled(tmp_path):
    """Same triangle topology, vertices relabeled (2,0,1) -> still K_3."""
    p = tmp_path / "cluster_99_relabeled.dat"
    # Adjacency is still all-ones-off-diag, just permuted.
    p.write_text(
        "# Cluster ID: 99\n"
        "# Order (number of sites): 3\n"
        "# Multiplicity: 1.0\n"
        "0 0.5 0.866 0.0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "# Adjacency Matrix:\n"
        "0 1 1\n"
        "1 0 1\n"
        "1 1 0\n"
    )
    return str(p)


@pytest.fixture
def tmp_cluster_file_path(tmp_path):
    """3-site path graph 0-1-2 (different topology from the triangle)."""
    p = tmp_path / "cluster_path.dat"
    p.write_text(
        "# Cluster ID: 7\n"
        "# Order (number of sites): 3\n"
        "# Adjacency Matrix:\n"
        "0 1 0\n"
        "1 0 1\n"
        "0 1 0\n"
    )
    return str(p)


def _make_options() -> EDOptions:
    return EDOptions(
        method="FULL",
        eigenvalues="FULL",
        thermo=True,
        temp_min=0.1,
        temp_max=10.0,
        temp_bins=50,
    )


def _make_ham_dir(root: str, suffix: str = "") -> str:
    """Plant a minimal Hamiltonian directory with a couple of files."""
    d = os.path.join(root, "ham")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "InterAll.dat"), "w") as f:
        f.write("3\n0 0 1 0 1.0 0.0\n1 0 2 0 1.0 0.0\n0 0 2 0 1.0 0.0\n")
    with open(os.path.join(d, "Trans.dat"), "w") as f:
        f.write(f"0\n# suffix={suffix}\n")
    return d


# ----- canonical hash ------------------------------------------------------


def test_canonical_hash_invariant_under_relabel(
    tmp_cluster_file, tmp_cluster_file_relabeled
):
    h1 = canonical_cluster_hash(tmp_cluster_file)
    h2 = canonical_cluster_hash(tmp_cluster_file_relabeled)
    assert h1 == h2, "K_3 hashes must match under vertex relabeling"


def test_canonical_hash_distinguishes_topologies(
    tmp_cluster_file, tmp_cluster_file_path
):
    h_triangle = canonical_cluster_hash(tmp_cluster_file)
    h_path = canonical_cluster_hash(tmp_cluster_file_path)
    assert h_triangle != h_path


def test_canonical_hash_stable_length():
    # Hash format / length should not regress.
    fake_path = "/nonexistent/cluster.dat"
    h = canonical_cluster_hash(fake_path)
    assert len(h) == 32 and all(c in "0123456789abcdef" for c in h)


# ----- key + digest --------------------------------------------------------


def test_eigenvalue_cache_key_digest_is_deterministic(tmp_path, tmp_cluster_file):
    ham = _make_ham_dir(str(tmp_path))
    opts = _make_options()
    cache = EigenvalueCache(str(tmp_path / "cache"), enabled=True)
    k1 = cache.compute_key("triangular_site", ham, tmp_cluster_file, opts, num_sites=3)
    k2 = cache.compute_key("triangular_site", ham, tmp_cluster_file, opts, num_sites=3)
    assert k1.digest() == k2.digest()


def test_eigenvalue_cache_key_changes_with_method(tmp_path, tmp_cluster_file):
    ham = _make_ham_dir(str(tmp_path))
    cache = EigenvalueCache(str(tmp_path / "cache"), enabled=True)
    k_full = cache.compute_key("triangular_site", ham, tmp_cluster_file,
                               _make_options(), num_sites=3)
    opts2 = _make_options()
    opts2.method = "LANCZOS"
    k_lan = cache.compute_key("triangular_site", ham, tmp_cluster_file,
                              opts2, num_sites=3)
    assert k_full.digest() != k_lan.digest()


def test_eigenvalue_cache_key_changes_with_hamiltonian(tmp_path, tmp_cluster_file):
    ham_a = _make_ham_dir(str(tmp_path / "a"), suffix="A")
    ham_b = _make_ham_dir(str(tmp_path / "b"), suffix="B")
    cache = EigenvalueCache(str(tmp_path / "cache"), enabled=True)
    k_a = cache.compute_key("triangular_site", ham_a, tmp_cluster_file,
                            _make_options(), num_sites=3)
    k_b = cache.compute_key("triangular_site", ham_b, tmp_cluster_file,
                            _make_options(), num_sites=3)
    assert k_a.digest() != k_b.digest()


# ----- store + lookup roundtrip -------------------------------------------


def test_eigenvalue_cache_roundtrip(tmp_path, tmp_cluster_file):
    ham = _make_ham_dir(str(tmp_path))
    cache = EigenvalueCache(str(tmp_path / "cache"), enabled=True)
    key = cache.compute_key("triangular_site", ham, tmp_cluster_file,
                            _make_options(), num_sites=3)

    # Plant a fake ED output.
    src = tmp_path / "run1"
    out = src / "output"
    out.mkdir(parents=True)
    h5 = out / "ed_results.h5"
    h5.write_bytes(b"\x89HDF\r\n\x1a\nFAKE-PAYLOAD")
    thermo = out / "thermo"
    thermo.mkdir()
    (thermo / "thermo_data.txt").write_text("# T E\n0.1 -1.0\n")

    # Cold lookup -> miss.
    assert cache.lookup(key, str(src / "_cold_target")) is False
    cache.store(key, str(src))
    assert cache.stats.stores == 1

    # Warm lookup into a fresh dir -> hit.
    target = tmp_path / "run2"
    assert cache.lookup(key, str(target)) is True
    assert (target / "output" / "ed_results.h5").is_file()
    assert (target / "output" / "thermo" / "thermo_data.txt").is_file()
    assert cache.stats.hits == 1


def test_eigenvalue_cache_disabled_is_noop(tmp_path, tmp_cluster_file):
    ham = _make_ham_dir(str(tmp_path))
    cache = EigenvalueCache(str(tmp_path / "cache"), enabled=False)
    key = cache.compute_key("triangular_site", ham, tmp_cluster_file,
                            _make_options(), num_sites=3)
    src = tmp_path / "run1" / "output"
    src.mkdir(parents=True)
    (src / "ed_results.h5").write_bytes(b"x")
    cache.store(key, str(tmp_path / "run1"))
    assert cache.lookup(key, str(tmp_path / "run2")) is False
    # No writes should have happened anywhere under the cache root.
    if (tmp_path / "cache" / "eigenvalues").exists():
        for _, _, files in os.walk(tmp_path / "cache" / "eigenvalues"):
            assert not files


# ----- subcluster cache ----------------------------------------------------


def test_subcluster_cache_roundtrip(tmp_path):
    info_a = tmp_path / "info_a"
    info_a.mkdir()
    # Two distinct clusters: a triangle + a path.
    (info_a / "cluster_1.dat").write_text(
        "# Adjacency Matrix:\n0 1 1\n1 0 1\n1 1 0\n"
    )
    (info_a / "cluster_2.dat").write_text(
        "# Adjacency Matrix:\n0 1 0\n1 0 1\n0 1 0\n"
    )
    (info_a / "subclusters_info.txt").write_text(
        "Cluster 1 (Order 3): Subclusters: (0,1)\n"
        "Cluster 2 (Order 3): Subclusters: (0,1)\n"
    )

    cache = SubclusterCache(str(tmp_path / "cache"), enabled=True)
    cache.store("triangular_site", 3, str(info_a))
    assert cache.stats.stores == 1

    # Fresh dir with the same cluster set but no subclusters_info.txt.
    info_b = tmp_path / "info_b"
    info_b.mkdir()
    shutil.copy(info_a / "cluster_1.dat", info_b / "cluster_1.dat")
    shutil.copy(info_a / "cluster_2.dat", info_b / "cluster_2.dat")
    assert not (info_b / "subclusters_info.txt").exists()

    assert cache.lookup("triangular_site", 3, str(info_b)) is True
    assert (info_b / "subclusters_info.txt").is_file()
    assert (info_b / "subclusters_info.txt").read_text() == \
        (info_a / "subclusters_info.txt").read_text()


# ----- default cache dir ---------------------------------------------------


def test_default_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("QED_NLCE_CACHE", str(tmp_path / "explicit"))
    assert default_cache_dir() == str(tmp_path / "explicit")
    monkeypatch.delenv("QED_NLCE_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == os.path.join(str(tmp_path / "xdg"), "qed_nlce")
