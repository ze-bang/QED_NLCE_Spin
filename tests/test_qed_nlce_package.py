"""Smoke tests for the standalone ``qed_nlce`` NLCE package.

Verifies the registry mechanics, the dense ED-options builder, and the
in-process dense backend without running full clusters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# Ensure qed_nlce is importable regardless of the cwd pytest is launched in.
REPO_CPP_DIR = Path(__file__).resolve().parents[1]
if str(REPO_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_CPP_DIR))


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------


def test_registries_are_populated_on_import():
    """Importing ``qed_nlce`` triggers geometry + pipeline registration."""
    import qed_nlce  # noqa: F401  -- triggers _autodiscover_extensions
    from qed_nlce.core import list_geometries, list_pipelines

    geoms = list_geometries()
    pipes = list_pipelines()

    assert "pyrochlore" in geoms
    assert "triangular_triangle" in geoms
    # Triangle-based is the ONLY triangular NLCE path; the site-based
    # expansion is retired from the runtime (kept as a correctness oracle),
    # so it must NOT be selectable.
    assert "triangular_site" not in geoms

    # Dense-only: full_ed is the sole pipeline.
    assert pipes == ["full_ed"]


def test_get_geometry_and_get_pipeline_return_instances():
    import qed_nlce  # noqa: F401
    from qed_nlce.core import (
        Geometry,
        Pipeline,
        get_geometry,
        get_pipeline,
    )

    for name in ("pyrochlore", "triangular_triangle"):
        g = get_geometry(name)
        assert isinstance(g, Geometry)
        assert g.name == name

    p = get_pipeline("full_ed")
    assert isinstance(p, Pipeline)
    assert p.name == "full_ed"


def test_unknown_geometry_or_pipeline_raises():
    import qed_nlce  # noqa: F401
    from qed_nlce.core import get_geometry, get_pipeline

    with pytest.raises(KeyError, match="Unknown geometry"):
        get_geometry("does_not_exist")
    with pytest.raises(KeyError, match="Unknown pipeline"):
        get_pipeline("does_not_exist")


# ---------------------------------------------------------------------------
# Add_arguments injects the right groups
# ---------------------------------------------------------------------------


def test_pyrochlore_adds_xyz_arguments():
    import qed_nlce  # noqa: F401
    from qed_nlce.core import get_geometry

    parser = argparse.ArgumentParser()
    get_geometry("pyrochlore").add_arguments(parser)
    flags = {a.dest for a in parser._actions}
    for needed in ("Jxx", "Jyy", "Jzz", "h", "field_dir", "random_field_width"):
        assert needed in flags


def test_triangular_adds_J1J2_and_anisotropic_arguments():
    import qed_nlce  # noqa: F401
    from qed_nlce.core import get_geometry

    parser = argparse.ArgumentParser()
    get_geometry("triangular_triangle").add_arguments(parser)
    flags = {a.dest for a in parser._actions}
    for needed in (
        "J1", "J2", "Jz_ratio", "h", "field_dir", "model",
        "Jzz", "Jpm", "Jpmpm", "Jzpm", "Gamma", "Gamma_prime",
        "g_ab", "g_c",
    ):
        assert needed in flags


# ---------------------------------------------------------------------------
# EDOptions / pipeline.make_ed_options
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Helper: build a Namespace with sensible defaults."""
    defaults = dict(
        max_order=4, base_dir="./_runs", temp_min=0.001, temp_max=20.0,
        temp_bins=100,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_full_ed_pipeline_make_ed_options_default_ok():
    import qed_nlce  # noqa: F401
    from qed_nlce.core import get_pipeline

    args = _make_args(
        method="FULL", thermo=False, measure_spin=False,
        symmetrized=False, scalapack_threshold=16, no_scalapack=False,
        symm_threshold=-1, streaming_symmetry=False,
    )
    pipe = get_pipeline("full_ed")
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.eigenvalues == "FULL"
    assert opts.use_symm is True


# ---------------------------------------------------------------------------
# In-process dense backend
# ---------------------------------------------------------------------------


def test_dense_backend_accepts_every_method():
    """The dense backend is method-agnostic: full diagonalization only."""
    import qed_nlce  # noqa: F401
    from qed_nlce.core import can_run_in_process

    for name in ("FULL", "anything", "FULL_SYMMETRIZED"):
        assert can_run_in_process(name)
