"""Tests for the in-process qed backend that replaces the ``./ED`` subprocess.

The legacy binary-discovery and subprocess-builder code has been
removed; the only ED execution path is now
:func:`qed_nlce.core.run_ed_in_process`, which calls the ``qed`` Python
package directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


# ---------------------------------------------------------------------------
# In-process backend dispatch table
# ---------------------------------------------------------------------------


def test_can_run_in_process_supports_full():
    from qed_nlce.core import qed_backend
    assert qed_backend.can_run_in_process("FULL")
    assert qed_backend.can_run_in_process("FULL_GPU")
    assert qed_backend.can_run_in_process("LANCZOS")
    assert qed_backend.can_run_in_process("FTLM")
    assert qed_backend.can_run_in_process("FTLM_GPU")


def test_can_run_in_process_rejects_mpi_only_methods():
    from qed_nlce.core import qed_backend
    assert not qed_backend.can_run_in_process("SCALAPACK")
    assert not qed_backend.can_run_in_process("SCALAPACK_MIXED")
    assert not qed_backend.can_run_in_process("mTPQ_MPI")
    # Unknown methods -> False (transparent fallback).
    assert not qed_backend.can_run_in_process("BOGUS_METHOD_NAME")


def test_can_run_in_process_supports_full_symmetrized():
    from qed_nlce.core import qed_backend
    assert qed_backend.can_run_in_process("FULL_SYMMETRIZED")
    assert qed_backend.can_run_in_process("FULL_SYMMETRIZED_GPU")


# ---------------------------------------------------------------------------
# FULL_SYMMETRIZED correctness: complete spectrum == legacy dense FULL
# ---------------------------------------------------------------------------


def _write_heisenberg_ring(directory, n):
    """Write Trans.dat / InterAll.dat for an n-site periodic Heisenberg
    ring into ``directory`` (the on-disk form qed_backend loads)."""
    import qed
    from qed.input import HamiltonianBuilder
    from qed.workflow import _write_operator_directory

    b = HamiltonianBuilder(n)
    bonds = [(i, (i + 1) % n) for i in range(n)]
    b.heisenberg(bonds, 1.0)
    op = b.to_operator()
    _write_operator_directory(op, str(directory))
    return op


def _read_eigendata(out_subdir):
    import h5py
    import numpy as np
    with h5py.File(str(out_subdir / "ed_results.h5"), "r") as f:
        return np.sort(np.asarray(f["eigendata"]["eigenvalues"][...],
                                  dtype=float))


def test_full_symmetrized_matches_dense_full(tmp_path):
    """The complete spectrum from FULL_SYMMETRIZED (decomposed by all
    Sz x spatial symmetries) must equal the legacy dense FULL spectrum
    as a multiset."""
    import numpy as np
    from qed_nlce.core import qed_backend
    from qed_nlce.core.ed_runner import EDOptions

    n = 6
    ham = tmp_path / "ham"
    ham.mkdir()
    _write_heisenberg_ring(ham, n)

    def _run(method):
        out_root = tmp_path / method
        out_root.mkdir()
        opts = EDOptions(method=method, eigenvalues="FULL")
        ok = qed_backend.run_ed_in_process(
            str(ham), str(out_root), n, opts, log_tag=method)
        assert ok, f"{method} dispatch failed"
        return _read_eigendata(out_root / "output")

    sym = _run("FULL_SYMMETRIZED")
    dense = _run("FULL_SZ_DECOMPOSED")

    assert len(sym) == (1 << n), f"expected {1 << n} eigs, got {len(sym)}"
    assert len(sym) == len(dense)
    assert float(np.max(np.abs(sym - dense))) < 1e-9


def test_qed_available_returns_true():
    """qed is a required dep -- must always be importable."""
    from qed_nlce.core import qed_backend
    assert qed_backend.qed_available() is True


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def test_cli_keeps_legacy_subprocess_flags_as_silent_noops():
    """The subprocess path is gone, but the legacy CLI flags
    (``--ed_executable``, ``--no_in_process``, ``--in_process``,
    ``--auto_in_process``) still parse cleanly so that pre-existing
    fit / convergence drivers do not break.
    """
    import argparse
    from qed_nlce import cli

    parser = argparse.ArgumentParser()
    cli._add_common_arguments(parser)

    args = parser.parse_args([
        '--max_order', '3',
        '--ed_executable', '/some/legacy/path/ED',
        '--no_in_process',
        '--in_process',
        '--auto_in_process',
    ])
    # The values are stored but the workflow ignores them.
    assert args.ed_executable == '/some/legacy/path/ED'
    assert args.no_in_process is True
    assert args.in_process is True
    assert args.auto_in_process is True


def test_cli_rejects_mpi_only_methods_in_preflight():
    """Preflight aborts when the user requests an MPI-only method,
    since the in-process backend cannot host MPI."""
    import argparse
    import pytest
    from qed_nlce import cli

    args = argparse.Namespace(method="SCALAPACK_MIXED", use_gpu=False)
    with pytest.raises(SystemExit):
        cli._preflight_build_introspection(args)
