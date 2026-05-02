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
