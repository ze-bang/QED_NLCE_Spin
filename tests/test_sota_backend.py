"""Tests for the SOTA upgrades: ED binary discovery + in-process QED backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def test_discover_ed_binary_honors_env_var(tmp_path, monkeypatch):
    from qed_nlce.core import ed_runner

    fake = tmp_path / "fake_ED"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)

    monkeypatch.setenv("QED_ED_BINARY", str(fake))
    discovered = ed_runner.discover_ed_binary()
    assert discovered == str(fake.resolve())


def test_discover_ed_binary_falls_back_to_path(tmp_path, monkeypatch):
    from qed_nlce.core import ed_runner

    monkeypatch.delenv("QED_ED_BINARY", raising=False)
    monkeypatch.delenv("ED_BINARY", raising=False)

    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    fake = fake_dir / "ED"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)

    monkeypatch.setenv("PATH", str(fake_dir))
    monkeypatch.chdir(tmp_path)  # so cwd-relative candidates do not match
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    discovered = ed_runner.discover_ed_binary()
    assert discovered == str(fake.resolve())


def test_discover_ed_binary_returns_none_when_missing(tmp_path, monkeypatch):
    from qed_nlce.core import ed_runner
    import qed as _qed

    monkeypatch.delenv('QED_ED_BINARY', raising=False)
    monkeypatch.delenv('ED_BINARY', raising=False)
    monkeypatch.setenv('PATH', str(tmp_path))  # empty
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('VIRTUAL_ENV', raising=False)
    # Force the qed-package fallback to point at a tmp dir with no ED
    # binary so the last-resort branch finds nothing either.
    fake_qed_dir = tmp_path / 'fake_qed'
    fake_qed_dir.mkdir()
    fake_init = fake_qed_dir / '__init__.py'
    fake_init.write_text('')
    with mock.patch.object(_qed, '__file__', str(fake_init)):
        assert ed_runner.discover_ed_binary() is None


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
    # Unknown methods → False (transparent fallback).
    assert not qed_backend.can_run_in_process("BOGUS_METHOD_NAME")


def test_qed_available_returns_true():
    """qed is a required dep -- must always be importable."""
    from qed_nlce.core import qed_backend
    assert qed_backend.qed_available() is True


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def test_cli_recognizes_in_process_flags():
    """``--no_in_process`` (subprocess opt-out) must be a valid CLI flag.
    Legacy ``--in_process`` / ``--auto_in_process`` are kept as silent aliases.
    """
    from qed_nlce import cli

    parser = __import__('argparse').ArgumentParser()
    cli._add_common_arguments(parser)

    args = parser.parse_args(['--max_order', '3', '--no_in_process'])
    assert args.no_in_process is True

    # Legacy aliases still parse cleanly.
    args = parser.parse_args([
        '--max_order', '3', '--in_process', '--auto_in_process',
    ])
    assert args.in_process is True
    assert args.auto_in_process is True
