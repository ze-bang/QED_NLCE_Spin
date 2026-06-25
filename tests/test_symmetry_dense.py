"""Correctness of the symmetry-adapted dense solver.

Every reduction the solver applies (U(1) S^z sectors, spatial
automorphism orbit basis, spin-flip Z2, and the real/time-reversal
arithmetic switch) must leave the *full eigenvalue multiset* unchanged.
These tests assert ``solve_spectrum(use_symmetry=True)`` equals the
brute-force dense spectrum for a range of models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from qed_nlce.ed import (  # noqa: E402
    OP_SM,
    OP_SP,
    OP_SZ,
    SpinHalfOperator,
    detect_spin_flip,
    solve_spectrum,
)


def _heisenberg_ring(n: int, jz: float = 1.0, h: float = 0.0) -> SpinHalfOperator:
    op = SpinHalfOperator(n)
    for i in range(n):
        j = (i + 1) % n
        op.add_two(OP_SZ, i, OP_SZ, j, jz)
        op.add_two(OP_SP, i, OP_SM, j, 0.5)
        op.add_two(OP_SM, i, OP_SP, j, 0.5)
    if h:
        for i in range(n):
            op.add_single(OP_SZ, i, -h)
    return op


def _pmpm_ring(n: int) -> SpinHalfOperator:
    """A ring with S^+S^+ / S^-S^- bonds: breaks U(1) but keeps spin-flip."""
    op = SpinHalfOperator(n)
    for i in range(n):
        j = (i + 1) % n
        op.add_two(OP_SZ, i, OP_SZ, j, 1.0)
        op.add_two(OP_SP, i, OP_SP, j, 0.3)
        op.add_two(OP_SM, i, OP_SM, j, 0.3)
    return op


def _assert_spectra_match(op: SpinHalfOperator) -> None:
    sym = solve_spectrum(op, use_symmetry=True)
    ref = solve_spectrum(op, use_symmetry=False)
    assert sym.shape == ref.shape
    assert np.max(np.abs(np.sort(sym) - np.sort(ref))) < 1e-9


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_heisenberg_ring_symmetry_matches_brute_force(n):
    _assert_spectra_match(_heisenberg_ring(n))


@pytest.mark.parametrize("n", [8, 10])
def test_xxz_ring_symmetry_matches_brute_force(n):
    _assert_spectra_match(_heisenberg_ring(n, jz=0.7))


@pytest.mark.parametrize("n", [8, 10])
def test_field_breaks_spin_flip_but_spectrum_matches(n):
    op = _heisenberg_ring(n, h=0.4)
    assert not detect_spin_flip(op)
    _assert_spectra_match(op)


@pytest.mark.parametrize("n", [6, 8])
def test_pmpm_ring_no_u1_but_spin_flip(n):
    op = _pmpm_ring(n)
    assert not op.conserves_sz()
    assert detect_spin_flip(op)
    _assert_spectra_match(op)


def test_spin_flip_detection_zero_field_on_field_off():
    assert detect_spin_flip(_heisenberg_ring(6))
    assert not detect_spin_flip(_heisenberg_ring(6, h=0.5))


def test_spin_flip_shrinks_largest_block():
    """Enabling spin-flip strictly reduces (or keeps) the largest block."""
    op = _heisenberg_ring(12)
    _, rep_flip = solve_spectrum(op, use_symmetry=True, return_report=True)
    _, rep_noflip = solve_spectrum(
        op, use_symmetry=True, use_spin_flip=False, return_report=True
    )
    assert rep_flip.largest_block <= rep_noflip.largest_block
