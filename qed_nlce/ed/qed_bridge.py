"""Exact full-spectrum dense ED via QED's C++ symmetry engine.

Replaces the historical in-process call to the pure-Python
:func:`qed_nlce.ed.dense.solve_spectrum` on the primary runtime path.
``qed.full_spectrum`` auto-detects spatial symmetry (abelian *and*
non-abelian point groups -- routed to the validated symmetry-adapted
basis / SAB engine) plus U(1)-Sz, spin-flip, and time-reversal, and
returns the complete sorted eigenvalue spectrum. This is a strictly
stronger reduction than the homegrown abelian-only Python path for
clusters with non-abelian point-group symmetry (e.g. pyrochlore
tetrahedra), which is the main lever for reaching NLCE order 8.

The pure-Python solver in :mod:`qed_nlce.ed.dense` /
:mod:`qed_nlce.ed.symmetry` is kept as the correctness oracle for
``tests/test_qed_bridge_parity.py`` and the existing unit tests; it is
no longer on the runtime path.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .operator import SpinHalfOperator
from .oftlm import spinhalf_to_qed

__all__ = ["full_spectrum_qed"]

# Exact marker from QED's build_sab_partition0 throw
# (QED/src/symmetry/symmetry_adapted.cpp): the cap message names its own
# env var. Deliberately NOT a fuzzy match -- substrings like "sab"/"cap"
# would match unrelated RuntimeErrors ("disabled", "capacity", ...) and
# silently reroute a genuine failure through the slower fallback,
# masking the real error.
_SAB_OVERFLOW_MARKER = "ED_SYM_SAB_MAX_DIM"


def _looks_like_sab_overflow(exc: Exception) -> bool:
    return _SAB_OVERFLOW_MARKER in str(exc)


def full_spectrum_qed(
    op: SpinHalfOperator,
    *,
    device: str = "cpu",
    log_tag: str = "qed-bridge",
) -> np.ndarray:
    """Complete sorted eigenvalue spectrum of ``op`` via ``qed.full_spectrum``.

    Auto-detects symmetry (spatial, U(1), spin-flip, time-reversal). If
    the non-abelian symmetry-adapted-basis engine overflows its
    in-memory enumeration cap (large Sz-broken clusters), falls back to
    an *explicit* abelian generator set -- computed by the existing,
    tested Python automorphism search -- so ``qed`` still gets the fast
    matrix-free rep-walk path instead of no spatial reduction at all.
    """
    import qed

    qop = spinhalf_to_qed(op)
    n = int(op.num_sites)

    t0 = time.monotonic()
    # Small-cluster fast path: below ~2^12 the plain dense eigensolve is
    # milliseconds, while symmetry='auto' pays a fixed per-call cost
    # (automorphism search + three discrete-symmetry detections + a
    # per-cluster tmpdir round-trip for the streaming lane) of order
    # seconds. NLCE runs solve HUNDREDS of tiny clusters, so auto-detect
    # overhead would dominate the whole exact tier. Correctness is
    # unaffected -- plain dense is the reference the symmetry paths are
    # tested against.
    if n <= 12:
        res = qed.full_spectrum(
            qop, symmetry=None, spin_flip="off", time_reversal="off",
            device=device,
        )
        evals = np.asarray(sorted(res.eigenvalues), dtype=np.float64)
        if evals.shape[0] != (1 << n):
            raise RuntimeError(
                f"[{log_tag}] qed.full_spectrum returned {evals.shape[0]} "
                f"eigenvalues for N={n}, expected {1 << n}."
            )
        logging.info(
            "[%s] N=%d exact spectrum via qed.full_spectrum (plain dense): "
            "%d eigenvalues, %.2fs",
            log_tag, n, evals.shape[0], time.monotonic() - t0,
        )
        return evals

    try:
        res = qed.full_spectrum(
            qop, symmetry="auto", spin_flip="auto", time_reversal="auto",
            device=device,
        )
    except RuntimeError as exc:
        if not _looks_like_sab_overflow(exc):
            raise
        logging.warning(
            "[%s] non-abelian SAB engine overflowed on N=%d (%s); "
            "falling back to an explicit abelian generator set",
            log_tag, n, exc,
        )
        from .symmetry import build_symmetry_group

        group, _has_flip = build_symmetry_group(op)
        gens = [list(p) for p in group.elements]
        res = qed.full_spectrum(
            qop, symmetry=gens, spin_flip="auto", time_reversal="auto",
            device=device,
        )
    dt = time.monotonic() - t0

    evals = np.asarray(sorted(res.eigenvalues), dtype=np.float64)
    expected = 1 << n
    if evals.shape[0] != expected:
        raise RuntimeError(
            f"[{log_tag}] qed.full_spectrum returned {evals.shape[0]} "
            f"eigenvalues for N={n} sites, expected {expected} "
            "(2**N) -- symmetry reduction dropped or duplicated states."
        )
    logging.info(
        "[%s] N=%d exact spectrum via qed.full_spectrum: %d eigenvalues, %.2fs",
        log_tag, n, evals.shape[0], dt,
    )
    return evals
