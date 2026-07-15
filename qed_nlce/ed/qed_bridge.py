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

# symmetry="auto" budget: above this group size, skip qed's
# commutation-graph + maximum-clique abelian-subgroup search (O(|Aut|^2)
# pair checks + NP-hard clique; hours at |Aut| ~ 3e4) and hand it greedy
# maximal-abelian generators instead. 512 keeps the auto path's
# non-abelian star reduction for every ordinary cluster (|Aut| <~ 10^2)
# while capping the pathological symmetric ones.
_AUTO_GROUP_MAX = 512
# Probe cap for the Python automorphism search (it bails to {identity}
# above this, which would silently re-enable the auto path -- keep it
# far above any group an NLCE cluster realistically has).
_AUTO_PROBE_CAP = 1 << 20


def full_spectrum_qed(
    op: SpinHalfOperator,
    *,
    device: str = "cpu",
    log_tag: str = "qed-bridge",
) -> np.ndarray:
    """Complete sorted eigenvalue spectrum of ``op`` via ``qed.full_spectrum``.

    Auto-detects symmetry (spatial, U(1), spin-flip, time-reversal),
    routed through QED's factorized little-group engine. For large
    automorphism groups the commutation-graph + max-clique cost of
    ``symmetry="auto"`` is skipped in favour of an *explicit* abelian
    generator set from the tested Python automorphism search, so ``qed``
    still gets the fast matrix-free rep-walk path.
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

    # ------------------------------------------------------------------
    # Large-automorphism-group guard. qed's symmetry="auto" resolves its
    # maximal abelian subgroup by building the full pairwise commutation
    # graph over ALL |Aut| group elements and then running MAXIMUM-CLIQUE
    # on it -- O(|Aut|^2) pure-Python pair checks followed by an NP-hard
    # search. Fine for |Aut| ~ 10^2, catastrophic for the highly symmetric
    # clusters where the engine should win biggest: the 16-site order-5
    # pyrochlore cluster (multiplicity 1/2) has |Aut| ~ 3.1e4 -> 4.8e8
    # pairs and a >5 h clique search, while the same cluster solves in
    # ~1 s given explicit generators. Probe the group size with the
    # tested Python automorphism search and, when it is large, hand qed a
    # greedy maximal-abelian generator set instead of "auto". The greedy
    # construction is maximal (not maximum); the marginally smaller
    # subgroup costs a slightly larger dense block, which is noise next
    # to hours of clique search.
    # ------------------------------------------------------------------
    from .symmetry import find_automorphisms, maximal_abelian_subgroup

    autos = find_automorphisms(op, max_group=_AUTO_PROBE_CAP)
    if len(autos) > _AUTO_GROUP_MAX:
        t_g = time.monotonic()
        ab = maximal_abelian_subgroup([(p, 0) for p in autos], n)
        # ab elements/generators are (perm, flip) composites with flip=0
        # by construction; qed's ``symmetry=`` takes bare site
        # permutations (spin flip is composed separately via
        # ``spin_flip="auto"``).
        gens = [list(g[0]) for g, _order in ab.gens]
        logging.info(
            "[%s] N=%d: |Aut|=%d exceeds the symmetry='auto' budget (%d); "
            "using a greedy maximal-abelian generator set "
            "(%d elements, %.1fs) instead of the commutation-graph "
            "max-clique path",
            log_tag, n, len(autos), _AUTO_GROUP_MAX,
            len(ab.elements), time.monotonic() - t_g,
        )
        res = qed.full_spectrum(
            qop, symmetry=gens, spin_flip="auto", time_reversal="auto",
            device=device,
        )
    else:
        # Small automorphism group: the little-group engine handles the full
        # "auto" reduction directly. (The former SAB-overflow fallback was
        # removed with QED's monolithic symmetry-adapted engine -- the
        # little-group engine supersedes it and does not overflow here.)
        res = qed.full_spectrum(
            qop, symmetry="auto", spin_flip="auto", time_reversal="auto",
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
