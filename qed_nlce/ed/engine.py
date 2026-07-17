"""The clean dense-ED engine: ONE symmetry resolution, ONE feasibility
plan, ONE solve call -- all on the QED library.

This module replaces the retired ``qed_bridge`` two-lane dispatch and the
router's duplicate pure-Python symmetry estimates. The contract:

* :func:`resolve_cluster_symmetry` -- a single detection point. One
  (memoised, clique-budgeted) ``qed.find_symmetries`` call yields the
  spatial :class:`~qed.discovery.GeneratorSet` -- abelian generators PLUS
  the non-abelian residue as ``star_perms``, so QED's factorized
  little-group lane keeps the WHOLE point group -- and one
  ``_core.detect_hamiltonian_symmetries`` call yields the conserved-
  quantity flags (U(1) Sz, native Sz-parity, spin flip, time reversal).
  The router and the solver both consume this one object; they can no
  longer disagree.

* :func:`plan_exact_solve` -- block-aware feasibility of the exact
  full-spectrum tier. The cheap bound is the Burnside estimate
  ``sector / |A|`` with the ACTUAL closed abelian group size; when that
  is borderline against the caller's limits, the REAL block plan is read
  from ``_core.little_group_full_spectrum(..., plan_only=True)``
  (``max(dim_k0)`` over the stars) instead of guessing.

* :func:`full_spectrum` -- one ``qed.full_spectrum`` call with the
  resolved GeneratorSet and every discrete toggle on ``"auto"``: U(1)-Sz
  (or native Sz-parity) blocking, spatial abelian momentum sectors, the
  non-abelian little-group projection, spin-flip transport half-sweep,
  and time-reversal folds all engage inside QED. No small-N Python lane,
  no |Aut| probe, no raw generator lists.

The pure-Python solver (``tests/oracle/``) remains the correctness
oracle for the parity suite; nothing here calls it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from math import comb
from typing import Optional

import numpy as np

__all__ = [
    "ClusterSymmetry",
    "ExactPlan",
    "resolve_cluster_symmetry",
    "plan_exact_solve",
    "full_spectrum",
]


@dataclass(frozen=True)
class ClusterSymmetry:
    """Everything the router and solver need to know about one cluster's
    symmetry, resolved once."""

    gens: Optional[object]      # qed.discovery.GeneratorSet (with star_perms) or None
    sz_conserved: bool          # U(1) Sz
    sz_parity: bool             # every term shifts Sz by an even amount
    spin_flip: bool             # [H, prod_i sigma^x_i] = 0
    time_reversal: bool         # antiunitary T -> real-arithmetic blocks
    abelian_size: int           # |A| of the closed abelian core (>= 1)
    num_star_perms: int         # retained non-abelian residue size


@dataclass(frozen=True)
class ExactPlan:
    """Feasibility verdict for the exact full-spectrum tier."""

    feasible: bool
    sector: int                 # largest raw (Sz | parity | full) sector
    max_block: int              # largest dense block (estimate or exact)
    block_exact: bool           # True when plan_only supplied max_block
    is_real: bool               # real dtype (time reversal) vs complex
    need_bytes: int
    reason: str


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 << 30  # conservative fallback


def resolve_cluster_symmetry(qop, *, verbose: bool = False) -> ClusterSymmetry:
    """ONE symmetry-resolution call per cluster (results are memoised on
    the operator's term content inside ``qed.find_symmetries``)."""
    import qed
    from qed import _core

    try:
        flags = dict(_core.detect_hamiltonian_symmetries(qop))
    except Exception:  # pragma: no cover - detection must never abort a run
        flags = {}
    u1 = bool(flags.get("u1", False))
    parity = bool(flags.get("sz_parity", False))
    flip = bool(flags.get("spin_flip", False))
    trev = bool(flags.get("time_reversal", False))

    gens = None
    try:
        report = qed.find_symmetries(qop, verbose=verbose)
        gens = report.full_set
        if gens is not None and not getattr(gens, "generators", None):
            gens = None
    except ImportError:
        # pynauty/networkx missing: no spatial projection, flags still count.
        gens = None

    ab = int(getattr(gens, "group_size", 1) or 1) if gens is not None else 1
    stars = len(getattr(gens, "star_perms", []) or []) if gens is not None else 0
    return ClusterSymmetry(
        gens=gens, sz_conserved=u1, sz_parity=parity, spin_flip=flip,
        time_reversal=trev, abelian_size=max(1, ab), num_star_perms=stars,
    )


def _abelian_and_residues(cs: ClusterSymmetry, n: int):
    """(A_elements, residues) for the factorized engine, mirroring what
    qed.full_spectrum's projection lane will actually use."""
    from qed.point_group_routing import split_nonabelian, _close

    if cs.gens is None:
        return [list(range(n))], []
    split = split_nonabelian(cs.gens)
    if isinstance(split, str):
        # Pure-abelian (no residue) or over-cap: close the generators.
        A = _close([list(g) for g in cs.gens.generators])
        if A is None:
            return [list(range(n))], []
        return [list(a) for a in A], []
    return split


def plan_exact_solve(qop, cs: ClusterSymmetry, options, *,
                     log_tag: str = "engine") -> ExactPlan:
    """Block-aware feasibility of the EXACT full-spectrum tier.

    The raw Hilbert dimension is the wrong gate: what the dense
    eigensolver actually faces is the largest SYMMETRY BLOCK --
    C(N, N/2) for Sz-conserving models (2^N / 2 under native Sz-parity,
    2^N otherwise), divided by the spatial reduction, in real arithmetic
    when time reversal holds. Deep NLCE orders are noise-dominated under
    any stochastic solver (weight cancellation ~ (J/T)^n against a flat
    ~1e-3 sample error), so every cluster that CAN be exact MUST be --
    this plan pushes the exact tier to the true hardware limit.

    Two-stage block bound: the Burnside estimate ``sector / |A|`` when it
    is decisive against the caller's limits; the REAL star plan
    (``little_group_full_spectrum(..., plan_only=True)``, ``max(dim_k0)``)
    when the estimate is within a factor of 4 of a limit -- exactly the
    regime where an ~|A|-fold guess admits or rejects the wrong clusters.
    """
    from qed import _core

    n = int(qop.num_sites)
    if cs.sz_conserved:
        sector = comb(n, n // 2)
    elif cs.sz_parity:
        sector = (1 << n) // 2
    else:
        sector = 1 << n

    block = max(1, sector // cs.abelian_size)
    block_exact = False

    limit = int(getattr(options, "exact_max_block", 120_000))
    sector_limit = int(getattr(options, "exact_max_sector", 800_000))
    avail = _mem_available_bytes()
    bytes_per = (8 if cs.time_reversal else 16) * 1.25

    def _need(b: int) -> int:
        return int(b * b * bytes_per)

    def _refine_worthwhile(b: int) -> bool:
        # The Burnside estimate is the AVERAGE block; the true max is
        # always >= it, so an over-limit estimate is already a certain
        # rejection -- never pay the plan for one. Refine only when the
        # estimate passes a limit but the max (which the estimate can
        # understate by up to ~|A|) could still overshoot it: within 4x
        # below the block cap or the RAM bound.
        if b > limit or _need(b) > 0.8 * avail:
            return False
        return (b * 4 > limit) or (_need(b * 4) > 0.8 * avail)

    if (cs.gens is not None and sector <= sector_limit
            and _refine_worthwhile(block)):
        try:
            A, residues = _abelian_and_residues(cs, n)
            t0 = time.monotonic()
            plan = dict(_core.little_group_full_spectrum(
                qop, A, residues,
                n_up=(n // 2 if cs.sz_conserved else -1),
                sz_parity=(0 if (cs.sz_parity and not cs.sz_conserved)
                           else -1),
                spin_flip=-1, time_reversal=-1, plan_only=True))
            dims = [int(st["dim_k0"]) for st in plan.get("stars", [])]
            if dims:
                block = max(dims)
                block_exact = True
                logging.info(
                    "[%s] plan_only block refinement: %.1fs, "
                    "max star dim %s (Burnside estimate was %s)",
                    log_tag, time.monotonic() - t0, f"{block:,}",
                    f"{max(1, sector // cs.abelian_size):,}")
        except Exception as exc:  # estimate remains authoritative
            logging.info("[%s] plan_only refinement unavailable (%s); "
                         "using the Burnside estimate", log_tag, exc)

    need = _need(block)
    ok = (block <= limit and sector <= sector_limit
          and need <= 0.8 * avail)
    reason = (
        f"N={n} sz_conserved={cs.sz_conserved} sz_parity={cs.sz_parity} "
        f"sector={sector:,} |A|={cs.abelian_size} "
        f"residue={cs.num_star_perms} block{'=' if block_exact else '~'}"
        f"{block:,} {'real' if cs.time_reversal else 'complex'} "
        f"need~{need / 2**30:.1f} GB avail~{avail / 2**30:.1f} GB "
        f"exact_max_block={limit} exact_max_sector={sector_limit}"
    )
    logging.info("[%s] exact-tier plan: %s -> %s", log_tag, reason,
                 "EXACT" if ok else "OFTLM")
    return ExactPlan(feasible=ok, sector=sector, max_block=block,
                     block_exact=block_exact, is_real=cs.time_reversal,
                     need_bytes=need, reason=reason)


def full_spectrum(qop, cs: ClusterSymmetry, *, device: str = "cpu",
                  point_group: str = "auto",
                  log_tag: str = "engine") -> np.ndarray:
    """Complete sorted eigenvalue spectrum via ONE ``qed.full_spectrum``
    call carrying the full resolved symmetry.

    ``cs.gens`` is a GeneratorSet whose ``star_perms`` carry the
    non-abelian residue, so high-|Aut| clusters ride the factorized
    little-group lane; U(1)-Sz / native Sz-parity, the flip-transport
    half-sweep, and time-reversal folds auto-engage inside the verb (a
    trivial spatial group takes the Sz/parity-blocked sweep, never a
    plain 2^N dense build). ``point_group`` follows the qed contract
    ("auto" projects where the lane accepts, "off" keeps the abelian
    rep lane with folds).
    """
    import qed

    n = int(qop.num_sites)
    t0 = time.monotonic()
    res = qed.full_spectrum(
        qop, symmetry=cs.gens, spin_flip="auto", time_reversal="auto",
        point_group=point_group, device=device,
    )
    evals = np.sort(np.asarray(res.eigenvalues, dtype=np.float64))
    expected = 1 << n
    if evals.shape[0] != expected:
        raise RuntimeError(
            f"[{log_tag}] qed.full_spectrum returned {evals.shape[0]} "
            f"eigenvalues for N={n} sites, expected {expected} (2**N) -- "
            "symmetry reduction dropped or duplicated states."
        )
    logging.info(
        "[%s] N=%d exact spectrum via qed.full_spectrum "
        "(|A|=%d, residue=%d, point_group=%s): %d eigenvalues, %.2fs",
        log_tag, n, cs.abelian_size, cs.num_star_perms, point_group,
        evals.shape[0], time.monotonic() - t0,
    )
    return evals
