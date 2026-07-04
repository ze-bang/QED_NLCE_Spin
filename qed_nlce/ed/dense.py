"""Symmetry-adapted dense exact diagonalization (scipy LAPACK backend).

The single public entry point :func:`solve_spectrum` returns the full
list of energy eigenvalues of a :class:`~qed_nlce.ed.operator.SpinHalfOperator`.
It block-diagonalizes the Hamiltonian using

  1. **U(1) total-S^z** sectors when ``S^z`` is conserved, or -- when only
     ``S^+ S^+`` / ``S^- S^-`` pair terms break it -- the residual
     **up-spin-parity Z2** sectors,
  2. a **spatial automorphism** symmetry-adapted (orbit/representative)
     basis inside each sector, and
  3. **antiunitary time reversal** ``T = U_F K`` (``T^2 = +1``) for the
     genuinely complex, ``S^z``-broken case (e.g. non-Kramers quantum spin
     ice): each ``T``-invariant sector is rotated into a real basis (real
     LAPACK path), or two ``T``-swapped equal-size sectors are recognised as
     isospectral and only one is diagonalised.

then diagonalizes every (much smaller) block with
:func:`scipy.linalg.eigh` (``eigvals_only=True``). scipy's default
standard-problem driver is ``evr`` (LAPACK MRRR, ``?syevr``/``?heevr``),
which is the appropriate eigenvalues-only path -- divide-and-conquer
(``?syevd``/``?heevd``) would needlessly also form the eigenvectors.
A block with no residual imaginary part is converted to real and routed
to the real-symmetric routine automatically (the time-reversal / reality
reduction). The concatenation of all block spectra is, as a multiset,
identical to the brute-force dense spectrum (verified in the test-suite).

For tiny systems, or when no symmetry is available, this gracefully
degrades to a single dense diagonalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as sla

from .operator import SpinHalfOperator
from .symmetry import (
    AbelianGroup,
    act_state,
    detect_spin_flip,
    detect_time_reversal,
    find_automorphisms,
    maximal_abelian_subgroup,
)

__all__ = ["solve_spectrum", "SymmetryReport"]

_REAL_TOL = 1e-10


@dataclass
class SymmetryReport:
    """Diagnostics describing the symmetry reduction that was applied."""

    num_sites: int
    full_dim: int
    sz_conserved: bool
    num_sz_sectors: int
    automorphism_order: int
    abelian_order: int
    abelian_factor_orders: tuple
    num_blocks: int
    largest_block: int
    sz_parity: bool = False
    time_reversal: bool = False
    num_real_blocks: int = 0

    def __str__(self) -> str:
        if self.sz_conserved:
            grading = f"Sz=on({self.num_sz_sectors} sectors)"
        elif self.sz_parity:
            grading = f"Sz=off Sz-parity=on({self.num_sz_sectors} sectors)"
        else:
            grading = "Sz=off"
        tr = ""
        if self.time_reversal:
            tr = f" T=on(real_blocks={self.num_real_blocks})"
        return (
            f"N={self.num_sites} dim={self.full_dim} {grading} "
            f"|Aut|={self.automorphism_order} "
            f"|Abelian|={self.abelian_order}{self.abelian_factor_orders} "
            f"blocks={self.num_blocks} largest={self.largest_block}{tr}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_spectrum(
    op: SpinHalfOperator,
    *,
    use_symmetry: bool = True,
    use_spin_flip: bool = True,
    use_time_reversal: bool = True,
    precomputed_group: Optional[AbelianGroup] = None,
    return_report: bool = False,
):
    """Return all eigenvalues of ``op`` (ascending), exploiting symmetry.

    Parameters
    ----------
    use_symmetry:
        Master switch. When False, performs a single brute-force dense
        diagonalization (used by the test-suite as ground truth).
    use_spin_flip:
        When True (and the operator is spin-flip invariant) the global
        ``S^z -> -S^z`` Z2 is added to the symmetry group, pairing the
        magnetisation sectors ``m`` and ``N-m``.
    use_time_reversal:
        When True (and the operator is time-reversal invariant) two extra
        reductions are exploited *when full ``U(1)`` is broken*:

        * **Sz-parity Z2** -- if every term changes ``S^z`` by an even
          amount (``S^+ S^+`` / ``S^- S^-`` pair terms only), the Hilbert
          space splits into even/odd up-spin-parity sectors.
        * **Antiunitary time reversal** ``T = U_F K`` -- with ``T^2 = +1``
          (non-Kramers) each ``T``-invariant sector is rotated into a real
          basis and diagonalised with the real-symmetric LAPACK path; when
          ``T`` instead swaps two equal-size sectors they are isospectral
          and only one is diagonalised. Both apply to the genuinely complex
          (e.g. non-Kramers quantum-spin-ice ``J_{+-}``) Hamiltonians where
          the unitary spin flip and z-basis reality are *broken*.

        These extra reductions are currently taken only when the spatial
        automorphism group is trivial (the regime where they matter most);
        a non-trivial spatial group keeps the existing complex orbit blocks.
    precomputed_group:
        A spatial :class:`AbelianGroup` computed once for this cluster
        geometry. When ``None`` it is detected from ``op``. A group whose
        elements carry spin flips triggers Sz-pair sectoring.
    return_report:
        When True, also return a :class:`SymmetryReport`.
    """
    n = op.num_sites
    full_dim = 1 << n

    if not use_symmetry:
        evals = _brute_force(op)
        if return_report:
            rep = SymmetryReport(
                n, full_dim, False, 1, 1, 1, (), 1, full_dim
            )
            return evals, rep
        return evals

    sz_ok = op.conserves_sz()

    # spatial automorphism abelian group (geometry symmetry), optionally
    # extended by the spin-flip Z2.
    if precomputed_group is not None:
        group = precomputed_group
        aut_order = group.size
    else:
        autos = find_automorphisms(op)
        elements = [(p, 0) for p in autos]
        if use_spin_flip and detect_spin_flip(op):
            elements += [(p, 1) for p in autos]
        group = maximal_abelian_subgroup(elements, n)
        aut_order = len(autos)
    has_flip = any(flip for _perm, flip in group.elements)

    # Antiunitary time reversal (only useful once full U(1) is broken).
    t_rev = (
        use_time_reversal and not sz_ok and detect_time_reversal(op)
    )

    # Choose the invariant subspaces and the time-reversal policy.
    #   realify          : rotate each (trivial-group) sector into a real
    #                      basis via T, then use the real LAPACK path.
    #   pair_sectors_by_T: T maps sector 0 <-> sector 1 bijectively, so they
    #                      are isospectral -- diagonalise one, mirror it.
    realify = False
    pair_sectors_by_T = False
    if sz_ok:
        sectors = _sz_pair_sectors(n) if has_flip else _sz_sectors(n)
    elif op.conserves_sz_parity() and not (has_flip and n % 2 == 1):
        # Spatial permutations preserve up-spin parity; the spin flip does
        # too iff N is even. (The excluded N-odd-with-flip case falls back to
        # the full space, which is still correct.)
        sectors = _parity_sectors(n)
        if t_rev:
            # T (= bit complement) preserves a parity sector iff N is even,
            # otherwise it swaps the two equal-size sectors.
            realify = n % 2 == 0
            pair_sectors_by_T = n % 2 == 1
    else:
        sectors = [np.arange(full_dim, dtype=np.int64)]
        if t_rev:
            realify = True  # T maps s <-> complement(s) within the full space

    all_evals: list[np.ndarray] = []
    num_blocks = 0
    num_real = 0
    largest = 0
    prev_evals: Optional[np.ndarray] = None
    for si, states in enumerate(sectors):
        if pair_sectors_by_T and si > 0:
            # T-image of sector 0: identical spectrum, no diagonalisation.
            assert prev_evals is not None and len(states) == prev_evals.shape[0]
            all_evals.append(prev_evals)
            continue
        sector_evals: list[np.ndarray] = []
        if realify and group.size == 1:
            index_of = {int(s): i for i, s in enumerate(states)}
            block = op.matrix_on_basis(states, index_of)
            block = _realify_block(block, states, n)
            num_blocks += 1
            num_real += 1
            largest = max(largest, block.shape[0])
            sector_evals.append(sla.eigh(block, eigvals_only=True))
        else:
            for block in _sector_blocks(op, states, group):
                num_blocks += 1
                if not np.iscomplexobj(block) or (
                    block.size and np.max(np.abs(block.imag)) < _REAL_TOL
                ):
                    num_real += 1
                largest = max(largest, block.shape[0])
                sector_evals.append(_eigh_block(block))
        ev = (
            np.concatenate(sector_evals) if sector_evals else np.zeros(0)
        )
        all_evals.append(ev)
        prev_evals = ev

    evals = np.sort(np.concatenate(all_evals)) if all_evals else np.zeros(0)

    if return_report:
        rep = SymmetryReport(
            num_sites=n,
            full_dim=full_dim,
            sz_conserved=sz_ok,
            num_sz_sectors=len(sectors),
            automorphism_order=aut_order,
            abelian_order=group.size,
            abelian_factor_orders=group.orders,
            num_blocks=num_blocks,
            largest_block=largest,
            sz_parity=(not sz_ok) and op.conserves_sz_parity(),
            time_reversal=t_rev,
            num_real_blocks=num_real,
        )
        return evals, rep
    return evals


# ---------------------------------------------------------------------------
# Block diagonalisation helpers
# ---------------------------------------------------------------------------


def _sz_sectors(n: int) -> list[np.ndarray]:
    """Partition the computational basis into fixed-popcount sectors."""
    buckets: list[list[int]] = [[] for _ in range(n + 1)]
    for s in range(1 << n):
        buckets[bin(s).count("1")].append(s)
    return [np.array(b, dtype=np.int64) for b in buckets if b]


def _sz_pair_sectors(n: int) -> list[np.ndarray]:
    """Like :func:`_sz_sectors` but merge each sector ``m`` with its
    spin-flipped partner ``N-m`` into one flip-invariant subspace.

    The middle sector (``m = N/2``, only for even ``N``) is flip-invariant
    on its own.
    """
    buckets: list[list[int]] = [[] for _ in range(n + 1)]
    for s in range(1 << n):
        buckets[bin(s).count("1")].append(s)
    out: list[np.ndarray] = []
    for m in range(n + 1):
        comp = n - m
        if m < comp:
            merged = buckets[m] + buckets[comp]
            if merged:
                out.append(np.array(merged, dtype=np.int64))
        elif m == comp:
            if buckets[m]:
                out.append(np.array(buckets[m], dtype=np.int64))
    return out


def _parity_sectors(n: int) -> list[np.ndarray]:
    """Partition the computational basis by up-spin parity (popcount mod 2).

    The residual ``Z2`` grading that survives when full ``U(1)`` is broken
    only by even-``Delta S^z`` (``S^+ S^+`` / ``S^- S^-``) terms.
    """
    buckets: list[list[int]] = [[], []]
    for s in range(1 << n):
        buckets[bin(s).count("1") & 1].append(s)
    return [np.array(b, dtype=np.int64) for b in buckets if b]


def _realify_block(H: np.ndarray, states: np.ndarray, n: int) -> np.ndarray:
    """Rotate a complex Hermitian block into a real-symmetric one using the
    antiunitary time reversal ``T = U_F K`` (``T^2 = +1``).

    ``T`` acts on the computational basis as the bit complement,
    ``T|s> = |comp(s)>`` (``U_F`` flips every spin, ``K`` is trivial on the
    real basis kets). On the orthonormal ``T``-invariant pair basis

        e1 = (|s> + |s_bar>)/sqrt2 ,   e2 = -i(|s> - |s_bar>)/sqrt2

    (one pair per ``{s, s_bar}``) every matrix element ``<e_a|H|e_b>`` is
    real because ``[T, H] = 0`` and ``T e = e``. The change of basis is the
    sparse unitary ``W`` (two non-zeros per column), so ``W^dagger H W`` is
    formed in ``O(D^2)`` by combining columns then rows -- no dense matmul.

    ``states`` must be closed under complementation (every ``s_bar`` present);
    this holds for the full space and, when ``N`` is even, for each up-spin
    parity sector. The residual imaginary part is asserted to be at numerical
    zero (a non-zero value would mean ``T`` is not actually a symmetry).
    """
    comp = (1 << n) - 1
    index_of = {int(s): i for i, s in enumerate(states)}
    D = len(states)
    if D % 2 != 0:
        raise ValueError("realification needs an even-dimensional subspace")
    seen = np.zeros(D, dtype=bool)
    I = np.empty(D // 2, dtype=np.int64)
    J = np.empty(D // 2, dtype=np.int64)
    p = 0
    for i in range(D):
        if seen[i]:
            continue
        j = index_of.get(int(states[i]) ^ comp)
        if j is None:
            raise ValueError(
                "time-reversal realification requires a complement-closed "
                "subspace"
            )
        if j == i:
            raise ValueError("realification: unexpected self-conjugate state")
        I[p] = i
        J[p] = j
        seen[i] = True
        seen[j] = True
        p += 1
    r = 1.0 / math.sqrt(2.0)

    # HW: columns in pair order, even col = e1, odd col = e2.
    HW = np.empty((D, D), dtype=np.complex128)
    HW[:, 0::2] = (H[:, I] + H[:, J]) * r
    HW[:, 1::2] = (H[:, J] - H[:, I]) * (1j * r)
    # W^dagger (HW): rows in pair order, even row = e1, odd row = e2.
    out = np.empty((D, D), dtype=np.complex128)
    out[0::2, :] = (HW[I, :] + HW[J, :]) * r
    out[1::2, :] = (HW[I, :] - HW[J, :]) * (1j * r)

    max_imag = float(np.max(np.abs(out.imag))) if out.size else 0.0
    scale = 1.0 + (float(np.max(np.abs(out.real))) if out.size else 0.0)
    if max_imag > 1e-7 * scale:
        raise ValueError(
            f"time-reversal realification left residual imaginary part "
            f"{max_imag:.2e} (operator is not T-symmetric?)"
        )
    return np.ascontiguousarray(out.real)


def _eigh_block(block: np.ndarray) -> np.ndarray:
    """Diagonalize one Hermitian block, using real arithmetic when the
    block is (numerically) real.

    ``overwrite_a=True`` is essential at scale: without it scipy COPIES
    the input, doubling peak memory -- a 65k complex block is 68 GB and
    the silent copy is the difference between fitting in RAM and an OOM
    kill hours into a cluster solve. No caller reuses the block after
    this. ``check_finite=False`` skips a full O(n^2) validation pass
    over data we just constructed ourselves.
    """
    if block.shape[0] == 1:
        return np.array([block[0, 0].real])
    if np.iscomplexobj(block) and np.max(np.abs(block.imag)) < _REAL_TOL:
        block = np.ascontiguousarray(block.real)
    return sla.eigh(block, eigvals_only=True, overwrite_a=True,
                    check_finite=False)


def _sector_blocks(
    op: SpinHalfOperator,
    states: np.ndarray,
    group: AbelianGroup,
):
    """Yield the dense symmetry-adapted Hamiltonian blocks for one
    invariant subspace ``states`` under the abelian ``group``.

    When the group is trivial (size 1) this yields a single dense block
    built directly on ``states`` (the plain Sz-sector path).
    """
    if group.size <= 1:
        index_of = {int(s): i for i, s in enumerate(states)}
        yield op.matrix_on_basis(states, index_of)
        return

    yield from _orbit_blocks(op, states, group)


def _orbit_blocks(
    op: SpinHalfOperator,
    states: np.ndarray,
    group: AbelianGroup,
):
    """Orbit/representative symmetry-adapted block construction.

    Implements the standard projector formula for an abelian group of
    permutations acting on the computational basis with unit phases::

        <b,k|H|a,k> = (|A| / sqrt(M_a M_b)) * sum_h chi_k(h)^* <b|H T_h|a>

    where a, b are orbit representatives, ``M_a = |A| * |Stab_a|`` and k
    is a character (one integer per cyclic factor).
    """
    A = group.elements
    nA = len(A)
    n = op.num_sites
    state_set = set(int(s) for s in states)

    # orbit representative + stabiliser for every state in this sector.
    rep_of: dict[int, int] = {}
    stab_of: dict[int, list] = {}
    images_cache: dict[int, list[int]] = {}
    for s in state_set:
        if s in rep_of:
            continue
        imgs = [act_state(g, s, n) for g in A]
        orbit = set(imgs)
        r = min(orbit)
        # stabiliser of the representative
        rimgs = [act_state(g, r, n) for g in A]
        stab = [A[i] for i in range(nA) if rimgs[i] == r]
        for x in orbit:
            rep_of[x] = r
        stab_of[r] = stab
        images_cache[r] = rimgs

    reps = sorted(set(rep_of.values()))
    rep_index = {r: i for i, r in enumerate(reps)}

    # Precompute k-independent H connections between representatives:
    #   (a_idx, b_idx, group_elem, amplitude)
    connections: list[tuple[int, int, tuple, complex]] = []
    for a in reps:
        ai = rep_index[a]
        for g in A:
            sg = act_state(g, a, n)
            for s_out, amp in op._apply_to_state(sg):
                b = rep_of.get(s_out)
                if b is None:
                    continue
                if s_out != b:
                    # only the bare representative integer contributes
                    continue
                connections.append((ai, rep_index[b], g, amp))

    M = {r: nA * len(stab_of[r]) for r in reps}

    # Build one block per character k.
    for k in group.all_momenta():
        # which representatives carry a basis vector at this momentum?
        allowed = np.zeros(len(reps), dtype=bool)
        for r in reps:
            ok = True
            for h in stab_of[r]:
                if abs(group.character(k, h) - 1.0) > 1e-9:
                    ok = False
                    break
            allowed[rep_index[r]] = ok
        n_k = int(np.count_nonzero(allowed))
        if n_k == 0:
            continue
        # local indexing among allowed reps
        local = -np.ones(len(reps), dtype=np.int64)
        local[allowed] = np.arange(n_k)

        H = np.zeros((n_k, n_k), dtype=np.complex128)
        for ai, bi, g, amp in connections:
            if not (allowed[ai] and allowed[bi]):
                continue
            la = local[ai]
            lb = local[bi]
            a = reps[ai]
            b = reps[bi]
            phase = np.conj(group.character(k, g))
            H[lb, la] += amp * phase * nA / math.sqrt(M[a] * M[b])

        # Hermitize away tiny numerical asymmetry.
        H = 0.5 * (H + H.conj().T)
        yield H


# ---------------------------------------------------------------------------
# Brute force ground truth
# ---------------------------------------------------------------------------


def _brute_force(op: SpinHalfOperator) -> np.ndarray:
    n = op.num_sites
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    index_of = {int(s): int(s) for s in states}
    H = op.matrix_on_basis(states, index_of)
    return _eigh_block(H)
