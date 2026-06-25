"""Symmetry detection and symmetry-adapted basis construction.

This module provides the geometric (spatial) and Hamiltonian symmetry
machinery used by :mod:`qed_nlce.ed.dense` to block-diagonalize a
spin-1/2 Hamiltonian before handing each block to LAPACK.

Two kinds of symmetry are exploited, both *always correct* (using a
smaller symmetry group only loses optimisation, never accuracy -- this
is verified by the test-suite which checks the blocked spectrum equals
the brute-force dense spectrum as a multiset):

* **Spatial automorphisms** -- site permutations that leave the
  Hamiltonian's term multiset invariant (i.e. genuine symmetries of the
  *specific* operator, including its complex bond phases). Detected by a
  colour-refined backtracking search on the term-coloured graph. Only
  the elements of a maximal *abelian* subgroup are used to build a
  symmetry-adapted (orbit/representative) basis, so the construction is
  exact for any cluster.

* **U(1) total-S^z** -- handled separately by :mod:`qed_nlce.ed.dense`
  as a grading (one invariant subspace per magnetisation sector).

The spatial automorphism group depends only on the *geometry plus which
operator types live on each bond*, so it can be precomputed once per
topological cluster and reused for every coupling choice (this is what
the NLCE "precompute the symmetry basis induced by the geometry" step
caches).
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .operator import SpinHalfOperator

__all__ = [
    "find_automorphisms",
    "detect_spin_flip",
    "detect_time_reversal",
    "build_symmetry_group",
    "AbelianGroup",
    "maximal_abelian_subgroup",
    "permute_state",
    "act_state",
]

_ROUND = 9  # decimal places for coefficient comparison


# ---------------------------------------------------------------------------
# Symmetry elements.
#
# A symmetry element is a pair ``(perm, flip)`` where ``perm`` is a site
# permutation tuple (``perm[i]`` = image of site ``i``) and ``flip`` is a
# bit: ``0`` = identity, ``1`` = global spin-flip (S^z -> -S^z, S^+ <-> S^-),
# realised on a basis integer as the bitwise complement over ``n`` sites.
#
# The spin-flip commutes with every site permutation, so the group of such
# elements is ``Aut x Z2`` and is abelian whenever ``Aut`` is. Pure spatial
# symmetry corresponds to ``flip = 0`` throughout (the historical path).
# ---------------------------------------------------------------------------

Element = tuple  # (perm_tuple, flip_bit)


def _compose(a: Element, b: Element) -> Element:
    pa, fa = a
    pb, fb = b
    perm = tuple(pa[pb[i]] for i in range(len(pa)))
    return (perm, fa ^ fb)


def _identity(n: int) -> Element:
    return (tuple(range(n)), 0)


def _order(p: Element) -> int:
    n = len(p[0])
    ident = _identity(n)
    q = p
    o = 1
    while q != ident:
        q = _compose(p, q)
        o += 1
    return o


def _cyclic_powers(p: Element) -> list[Element]:
    n = len(p[0])
    ident = _identity(n)
    out = [ident]
    q = p
    while q != ident:
        out.append(q)
        q = _compose(p, q)
    return out


def permute_state(p: tuple[int, ...], s: int) -> int:
    """Image of basis integer ``s`` under site permutation ``p``.

    Bit at site ``i`` moves to site ``p[i]``.
    """
    out = 0
    i = 0
    ss = s
    while ss:
        if ss & 1:
            out |= 1 << p[i]
        ss >>= 1
        i += 1
    return out


def act_state(elem: Element, s: int, n: int) -> int:
    """Image of basis integer ``s`` under a symmetry element ``(perm, flip)``."""
    perm, flip = elem
    out = permute_state(perm, s)
    if flip:
        out ^= (1 << n) - 1
    return out


# ---------------------------------------------------------------------------
# Colored-graph automorphism search (exact Hamiltonian symmetries).
# ---------------------------------------------------------------------------


def _term_signatures(op: SpinHalfOperator):
    """Build per-site and per-ordered-pair term signatures.

    Returns ``(site_sig, bond_sig)`` where ``site_sig[i]`` is a hashable
    signature of single-site terms on ``i`` and ``bond_sig[(i, j)]`` of
    two-site terms with ``site1=i, site2=j``.

    Each two-site term is recorded under *both* ordered keys -- ``(s1, s2)``
    with op pair ``(o1, o2)`` and ``(s2, s1)`` with the swapped pair
    ``(o2, o1)``. This makes the per-bond signature orientation-correct:
    a site permutation that reverses a bond's stored endpoint order (every
    reflection does) is still recognised as a symmetry when the term
    *multiset* on the unordered bond is invariant (e.g. an XXZ bond carries
    both ``S^+ S^-`` and ``S^- S^+``), while a genuinely chiral bond term
    (a bare ``S^+ S^-`` with no conjugate) correctly stays asymmetric and
    blocks the reversal. Keying only the caller's stored order would
    silently discard every orientation-reversing automorphism.
    """
    n = op.num_sites
    site_terms: dict[int, list] = {i: [] for i in range(n)}
    for o, site, c in op.single:
        site_terms[site].append((o, round(c.real, _ROUND), round(c.imag, _ROUND)))
    bond_terms: dict[tuple[int, int], list] = {}
    for o1, s1, o2, s2, c in op.two:
        re, im = round(c.real, _ROUND), round(c.imag, _ROUND)
        bond_terms.setdefault((s1, s2), []).append((o1, o2, re, im))
        bond_terms.setdefault((s2, s1), []).append((o2, o1, re, im))
    site_sig = {i: tuple(sorted(site_terms[i])) for i in range(n)}
    bond_sig = {k: tuple(sorted(v)) for k, v in bond_terms.items()}
    return site_sig, bond_sig


def _refine_colors(n, site_sig, bond_sig):
    """Weisfeiler-Lehman colour refinement to shrink candidate images."""
    # initial colour: single-site signature + sorted multiset of incident
    # bond signatures (both directions).
    incident: dict[int, list] = {i: [] for i in range(n)}
    for (i, j), sig in bond_sig.items():
        incident[i].append(("out", sig))
        incident[j].append(("in", sig))
    color = {}
    for i in range(n):
        color[i] = (site_sig[i], tuple(sorted(incident[i])))
    # iterate refinement
    for _ in range(n):
        newcolor = {}
        for i in range(n):
            nbr = []
            for (a, b), sig in bond_sig.items():
                if a == i:
                    nbr.append(("out", sig, color[b]))
                if b == i:
                    nbr.append(("in", sig, color[a]))
            newcolor[i] = (color[i], tuple(sorted(nbr)))
        # compress to small ints keyed by value
        palette = {}
        comp = {}
        for i in range(n):
            v = newcolor[i]
            if v not in palette:
                palette[v] = len(palette)
            comp[i] = palette[v]
        if comp == color:
            break
        color = comp
    return color


def find_automorphisms(
    op: SpinHalfOperator, max_group: int = 4096
) -> list[tuple[int, ...]]:
    """Return all site permutations that leave ``op`` invariant.

    Includes the identity. The search is colour-refined backtracking and
    each candidate is fully verified against the operator's term multiset
    (so complex bond phases are respected exactly). Bails out returning
    only the identity if the group would exceed ``max_group`` (defensive;
    not expected for NLCE clusters).
    """
    n = op.num_sites
    ident = tuple(range(n))
    if n <= 1:
        return [ident]

    site_sig, bond_sig = _term_signatures(op)
    color = _refine_colors(n, site_sig, bond_sig)

    # candidate images for each site: same refined colour.
    by_color: dict[int, list[int]] = {}
    for i in range(n):
        by_color.setdefault(color[i], []).append(i)
    candidates = {i: list(by_color[color[i]]) for i in range(n)}

    # adjacency for partial-consistency pruning
    bond_keys = set(bond_sig.keys())

    # order sites by fewest candidates first to prune early
    search_order = sorted(range(n), key=lambda i: len(candidates[i]))

    autos: list[tuple[int, ...]] = []
    perm = [-1] * n
    used = [False] * n

    def consistent(i: int, img: int) -> bool:
        # check all already-assigned bonds involving i map correctly
        for k in search_order:
            if perm[k] == -1 or k == i:
                continue
            # bond (i,k)
            a = bond_sig.get((i, k))
            b = bond_sig.get((img, perm[k]))
            if a != b:
                return False
            a = bond_sig.get((k, i))
            b = bond_sig.get((perm[k], img))
            if a != b:
                return False
        return True

    def backtrack(idx: int) -> bool:
        if idx == n:
            cand = tuple(perm)
            autos.append(cand)
            return len(autos) >= max_group
        i = search_order[idx]
        for img in candidates[i]:
            if used[img]:
                continue
            if site_sig[i] != site_sig[img]:
                continue
            if not consistent(i, img):
                continue
            perm[i] = img
            used[img] = True
            if backtrack(idx + 1):
                return True
            perm[i] = -1
            used[img] = False
        return False

    overflow = backtrack(0)
    if overflow or len(autos) > max_group:
        return [ident]
    # ``autos`` already contains only verified bijections satisfying every
    # bond constraint, hence genuine automorphisms.
    return autos


# ---------------------------------------------------------------------------
# Abelian subgroup extraction + direct-product character structure.
# ---------------------------------------------------------------------------


@dataclass
class AbelianGroup:
    """A finite abelian group of site permutations, stored as a direct
    product of cyclic factors with explicit per-element exponents.

    Attributes
    ----------
    elements : list of permutation tuples (the full group).
    gens : list of (generator perm, order) for each cyclic factor.
    exponents : dict mapping each element -> tuple of exponents (m_a).
    orders : tuple of cyclic-factor orders (n_a).
    """

    n_sites: int
    elements: list[tuple[int, ...]]
    gens: list[tuple[tuple[int, ...], int]]
    exponents: dict[tuple[int, ...], tuple[int, ...]]
    orders: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.elements)

    def character(self, k: tuple[int, ...], elem: tuple[int, ...]) -> complex:
        """Character chi_k(elem) for momentum label ``k`` (one int per
        cyclic factor)."""
        m = self.exponents[elem]
        phase = 0.0
        for a in range(len(self.orders)):
            phase += k[a] * m[a] / self.orders[a]
        return np.exp(2j * np.pi * phase)

    def all_momenta(self):
        """Iterate every character label k = (k_0, ..., k_{r-1})."""
        return itertools.product(*[range(o) for o in self.orders])


def maximal_abelian_subgroup(
    elements: list[tuple[int, ...]], n_sites: int
) -> AbelianGroup:
    """Greedily build a maximal abelian subgroup as a direct product of
    independent cyclic factors.

    Prefers high-order generators (bigger per-factor reduction). The
    result is exact: every returned element commutes with all others and
    the factorisation into exponents is unique.
    """
    ident = _identity(n_sites)
    elem_set = set(elements)

    gens: list[tuple[tuple[int, ...], int]] = []
    current: set[tuple[int, ...]] = {ident}
    # map element -> exponent tuple, grown as factors are added.
    exps: dict[tuple[int, ...], tuple[int, ...]] = {ident: ()}

    # sort candidate generators by descending order, then lexicographically
    cand_sorted = sorted(elem_set, key=lambda p: (-_order(p), p))

    for g in cand_sorted:
        if g in current:
            continue
        # must commute with every element already in the subgroup
        if not all(_compose(g, h) == _compose(h, g) for h in current):
            continue
        og = _order(g)
        powers = _cyclic_powers(g)  # [id, g, ..., g^{og-1}]
        # independence: <g> ∩ current = {id}
        if any((p in current) and (p != ident) for p in powers):
            continue
        # form direct product current × <g>; require |new| = |current|*og
        new_elems: dict[tuple[int, ...], tuple[int, ...]] = {}
        ok = True
        for h in current:
            he = exps[h]
            for e_pow, p in enumerate(powers):
                prod = _compose(h, p)
                if prod in new_elems:
                    ok = False
                    break
                if prod not in elem_set:
                    ok = False
                    break
                new_elems[prod] = he + (e_pow,)
            if not ok:
                break
        if not ok:
            continue
        # accept generator: extend existing exponents with a 0 for new factor
        gens.append((g, og))
        # pad previous exponents with trailing 0, then merge new ones
        merged: dict[tuple[int, ...], tuple[int, ...]] = {}
        for elem, e in new_elems.items():
            merged[elem] = e
        current = set(new_elems.keys())
        exps = merged
        if len(current) == len(elem_set):
            break

    orders = tuple(o for _g, o in gens)
    return AbelianGroup(
        n_sites=n_sites,
        elements=list(current),
        gens=gens,
        exponents=exps,
        orders=orders,
    )


# ---------------------------------------------------------------------------
# Spin-flip Z2 detection and full symmetry-group assembly.
# ---------------------------------------------------------------------------

_FLIP_OP = {0: 1, 1: 0, 2: 2}  # S+<->S-, Sz->Sz
_FLIP_SIGN = {0: 1.0, 1: 1.0, 2: -1.0}  # Sz picks up -1


def _single_key(op: int, site: int, c: complex):
    return (op, site, round(c.real, _ROUND), round(c.imag, _ROUND))


def _two_key(o1: int, s1: int, o2: int, s2: int, c: complex):
    # operators on distinct sites commute -> canonicalise the pair order.
    pair = tuple(sorted(((s1, o1), (s2, o2))))
    return (pair, round(c.real, _ROUND), round(c.imag, _ROUND))


def detect_spin_flip(op: SpinHalfOperator) -> bool:
    """True iff the Hamiltonian is invariant under the global spin flip
    ``F`` (``S^z -> -S^z``, ``S^+ <-> S^-`` on every site).

    The test compares the multiset of terms against its image under ``F``.
    Holds for, e.g., zero-field XXZ/Kitaev/anisotropic-exchange models and
    breaks once a longitudinal Zeeman term is added.
    """
    orig = Counter()
    flip = Counter()
    for o, site, c in op.single:
        orig[_single_key(o, site, c)] += 1
        fc = c * _FLIP_SIGN[o]
        flip[_single_key(_FLIP_OP[o], site, fc)] += 1
    for o1, s1, o2, s2, c in op.two:
        orig[_two_key(o1, s1, o2, s2, c)] += 1
        fc = c * _FLIP_SIGN[o1] * _FLIP_SIGN[o2]
        flip[_two_key(_FLIP_OP[o1], s1, _FLIP_OP[o2], s2, fc)] += 1
    return orig == flip


def detect_time_reversal(op: SpinHalfOperator) -> bool:
    """True iff the Hamiltonian commutes with the antiunitary time reversal
    ``T = U_F · K``.

    Here ``U_F`` is the unitary global spin flip (``S^z -> -S^z``,
    ``S^+ <-> S^-``; the same ``U`` used by :func:`detect_spin_flip`) and
    ``K`` is complex conjugation in the ``S^z`` basis. Because ``K``
    conjugates the c-number coefficients, this is exactly the spin-flip
    test applied to ``conj(H)``::

        T H T^{-1} = U_F (conj H) U_F^{-1}  ==  H  ?

    For real coefficients ``T`` and the unitary spin flip coincide; for a
    genuinely complex model (e.g. the ``gamma S^+ S^+ + conj(gamma) S^- S^-``
    pair term of non-Kramers quantum spin ice) the *unitary* flip is broken
    while ``T`` survives. ``U_F`` is a real permutation (bit complement) and
    squares to the identity, so ``T^2 = +1`` here -- the non-Kramers case
    with no protected Kramers degeneracy; the spectrum may be brought to
    real form rather than forced into degenerate pairs.
    """
    orig = Counter()
    trev = Counter()
    for o, site, c in op.single:
        orig[_single_key(o, site, c)] += 1
        tc = c.conjugate() * _FLIP_SIGN[o]
        trev[_single_key(_FLIP_OP[o], site, tc)] += 1
    for o1, s1, o2, s2, c in op.two:
        orig[_two_key(o1, s1, o2, s2, c)] += 1
        tc = c.conjugate() * _FLIP_SIGN[o1] * _FLIP_SIGN[o2]
        trev[_two_key(_FLIP_OP[o1], s1, _FLIP_OP[o2], s2, tc)] += 1
    return orig == trev


def build_symmetry_group(
    op: SpinHalfOperator,
    *,
    use_spin_flip: bool = True,
    max_group: int = 4096,
) -> tuple[AbelianGroup, bool]:
    """Assemble the maximal abelian symmetry group used for blocking.

    Returns ``(group, has_flip)`` where ``group`` is the maximal abelian
    subgroup of ``Aut x (Z2 spin-flip)?`` and ``has_flip`` reports whether
    any retained element carries a spin flip (in which case the dense
    layer must pair Sz sectors ``m`` and ``N-m``).
    """
    n = op.num_sites
    autos = find_automorphisms(op, max_group=max_group)
    elements: list[Element] = [(p, 0) for p in autos]
    if use_spin_flip and detect_spin_flip(op):
        elements += [(p, 1) for p in autos]
    group = maximal_abelian_subgroup(elements, n)
    has_flip = any(flip for _perm, flip in group.elements)
    return group, has_flip
