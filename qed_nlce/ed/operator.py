"""In-memory spin-1/2 operator (self-contained; no qed / edlib).

A :class:`SpinHalfOperator` is a sum of single-site and two-site terms
expressed in the ladder/diagonal basis used throughout this package:

    op code 0 = S^+      (raising)
    op code 1 = S^-      (lowering)
    op code 2 = S^z      (diagonal, eigenvalues +-1/2)

This mirrors the ``Trans.dat`` / ``InterAll.dat`` encoding the project
historically produced via the external ``edlib`` helpers, but kept fully
in memory so the dense ED layer can build symmetry-adapted blocks without
any file round-trip or external dependency.

Computational basis convention
------------------------------
A basis state is an ``int`` whose bit ``i`` encodes site ``i``:

    bit = 1  -> spin up   (S^z = +1/2)
    bit = 0  -> spin down (S^z = -1/2)

so ``S^z_i = (bit_i - 1/2)`` and ``S^+_i`` flips a 0->1, ``S^-_i`` a 1->0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

OP_SP = 0  # S^+
OP_SM = 1  # S^-
OP_SZ = 2  # S^z

__all__ = ["SpinHalfOperator", "OP_SP", "OP_SM", "OP_SZ"]


@dataclass
class SpinHalfOperator:
    """A Hermitian spin-1/2 operator on ``num_sites`` sites.

    Terms are stored as plain Python tuples with complex coefficients:

    * single-site: ``(op, site, coeff)``
    * two-site:    ``(op1, site1, op2, site2, coeff)``

    Use :meth:`add_single`, :meth:`add_two` to accumulate terms. The
    caller is responsible for supplying a Hermitian set of terms (the
    dense builder does not symmetrize).
    """

    num_sites: int
    single: list[tuple[int, int, complex]] = field(default_factory=list)
    two: list[tuple[int, int, int, int, complex]] = field(default_factory=list)

    # -- construction ----------------------------------------------------

    def add_single(self, op: int, site: int, coeff: complex) -> None:
        if abs(coeff) < 1e-15:
            return
        self.single.append((int(op), int(site), complex(coeff)))

    def add_two(
        self, op1: int, site1: int, op2: int, site2: int, coeff: complex
    ) -> None:
        if abs(coeff) < 1e-15:
            return
        self.two.append(
            (int(op1), int(site1), int(op2), int(site2), complex(coeff))
        )

    # -- introspection used by symmetry detection ------------------------

    def conserves_sz(self) -> bool:
        """True iff every term commutes with total ``S^z``.

        Single-site terms must be ``S^z`` only; two-site terms must have
        matched raising/lowering content (S^z S^z, or S^+ S^- / S^- S^+).
        """
        for op, _site, _c in self.single:
            if op != OP_SZ:
                return False
        for op1, _s1, op2, _s2, _c in self.two:
            dm = _delta_m(op1) + _delta_m(op2)
            if dm != 0:
                return False
        return True

    def is_real_in_z_basis(self) -> bool:
        """True iff the Hamiltonian matrix is real in the S^z basis."""
        for _op, _site, c in self.single:
            if abs(c.imag) > 1e-12:
                return False
        for _o1, _s1, _o2, _s2, c in self.two:
            if abs(c.imag) > 1e-12:
                return False
        return True

    # -- dense / sparse construction -------------------------------------

    def matrix_on_basis(
        self,
        states: np.ndarray,
        index_of: dict[int, int],
        dtype=np.complex128,
    ) -> np.ndarray:
        """Build the dense Hamiltonian block on an explicit basis.

        ``states`` is an array of basis integers spanning an invariant
        subspace (e.g. a fixed-S^z sector or the whole space) and
        ``index_of`` maps each basis integer to its row/column index.
        Off-sector matrix elements (an operator taking a state outside
        ``states``) are dropped -- callers must pass a genuinely
        invariant subspace.
        """
        dim = len(states)
        H = np.zeros((dim, dim), dtype=dtype)
        for col, s in enumerate(states):
            for new_s, amp in self._apply_to_state(int(s)):
                row = index_of.get(new_s)
                if row is not None:
                    H[row, col] += amp
        return H

    def _apply_to_state(self, s: int) -> Iterable[tuple[int, complex]]:
        """Yield ``(new_state, amplitude)`` for ``H|s>`` (one column)."""
        # single-site terms
        for op, site, coeff in self.single:
            res = _apply_op(op, site, s)
            if res is not None:
                ns, amp = res
                yield ns, coeff * amp
        # two-site terms: apply op2 (site2) then op1 (site1)
        for op1, site1, op2, site2, coeff in self.two:
            res2 = _apply_op(op2, site2, s)
            if res2 is None:
                continue
            mid, amp2 = res2
            res1 = _apply_op(op1, site1, mid)
            if res1 is None:
                continue
            ns, amp1 = res1
            yield ns, coeff * amp1 * amp2


# ---------------------------------------------------------------------------
# Low-level single-operator action on a basis integer.
# ---------------------------------------------------------------------------


def _delta_m(op: int) -> int:
    """Change in total S^z (in units of 1) produced by a ladder op."""
    if op == OP_SP:
        return +1
    if op == OP_SM:
        return -1
    return 0


def _apply_op(op: int, site: int, s: int) -> Optional[tuple[int, float]]:
    """Apply one elementary operator to basis state ``s``.

    Returns ``(new_state, amplitude)`` or ``None`` if the operator
    annihilates the state. Amplitudes are real for spin-1/2 ladder /
    diagonal operators.
    """
    bit = (s >> site) & 1
    if op == OP_SZ:
        return s, (0.5 if bit else -0.5)
    if op == OP_SP:
        if bit:  # already up -> S^+ annihilates
            return None
        return s | (1 << site), 1.0
    if op == OP_SM:
        if not bit:  # already down -> S^- annihilates
            return None
        return s & ~(1 << site), 1.0
    raise ValueError(f"unknown op code {op}")
