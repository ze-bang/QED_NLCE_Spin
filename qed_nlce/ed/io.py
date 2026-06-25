"""Self-contained reader/writer for the ``Trans.dat`` / ``InterAll.dat``
Hamiltonian text format historically produced by ``edlib``.

The format is a five-line banner followed by one term per line::

    ===================
    num       <count>
    ===================
    ===================
    ===================
    <op> <site> <re> <im>                       # Trans.dat   (single-site)
    <op1> <s1> <op2> <s2> <re> <im>             # InterAll.dat (two-site)

Operator codes are the project convention (``0=S^+``, ``1=S^-``, ``2=S^z``).

Both reader and writer live here so the package no longer needs the
external ``edlib`` helpers. The writer emits full ``repr`` precision (the
legacy edlib path truncated couplings to six decimals via ``{:8f}``).
"""

from __future__ import annotations

import os

from .operator import SpinHalfOperator

__all__ = ["write_operator", "read_operator", "write_site_info"]

_BANNER = (
    "===================\n"
    "num   {count:>7d}\n"
    "===================\n"
    "===================\n"
    "===================\n"
)


def _fmt(x: float) -> str:
    return repr(float(x))


def write_operator(op: SpinHalfOperator, ham_subdir: str) -> None:
    """Write ``op`` to ``Trans.dat`` + ``InterAll.dat`` in ``ham_subdir``."""
    os.makedirs(ham_subdir, exist_ok=True)

    trans_path = os.path.join(ham_subdir, "Trans.dat")
    with open(trans_path, "w") as f:
        f.write(_BANNER.format(count=len(op.single)))
        for o, site, c in op.single:
            f.write(f"{o:9d} {site:9d}  {_fmt(c.real)} {_fmt(c.imag)}\n")

    inter_path = os.path.join(ham_subdir, "InterAll.dat")
    with open(inter_path, "w") as f:
        f.write(_BANNER.format(count=len(op.two)))
        for o1, s1, o2, s2, c in op.two:
            f.write(
                f"{o1:9d} {s1:9d} {o2:9d} {s2:9d}  "
                f"{_fmt(c.real)} {_fmt(c.imag)}\n"
            )


def _iter_data_rows(path: str):
    """Yield numeric token lists from one Trans/InterAll file, skipping
    the banner lines."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("=") or s.lower().startswith("num"):
                continue
            yield s.split()


def read_operator(ham_subdir: str, num_sites: int) -> SpinHalfOperator:
    """Reconstruct a :class:`SpinHalfOperator` from ``ham_subdir``."""
    op = SpinHalfOperator(num_sites=int(num_sites))

    for tok in _iter_data_rows(os.path.join(ham_subdir, "Trans.dat")):
        o, site = int(tok[0]), int(tok[1])
        re = float(tok[2])
        im = float(tok[3]) if len(tok) > 3 else 0.0
        op.add_single(o, site, complex(re, im))

    for tok in _iter_data_rows(os.path.join(ham_subdir, "InterAll.dat")):
        o1, s1, o2, s2 = int(tok[0]), int(tok[1]), int(tok[2]), int(tok[3])
        re = float(tok[4])
        im = float(tok[5]) if len(tok) > 5 else 0.0
        op.add_two(o1, s1, o2, s2, complex(re, im))

    return op


def write_site_info(
    ham_subdir: str, cluster_id: int, order: int, num_sites: int
) -> None:
    """Write a minimal ``*_site_info.dat`` (one line per site).

    The NLCE workflow only counts its non-comment lines to recover the
    site count, so a bare index per line suffices.
    """
    os.makedirs(ham_subdir, exist_ok=True)
    path = os.path.join(
        ham_subdir, f"cluster_{cluster_id}_order_{order}_site_info.dat"
    )
    with open(path, "w") as f:
        f.write("# site\n")
        for i in range(int(num_sites)):
            f.write(f"{i}\n")
