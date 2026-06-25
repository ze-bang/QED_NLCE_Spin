#!/usr/bin/env python3
"""Benchmark: symmetry-adapted dense ED vs plain dense ED.

For a range of Heisenberg-ring sizes this measures

  * wall-clock time for the full eigenvalue spectrum with symmetry
    reduction on vs off, and
  * the size of the largest dense block the solver actually has to
    diagonalize (the cubic cost driver),

and checks the two spectra agree to machine precision.  This is the
single source of truth for "does exhausting every symmetry actually
buy us higher order?".

Usage::

    python scripts/benchmark_symmetry.py            # default sizes
    python scripts/benchmark_symmetry.py --max-n 14 --jz 0.7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from qed_nlce.ed import (  # noqa: E402
    OP_SM,
    OP_SP,
    OP_SZ,
    SpinHalfOperator,
    solve_spectrum,
)


def heisenberg_ring(n: int, jz: float = 1.0) -> SpinHalfOperator:
    op = SpinHalfOperator(n)
    for i in range(n):
        j = (i + 1) % n
        op.add_two(OP_SZ, i, OP_SZ, j, jz)
        op.add_two(OP_SP, i, OP_SM, j, 0.5)
        op.add_two(OP_SM, i, OP_SP, j, 0.5)
    return op


def _time_solve(op: SpinHalfOperator, *, use_symmetry: bool):
    t0 = time.perf_counter()
    if use_symmetry:
        spec, report = solve_spectrum(op, use_symmetry=True, return_report=True)
        largest = report.largest_block
    else:
        spec = solve_spectrum(op, use_symmetry=False)
        largest = 1 << op.num_sites
    dt = time.perf_counter() - t0
    return np.sort(spec), largest, dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=6)
    ap.add_argument("--max-n", type=int, default=12)
    ap.add_argument("--jz", type=float, default=1.0)
    args = ap.parse_args()

    header = (
        f"{'N':>3} {'dim':>7} {'block_sym':>10} {'t_plain[s]':>11} "
        f"{'t_sym[s]':>10} {'speedup':>8} {'maxdiff':>10}"
    )
    print(header)
    print("-" * len(header))

    for n in range(args.min_n, args.max_n + 1, 2):
        op = heisenberg_ring(n, jz=args.jz)
        spec_sym, block_sym, t_sym = _time_solve(op, use_symmetry=True)
        spec_plain, _, t_plain = _time_solve(op, use_symmetry=False)
        maxdiff = float(np.max(np.abs(spec_sym - spec_plain)))
        speedup = t_plain / t_sym if t_sym > 0 else float("inf")
        print(
            f"{n:>3} {1 << n:>7} {block_sym:>10} {t_plain:>11.3f} "
            f"{t_sym:>10.3f} {speedup:>7.2f}x {maxdiff:>10.1e}"
        )
        assert maxdiff < 1e-9, f"spectra disagree at N={n}: {maxdiff}"

    print("\nAll symmetry-reduced spectra match the plain dense spectrum.")


if __name__ == "__main__":
    main()
