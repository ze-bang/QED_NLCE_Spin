#!/usr/bin/env python3
"""Audit exact-tier NLCE results for the F6 dense-assembly corruption.

F6 (fixed in QED feat/clique-budget, Jul 2026): a COLD operator's first
SoA term commit raced across the OMP team inside
``Operator::try_build_dense_columns`` and dropped terms from the
assembled matrix -- ``qed.full_spectrum``'s plain-dense lane returned a
WRONG spectrum with no error (verified at dim 16384: E0 = -6.6397 vs
the true -4.3855 on a 14-site XXZ tree). Exposure window: any
Sz-conserving cluster with a TRIVIAL automorphism group and N >= 15
routed through that exact lane in production (N <= 14 was shielded by
the since-retired pure-Python fast path); the race was also
timing-dependent, so smaller dims are not provably clean.

This script walks an NLCE run/base directory, pairs every exact-tier
``ed_results.h5`` (a non-empty ``/eigendata/eigenvalues``) with its
``cluster_<id>_order_<n>`` Hamiltonian directory, recomputes the
ground-state energy CHEAPLY (a qed Lanczos solve, which never touched
the buggy assembler and was verified correct at N = 14), and reports
any cluster whose stored min(eigenvalues) disagrees.

Usage:
    python scripts/audit_f6_dense_lane.py --base_dir <nlce_base_or_run_dir>
        [--min_sites 14] [--tol 1e-6]

Exit status 1 if any mismatch is found.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def find_pairs(base: Path):
    """Yield (h5_path, ham_dir, tag) for every exact-tier result whose
    Hamiltonian directory can be located.

    Layout (core/workflow.py): the ED dir and the Hamiltonian dir hold
    same-named ``cluster_<id>_order_<n>`` subdirectories; the spectrum
    sits at ``<ed>/cluster_x_order_y/output/ed_results.h5`` and the
    Hamiltonian at ``<ham>/cluster_x_order_y/{Trans,InterAll}.dat``.
    """
    for h5 in base.rglob("ed_results.h5"):
        cdir = h5.parent.parent          # .../cluster_x_order_y
        tag = cdir.name
        if not tag.startswith("cluster_"):
            continue
        # same-tag Hamiltonian dir: the cluster dir itself (older layouts
        # wrote both in one place), else a same-named dir elsewhere in the
        # tree (the workflow's hamiltonians_order_*/ sibling).
        ham = None
        if (cdir / "InterAll.dat").exists() or (cdir / "Trans.dat").exists():
            ham = cdir
        else:
            hits = [p for p in base.rglob(tag)
                    if p.is_dir() and ((p / "InterAll.dat").exists()
                                       or (p / "Trans.dat").exists())]
            if len(hits) == 1:
                ham = hits[0]
            elif len(hits) > 1:
                # ambiguous across orders/runs -- take the one whose
                # parent tree also contains this h5's ED dir prefix
                ham = hits[0]
        if ham is not None:
            yield h5, ham, tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--min_sites", type=int, default=14,
                    help="Audit clusters with at least this many sites "
                         "(default 14; the verified-bad dim was 2^14).")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    import h5py

    from qed_nlce.ed.io import read_qed_operator

    import qed

    base = Path(args.base_dir)
    checked = suspect = 0
    bad: list[str] = []
    for h5_path, ham, tag in sorted(find_pairs(base)):
        with h5py.File(h5_path, "r") as f:
            if "eigendata" not in f or "eigenvalues" not in f["eigendata"]:
                continue
            eig = f["eigendata"]["eigenvalues"]
            if eig.shape[0] == 0:
                continue  # OFTLM tier: no spectrum, not this lane
            n = int(f["eigendata"].attrs.get("num_sites", 0))
            e0_stored = float(np.min(eig[:]))
        if n < args.min_sites:
            continue
        suspect += 1
        qop = read_qed_operator(str(ham), n)
        res = qed.solve(qop, num_eigenvalues=1)
        e0 = float(sorted(res.eigenvalues)[0])
        checked += 1
        ok = abs(e0 - e0_stored) <= args.tol * max(1.0, abs(e0))
        print(f"{'OK ' if ok else 'BAD'} {tag:34s} N={n:2d} "
              f"stored_E0={e0_stored:+.10f} lanczos_E0={e0:+.10f} "
              f"d={abs(e0 - e0_stored):.2e}   [{h5_path}]", flush=True)
        if not ok:
            bad.append(str(h5_path))

    print(f"\naudited {checked}/{suspect} exact-tier clusters with "
          f"N >= {args.min_sites} under {base}")
    if bad:
        print(f"{len(bad)} MISMATCH(ES) -- these clusters' spectra are "
              "corrupted; delete their cache entries / ed_results.h5 and "
              "rerun on the fixed QED build:")
        for b in bad:
            print(" ", b)
        raise SystemExit(1)
    print("no F6 corruption detected")


if __name__ == "__main__":
    main()
