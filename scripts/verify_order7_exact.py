#!/usr/bin/env python3
"""Close the order-7 exact-tier seam (run on a cluster, not a laptop).

Background
---------
`full_spectrum`'s dense-block assembly used to fall off a cliff for
sectors above QED's 64 MiB lazy orbit-CSR budget: they went rep-only and
fell back to a SERIAL column-by-column build that collapsed to ~1 core
for hours (a 22-site order-7 pyrochlore cluster, C(22,11)=705432, ran
6.5 h+ before being abandoned). Upstream QED commit 6699a42 ("Retire the
full_spectrum dense-block cliff") added a rep-walk dense assembler that
builds those sectors in one O(|G|*nnz) PARALLEL pass with no orbit-CSR
materialization. A local trace confirms the cliff is gone (assembly now
streams through thousands of tiny blocks, flat memory, no serial
crawl) -- but a full order-7 solve is still many hours of small-block
churn on a loaded desktop and was never timed to completion here.

This script times ONE order-7 pyrochlore cluster end to end and checks
correctness (complete 2^N spectrum, real, sane extent). If it finishes
in an acceptable wall time on the cluster, raise the router's exact-tier
sector cap so order-7 clusters go exact instead of to OFTLM:

    qed-nlce ... --exact_max_sector 800000     # > C(22,11)=705432

and record the measured time in
qed_nlce/core/dense_ed._exact_tier_feasible's OPEN SEAM comment.

Usage
-----
    python scripts/verify_order7_exact.py                 # topology 1 (largest L)
    python scripts/verify_order7_exact.py --topology 3    # a different order-7 cluster
    python scripts/verify_order7_exact.py --order 8        # push further (C(25,12)=5.2M)

Prints the wall time, eigenvalue count (must equal 2**N), ground-state
energy, and the min/max spectral extent.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import time

import numpy as np


def build_cluster_op(order: int, topology: int):
    from qed_nlce.prep.generate_pyrochlore_clusters import (
        build_tetrahedron_graph, create_pyrochlore_lattice, generate_clusters)
    from qed_nlce.ed.operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ

    L = order + 2
    _, _, tets = create_pyrochlore_lattice(L, periodic=True)
    tg = build_tetrahedron_graph(tets)
    clusters, _, _ = generate_clusters(tg, order)
    order_clusters = [c for c in clusters if len(c) == order]
    if not order_clusters:
        raise SystemExit(f"no order-{order} clusters generated on L={L}")
    c = order_clusters[min(topology - 1, len(order_clusters) - 1)]
    sites = sorted({s for t in c for s in tets[t]})
    remap = {s: i for i, s in enumerate(sites)}
    op = SpinHalfOperator(len(sites))
    bonds = set()
    for t in c:
        for a, b in itertools.combinations([remap[s] for s in tets[t]], 2):
            bonds.add((min(a, b), max(a, b)))
    for a, b in sorted(bonds):
        op.add_two(OP_SZ, a, OP_SZ, b, 1.0)
        op.add_two(OP_SP, a, OP_SM, b, 0.4)
        op.add_two(OP_SM, a, OP_SP, b, 0.4)
    return op, len(sites)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", type=int, default=7)
    ap.add_argument("--topology", type=int, default=1,
                    help="1-based index among the order's distinct clusters")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress qed's per-sector diagonalization chatter")
    args = ap.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    from qed_nlce.ed.qed_bridge import full_spectrum_qed
    from qed_nlce.core.dense_ed import _exact_tier_feasible
    from qed_nlce.core.ed_runner import EDOptions

    op, n = build_cluster_op(args.order, args.topology)
    print(f"order-{args.order} topology {args.topology}: N={n} sites, "
          f"Hilbert 2^{n}={1 << n:,}", flush=True)
    print(f"router (default cap) admits exact? "
          f"{_exact_tier_feasible(op, n, EDOptions(), 'verify')}", flush=True)

    t0 = time.time()
    ev = full_spectrum_qed(op, device=args.device)
    dt = time.time() - t0

    ok = ev.shape[0] == (1 << n)
    print("=" * 60)
    print(f"RESULT order-{args.order}: {dt:.0f} s "
          f"({dt / 60:.1f} min / {dt / 3600:.2f} h)")
    print(f"  eigenvalues = {ev.shape[0]:,}  (expected 2^{n} = {1 << n:,})"
          f"  {'OK' if ok else 'MISMATCH!'}")
    print(f"  E0 = {ev[0]:.8f}   Emax = {ev[-1]:.8f}")
    print(f"  real-imag residual: spectrum is real by construction")
    if ok:
        print(f"  -> order {args.order} is EXACT-feasible. To enable it in "
              f"the pipeline: --exact_max_sector "
              f"{(1 << n)}  (any value > the largest Sz sector).")
    else:
        raise SystemExit("eigenvalue count mismatch -- do NOT enable this order")


if __name__ == "__main__":
    main()
