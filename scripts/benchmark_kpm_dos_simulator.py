#!/usr/bin/env python3
"""Per-cluster simulator for KPM-DOS thermodynamics.

Uses the *exact eigenvalue spectrum* (loaded from a finished full_ed run)
to evaluate the KPM-DOS estimator analytically:

    μ_k = Σ_n T_k((E_n - b)/a)      (exact; no Hutchinson noise here)
    μ_k^(stoch) = (D/R) Σ_r ⟨r|T_k|r⟩,  ⟨r|T_k|r⟩ = Σ_n |c_{r,n}|² T_k((E_n-b)/a)

We can synthesise the stochastic estimator by drawing R complex-Gaussian
``c_r`` vectors of length D and contracting against the moment table.

Then we apply the kernel and Chebyshev-Gauss quadrature exactly the same
way `kpm_dos.cpp` does, to get Z(β), E(β), C(β), S(β).

This isolates the *statistical* error — the only thing that's cluster-size
dependent.

Usage::

    python scripts/benchmark_kpm_dos_simulator.py \
        --finished_run /tmp/full_ed_o7 \
        --report_dir   /tmp/kpm_dos_sim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import h5py
import numpy as np


def thermo_from_eigs(eigs: np.ndarray, T: np.ndarray) -> dict[str, np.ndarray]:
    beta = 1.0 / T
    arg = -np.outer(beta, eigs - eigs.min())
    M = arg.max(axis=1, keepdims=True)
    Z_shifted = np.exp(arg - M).sum(axis=1)
    logZ = M.ravel() + np.log(Z_shifted) - beta * eigs.min()
    w = np.exp(arg - M)
    p = w / w.sum(axis=1, keepdims=True)
    E = (p * eigs[None, :]).sum(axis=1)
    E2 = (p * eigs[None, :] ** 2).sum(axis=1)
    C = (E2 - E ** 2) * beta ** 2
    F = -T * logZ
    S = (E - F) / T
    return {"energy": E, "specific_heat": C, "entropy": S}


def jackson_kernel(M: int) -> np.ndarray:
    k = np.arange(M)
    Mp1 = M + 1
    return ((Mp1 - k) * np.cos(np.pi * k / Mp1)
            + np.sin(np.pi * k / Mp1) / np.tan(np.pi / Mp1)) / Mp1


def kpm_dos_thermo(
    eigs: np.ndarray,
    T: np.ndarray,
    M: int,
    R: int,
    rng: np.random.Generator,
    buffer: float = 0.05,
) -> dict[str, np.ndarray]:
    D = len(eigs)
    e_min, e_max = eigs.min(), eigs.max()
    if e_max <= e_min:
        e_max = e_min + 1.0
    BW = e_max - e_min
    a = (BW + 2 * buffer * BW) / 2.0
    b = (e_max + e_min) / 2.0
    x = (eigs - b) / a  # in [-1, 1] (safely)
    shift = e_min

    # Precompute T_k(x_n) for n=0..D-1, k=0..M-1 by recursion (D x M).
    # Memory: D*M doubles. For D=2^14=16384, M=2048 → 256 MB. Use int M moderate.
    Tk = np.empty((D, M))
    Tk[:, 0] = 1.0
    if M > 1:
        Tk[:, 1] = x
    for k in range(2, M):
        Tk[:, k] = 2.0 * x * Tk[:, k - 1] - Tk[:, k - 2]

    # Stochastic Hutchinson: |c_r|² are drawn from D-dim complex Gaussian
    # normalized to unit norm (matches generateGaussianRandomVector).
    # Then ⟨r|T_k|r⟩ = Σ_n |c_{r,n}|² T_k(x_n).
    real = rng.standard_normal((R, D))
    imag = rng.standard_normal((R, D))
    p = real * real + imag * imag                       # ~ chi-square(2)
    p = p / p.sum(axis=1, keepdims=True)                # |c_{r,n}|² normalised
    # Sample-average then scale by D (Hutchinson normalisation).
    mu_stoch = D * (p @ Tk).mean(axis=0)                # length M

    # Apply Jackson kernel.
    g = jackson_kernel(M)
    mu_w = g * mu_stoch

    # Chebyshev-Gauss quadrature, N_quad = 2*M.
    N_q = 2 * M
    i = np.arange(N_q)
    xq = np.cos(np.pi * (i + 0.5) / N_q)
    Eq = b + a * xq
    # bracket = g_0 μ_0 + 2 Σ_{k≥1} g_k μ_k T_k(x_q)
    Tq = np.empty((N_q, M))
    Tq[:, 0] = 1.0
    if M > 1:
        Tq[:, 1] = xq
    for k in range(2, M):
        Tq[:, k] = 2.0 * xq * Tq[:, k - 1] - Tq[:, k - 2]
    bracket = mu_w[0] + 2.0 * (Tq[:, 1:] @ mu_w[1:])

    beta = 1.0 / T
    Es = np.empty_like(T)
    Cs = np.empty_like(T)
    for ti, b_ in enumerate(beta):
        w = np.exp(-b_ * (Eq - shift))
        Z = float((bracket * w).mean())
        if Z <= 0:
            Z = 1e-300
        E_mean = float((bracket * w * Eq).mean()) / Z
        E2_mean = float((bracket * w * Eq * Eq).mean()) / Z
        Es[ti] = E_mean
        Cs[ti] = (E2_mean - E_mean ** 2) * b_ ** 2

    return {"energy": Es, "specific_heat": Cs}


def err_metrics(truth: np.ndarray, est: np.ndarray) -> dict:
    diff = est - truth
    abs_err = np.abs(diff)
    denom = np.where(np.abs(truth) > 1e-9, np.abs(truth), np.nan)
    rel = abs_err / denom
    return {
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "mean_rel": float(np.nanmean(rel)),
        "max_rel": float(np.nanmax(rel)),
    }


def discover_clusters(base_dir: str) -> list[tuple[int, int, str]]:
    out = []
    for entry in sorted(os.listdir(base_dir)):
        if not entry.startswith("ed_results_order_"):
            continue
        ed_dir = os.path.join(base_dir, entry)
        for sub in sorted(os.listdir(ed_dir)):
            if not sub.startswith("cluster_"):
                continue
            try:
                _, cid, _, ordr = sub.split("_")
                cluster_id, order = int(cid), int(ordr)
            except Exception:
                continue
            h5 = os.path.join(ed_dir, sub, "output", "ed_results.h5")
            if os.path.isfile(h5):
                out.append((cluster_id, order, h5))
    return out


def load_eigs(h5_path: str) -> Optional[np.ndarray]:
    try:
        with h5py.File(h5_path, "r") as f:
            if "/eigendata/eigenvalues" in f:
                return np.sort(np.asarray(
                    f["/eigendata/eigenvalues"][:], dtype=float))
    except Exception:
        return None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--finished_run", required=True)
    p.add_argument("--report_dir", required=True)
    p.add_argument("--temp_min", type=float, default=0.1)
    p.add_argument("--temp_max", type=float, default=10.0)
    p.add_argument("--temp_bins", type=int, default=80)
    p.add_argument("--moments", nargs="+", type=int, default=[256, 1024, 2048])
    p.add_argument("--samples", nargs="+", type=int, default=[20, 60])
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    T = np.geomspace(args.temp_min, args.temp_max, args.temp_bins)

    clusters = discover_clusters(args.finished_run)
    if not clusters:
        print("no clusters", file=sys.stderr); return 2
    print(f"[bench] {len(clusters)} clusters")

    rng = np.random.default_rng(args.seed)

    # bin clusters by Hilbert dim
    by_dim: dict[int, list] = {}
    for (cid, ordr, h5) in clusters:
        eigs = load_eigs(h5)
        if eigs is None or len(eigs) < 4:
            continue
        D = len(eigs)
        by_dim.setdefault(D, []).append(eigs)

    print("D bins:", {D: len(v) for D, v in sorted(by_dim.items())})

    rows = []
    for D in sorted(by_dim.keys()):
        eigs_list = by_dim[D]
        n_sites = int(np.log2(D))
        for M in args.moments:
            if M > 4 * D:
                continue
            for R in args.samples:
                t0 = time.perf_counter()
                e_errs, c_errs = [], []
                for eigs in eigs_list:
                    truth = thermo_from_eigs(eigs, T)
                    est = kpm_dos_thermo(eigs, T, M=M, R=R, rng=rng)
                    e_errs.append(err_metrics(truth["energy"], est["energy"]))
                    c_errs.append(err_metrics(truth["specific_heat"],
                                              est["specific_heat"]))
                wall = time.perf_counter() - t0

                row = {
                    "D": D, "n_sites": n_sites, "n_clusters": len(eigs_list),
                    "M": M, "R": R, "wall_s": wall,
                    "C_median_rel": float(np.median([e["mean_rel"] for e in c_errs])),
                    "C_p95_rel":   float(np.percentile([e["mean_rel"] for e in c_errs], 95)),
                    "E_median_rel": float(np.median([e["mean_rel"] for e in e_errs])),
                }
                rows.append(row)
                print(f"  D={D:>5d} N={n_sites:>2d} M={M:>4d} R={R:>3d}  "
                      f"C med={row['C_median_rel']:.2e}  "
                      f"C p95={row['C_p95_rel']:.2e}  "
                      f"E med={row['E_median_rel']:.2e}  "
                      f"wall={wall:.1f}s")

    out_path = os.path.join(args.report_dir, "kpm_dos_simulator.json")
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\n[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
