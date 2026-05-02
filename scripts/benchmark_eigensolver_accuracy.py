#!/usr/bin/env python3
"""Per-cluster accuracy benchmark of the three eigensolver strategies.

Walks a finished full_ed NLCE run, loads the exact eigenvalue spectrum
of every cluster from its ``ed_results.h5`` file, and computes
thermodynamics three ways:

  1. FULL ED (truth):  C(T), E(T), S(T) from the full spectrum.
  2. LANCZOS-BOOST emulation: same observables from the lowest-K
     eigenvalues only (truncated spectrum).
  3. FTLM emulation: Hutchinson-style stochastic estimator,
     diag(O) ≈ (D / R) Σ_r ⟨r| O |r⟩, where O = e^{-βH} f(H), and the
     ⟨r| e^{-βH} f(H) |r⟩ traces are themselves computed from the
     full spectrum projected onto a random Krylov subspace (the
     "tridiagonal model" used in real FTLM, faithfully simulated by
     drawing M random vectors and projecting onto a K-dim Krylov
     basis).

The advantage of this test is that all three estimators consume the
*same eigenvalue spectrum*, so any deviation is purely due to the
truncation / sampling strategy — not due to differences in the NLCE
summation kernel, the cluster discovery, or the cluster Hamiltonian
build. The answer this produces is the one that actually matters for
"which protocol should I use".

Usage::

    python scripts/benchmark_eigensolver_accuracy.py \
        --finished_run /tmp/pipeline_bench/run_full_ed \
        --report_dir   /tmp/eigensolver_bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


# ----------------------------- thermodynamics ------------------------------


def thermo_from_eigs(eigs: np.ndarray, T: np.ndarray) -> dict[str, np.ndarray]:
    """Exact thermodynamics from a (real) eigenvalue spectrum.

    Returns dict with keys energy, specific_heat, entropy.
    Uses log-sum-exp for numerical stability.
    """
    beta = 1.0 / T
    # shape (n_T, n_eig)
    arg = -np.outer(beta, eigs - eigs.min())  # subtract ground state for stability
    # logZ_shifted = log(sum_n e^{-beta (E_n - E_0)})
    M = arg.max(axis=1, keepdims=True)
    Z_shifted = np.exp(arg - M).sum(axis=1)
    logZ_shifted = M.ravel() + np.log(Z_shifted)
    # logZ = logZ_shifted - beta * E_0
    logZ = logZ_shifted - beta * eigs.min()

    # Boltzmann weights p_n = e^{-beta E_n} / Z
    w = np.exp(arg - M)  # unnormalized, shape (n_T, n_eig)
    Zsh = w.sum(axis=1)
    p = w / Zsh[:, None]

    E = (p * eigs[None, :]).sum(axis=1)
    E2 = (p * eigs[None, :] ** 2).sum(axis=1)
    C = (E2 - E ** 2) * beta ** 2
    F = -T * logZ
    S = (E - F) / T
    return {"energy": E, "specific_heat": C, "entropy": S}


# ----------------------------- FTLM emulator -------------------------------


def ftlm_thermo(
    eigs: np.ndarray,
    eigvecs: Optional[np.ndarray],  # not used; we synthesize H from eigs
    T: np.ndarray,
    n_samples: int,
    krylov_dim: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Faithful FTLM estimator computed from a known spectrum.

    We can't run real Lanczos here without the full Hamiltonian matrix,
    so we model what FTLM produces. The key observation: in an
    eigenbasis, ``H = diag(eigs)``, the FTLM partial-trace estimator
    over a random vector :math:`|r\rangle = \\sum_n c_n |n\\rangle`
    with i.i.d. Gaussian :math:`c_n` projects onto a K-dim Krylov
    subspace. After convergence (K large enough to span the
    populated thermal states), the FTLM estimator is unbiased; the
    only error is the Hutchinson-trace variance ~ 1/sqrt(M).

    To realistically simulate this we:
      * draw ``n_samples`` random unit vectors with shape (D,);
      * for each, diagonalize the K-dim Krylov tridiagonal built from
        ``H = diag(eigs)`` starting from that vector;
      * accumulate the trace estimators ``D * ⟨r| O |r⟩``.
    """
    D = len(eigs)
    K = min(krylov_dim, D)

    # Pre-compute T-grid pieces.
    beta = 1.0 / T
    nT = len(T)

    # Accumulators.
    Z_acc = np.zeros(nT)
    E_acc = np.zeros(nT)
    E2_acc = np.zeros(nT)

    for _ in range(n_samples):
        # Random unit vector in the eigenbasis.
        c = rng.standard_normal(D)
        c /= np.linalg.norm(c)

        # Build Krylov tridiagonal of H = diag(eigs) starting from c.
        # Lanczos with selective reorthogonalization (cheap because we
        # store the Krylov basis Q explicitly).
        Q = np.zeros((D, K + 1))
        Q[:, 0] = c
        alphas = np.zeros(K)
        betas = np.zeros(K)
        for j in range(K):
            v = eigs * Q[:, j]  # H q_j  (since H is diagonal in this basis)
            if j > 0:
                v -= betas[j - 1] * Q[:, j - 1]
            a = float(Q[:, j] @ v)
            alphas[j] = a
            v -= a * Q[:, j]
            # Selective reorthogonalization
            v -= Q[:, : j + 1] @ (Q[:, : j + 1].T @ v)
            b = float(np.linalg.norm(v))
            if b < 1e-12:
                # invariant subspace exhausted
                K_eff = j + 1
                alphas = alphas[:K_eff]
                betas = betas[:K_eff - 1] if K_eff > 1 else np.zeros(0)
                break
            betas[j] = b
            Q[:, j + 1] = v / b
        else:
            K_eff = K

        # Diagonalize the (K_eff x K_eff) tridiagonal.
        T_mat = np.diag(alphas) + np.diag(betas[:max(K_eff - 1, 0)], 1) \
                                + np.diag(betas[:max(K_eff - 1, 0)], -1)
        e_K, V_K = np.linalg.eigh(T_mat)
        # Components of the starting vector in the Krylov-basis eigenstates.
        # In Lanczos, q_0 is the starting unit vector; ⟨q_0 | v_k⟩ = V_K[0, k].
        u0 = V_K[0, :]

        # For each T, accumulate Hutchinson estimator.
        for ti, b in enumerate(beta):
            arg = -b * (e_K - e_K.min())
            wK = np.exp(arg)  # un-normalized Boltzmann weights in Krylov basis
            # ⟨r| e^{-βH} |r⟩ = Σ_k |u0_k|² e^{-β e_k}
            zr = float((u0 ** 2 * wK).sum())
            er = float((u0 ** 2 * wK * e_K).sum())
            er2 = float((u0 ** 2 * wK * e_K ** 2).sum())
            # Renormalize the shift (we computed against e_K.min(), restore)
            # Note: the shift cancels in C(T) but matters for E,S — track it.
            shift = e_K.min()
            # Z(T) ∝ D * zr * e^{-β shift}, so log-shift accumulates additively.
            Z_acc[ti] += D * zr * np.exp(-b * shift)
            E_acc[ti] += D * er * np.exp(-b * shift)
            E2_acc[ti] += D * er2 * np.exp(-b * shift)

    Z = Z_acc / n_samples
    E = (E_acc / n_samples) / Z
    E2 = (E2_acc / n_samples) / Z
    C = (E2 - E ** 2) * beta ** 2
    # Z is the unbiased Hutchinson estimator of Tr(e^{-βH}), i.e. the
    # full partition function. F = -T log Z, S = (E - F)/T.
    F = -T * np.log(np.maximum(Z, 1e-300))
    S = (E - F) / T
    return {"energy": E, "specific_heat": C, "entropy": S}


def lanczos_boost_thermo(eigs: np.ndarray, n_keep: int,
                         T: np.ndarray) -> dict[str, np.ndarray]:
    """Thermodynamics from the lowest ``n_keep`` eigenvalues only."""
    if n_keep >= len(eigs):
        return thermo_from_eigs(eigs, T)
    truncated = np.sort(eigs)[:n_keep]
    return thermo_from_eigs(truncated, T)


# --------------------------- per-cluster runner ----------------------------


@dataclass
class ClusterEntry:
    cluster_id: int
    order: int
    h5_path: str
    multiplicity: float = 1.0


def discover_clusters(base_dir: str) -> list[ClusterEntry]:
    """Walk ``base_dir/ed_results_order_*/cluster_*_order_*/output/ed_results.h5``."""
    out = []
    for entry in sorted(os.listdir(base_dir)):
        if not entry.startswith("ed_results_order_"):
            continue
        ed_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(ed_dir):
            continue
        for sub in sorted(os.listdir(ed_dir)):
            if not sub.startswith("cluster_"):
                continue
            try:
                _, cid, _, ordr = sub.split("_")
                cluster_id = int(cid)
                order = int(ordr)
            except Exception:
                continue
            h5 = os.path.join(ed_dir, sub, "output", "ed_results.h5")
            if os.path.isfile(h5):
                out.append(ClusterEntry(cluster_id, order, h5))
    return out


def load_eigs(h5_path: str) -> Optional[np.ndarray]:
    try:
        with h5py.File(h5_path, "r") as f:
            if "/eigendata/eigenvalues" in f:
                eigs = np.asarray(f["/eigendata/eigenvalues"][:], dtype=float)
                return np.sort(eigs)
    except Exception:
        return None
    return None


# ----------------------- accuracy report aggregation -----------------------


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--finished_run", required=True,
                   help="A finished full_ed NLCE base_dir.")
    p.add_argument("--report_dir", required=True)
    p.add_argument("--temp_min", type=float, default=0.1)
    p.add_argument("--temp_max", type=float, default=10.0)
    p.add_argument("--temp_bins", type=int, default=80)
    p.add_argument("--ftlm_samples", nargs="+", type=int,
                   default=[20, 80, 200])
    p.add_argument("--ftlm_krylov", nargs="+", type=int,
                   default=[20, 50, 100])
    p.add_argument("--lb_n_keep", nargs="+", type=int,
                   default=[5, 20, 50, 200])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    T = np.geomspace(args.temp_min, args.temp_max, args.temp_bins)

    clusters = discover_clusters(args.finished_run)
    if not clusters:
        print(f"No clusters found under {args.finished_run}", file=sys.stderr)
        return 2
    print(f"[bench] {len(clusters)} clusters discovered under "
          f"{args.finished_run}")

    rng = np.random.default_rng(args.seed)

    # Per-cluster accumulator.
    # results[method][config_key][observable] -> list of err_metrics (per cluster)
    results: dict = {"ftlm": {}, "lanczos_boost": {}}
    runtime: dict = {"ftlm": {}, "lanczos_boost": {}}

    obs_keys = ("energy", "specific_heat", "entropy")
    skipped = 0

    for cl in clusters:
        eigs = load_eigs(cl.h5_path)
        if eigs is None or len(eigs) < 2:
            skipped += 1
            continue
        truth = thermo_from_eigs(eigs, T)

        # Lanczos-boost emulator: keep K lowest eigenvalues.
        for K in args.lb_n_keep:
            cfg = f"K={K}"
            results["lanczos_boost"].setdefault(cfg, [])
            runtime["lanczos_boost"].setdefault(cfg, 0.0)
            t0 = time.perf_counter()
            est = lanczos_boost_thermo(eigs, K, T)
            runtime["lanczos_boost"][cfg] += time.perf_counter() - t0
            entry = {obs: err_metrics(truth[obs], est[obs]) for obs in obs_keys}
            entry["num_sites"] = int(np.log2(len(eigs)).round())
            entry["hilbert_dim"] = int(len(eigs))
            results["lanczos_boost"][cfg].append(entry)

        # FTLM emulator: M random vectors x K-dim Krylov.
        for M in args.ftlm_samples:
            for Kk in args.ftlm_krylov:
                if Kk > len(eigs):
                    continue
                cfg = f"M={M},K={Kk}"
                results["ftlm"].setdefault(cfg, [])
                runtime["ftlm"].setdefault(cfg, 0.0)
                t0 = time.perf_counter()
                est = ftlm_thermo(eigs, None, T, n_samples=M,
                                   krylov_dim=Kk, rng=rng)
                runtime["ftlm"][cfg] += time.perf_counter() - t0
                entry = {obs: err_metrics(truth[obs], est[obs]) for obs in obs_keys}
                entry["num_sites"] = int(np.log2(len(eigs)).round())
                entry["hilbert_dim"] = int(len(eigs))
                results["ftlm"][cfg].append(entry)

    # ----- aggregate per (method, config) across clusters -----
    summary: dict = {"ftlm": {}, "lanczos_boost": {}}
    for method in ("ftlm", "lanczos_boost"):
        for cfg, per_cluster in results[method].items():
            agg: dict = {"n_clusters": len(per_cluster),
                         "wall_seconds": runtime[method][cfg]}
            for obs in obs_keys:
                rels = [pc[obs]["mean_rel"] for pc in per_cluster
                        if np.isfinite(pc[obs]["mean_rel"])]
                maxs = [pc[obs]["max_rel"] for pc in per_cluster
                        if np.isfinite(pc[obs]["max_rel"])]
                agg[obs] = {
                    "median_rel": float(np.median(rels)) if rels else float("nan"),
                    "mean_rel":   float(np.mean(rels)) if rels else float("nan"),
                    "p95_rel":    float(np.percentile(rels, 95)) if rels else float("nan"),
                    "max_rel":    float(np.max(maxs)) if maxs else float("nan"),
                }
            summary[method][cfg] = agg

    report = {
        "config": vars(args),
        "n_clusters_used": sum(1 for _ in clusters) - skipped,
        "n_clusters_skipped": skipped,
        "summary": summary,
    }
    out_json = os.path.join(args.report_dir, "eigensolver_accuracy.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[bench] wrote {out_json}\n")

    # ----- table ------
    print("=" * 100)
    print(f"{'method':<14} {'config':<15} {'C: med rel':>12} "
          f"{'C: p95 rel':>12} {'E: med rel':>12} {'S: med rel':>12} "
          f"{'wall (s)':>10}")
    print("=" * 100)
    for method in ("lanczos_boost", "ftlm"):
        for cfg, agg in sorted(summary[method].items()):
            print(f"{method:<14} {cfg:<15} "
                  f"{agg['specific_heat']['median_rel']:>12.2e} "
                  f"{agg['specific_heat']['p95_rel']:>12.2e} "
                  f"{agg['energy']['median_rel']:>12.2e} "
                  f"{agg['entropy']['median_rel']:>12.2e} "
                  f"{agg['wall_seconds']:>10.2f}")
        print("-" * 100)

    # ----- verdict -----
    # Score: lowest p95 relative error on C(T), tiebreaker = wall time.
    candidates = []
    for method in ("lanczos_boost", "ftlm"):
        for cfg, agg in summary[method].items():
            candidates.append((method, cfg, agg["specific_heat"]["p95_rel"],
                               agg["wall_seconds"]))
    candidates.sort(key=lambda x: (x[2], x[3]))
    print("\n[bench] verdict (lowest p95 C(T) relative error, then runtime):")
    for i, (m, c, p95, w) in enumerate(candidates[:5]):
        print(f"  #{i + 1}  {m:<14} {c:<15}  p95(C)={p95:.2e}  wall={w:.2f}s")
    print(f"\n  *** Best protocol: {candidates[0][0]} with {candidates[0][1]} ***\n")
    report["verdict"] = {
        "ranking": [
            {"method": m, "config": c, "p95_C_rel": p, "wall_seconds": w}
            for (m, c, p, w) in candidates[:10]
        ]
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
