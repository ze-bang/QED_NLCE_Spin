#!/usr/bin/env python3
"""Accuracy + runtime benchmark: full_ed (truth) vs lanczos_boost vs ftlm.

Runs all three pipelines on the same triangular Heisenberg model at the
same max_order, using the on-disk eigenvalue cache so cluster
generation + Hamiltonian prep are paid only once. Then loads each
pipeline's NLC summation outputs and computes the per-temperature
relative deviation from the full-ED reference.

Outputs JSON + a markdown summary table to ``--report_dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

import numpy as np


PIPELINES = ["full_ed", "lanczos_boost", "ftlm", "auto"]


def run_pipeline(name: str, base_dir: str, args) -> dict:
    """Invoke ``qed-nlce`` for one pipeline; return runtime + status."""
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "qed_nlce",
        "--geometry", args.geometry,
        "--pipeline", name,
        "--max_order", str(args.max_order),
        "--base_dir", base_dir,
        "--J1", str(args.J1),
        "--temp_min", str(args.temp_min),
        "--temp_max", str(args.temp_max),
        "--temp_bins", str(args.temp_bins),
        "--thermo",
    ]
    # Pipeline-specific tuning: keep FTLM samples + Krylov respectable.
    if name == "ftlm":
        cmd += ["--ftlm_samples", str(args.ftlm_samples),
                "--krylov_dim", str(args.ftlm_krylov)]

    log_path = os.path.join(base_dir, "pipeline.log")
    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    return {
        "pipeline": name,
        "wall_seconds": dt,
        "exit_code": proc.returncode,
        "log": log_path,
    }


def _parse_peak_rss_mb(log_path: str) -> Optional[float]:
    """Parse ``Maximum resident set size`` (KiB) from a GNU ``/usr/bin/time
    -v`` log and convert to MiB. Returns ``None`` if the line is absent
    (e.g. ``/usr/bin/time`` unavailable)."""
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "Maximum resident set size" in line:
                    kb = float(line.rsplit(":", 1)[1].strip())
                    return kb / 1024.0
    except Exception:
        pass
    return None


def run_full_ed_method(method: str, base_dir: str, args) -> dict:
    """Run the ``full_ed`` pipeline with an explicit ``--method`` and
    capture wall time + peak RSS (via ``/usr/bin/time -v`` when present)."""
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "qed_nlce",
        "--geometry", args.geometry,
        "--pipeline", "full_ed",
        "--max_order", str(args.max_order),
        "--base_dir", base_dir,
        "--temp_min", str(args.temp_min),
        "--temp_max", str(args.temp_max),
        "--temp_bins", str(args.temp_bins),
        "--thermo",
        "--method", method,
    ]
    # --J1 is a triangular-geometry knob; pyrochlore rejects it.
    if str(args.geometry).startswith("triangular"):
        cmd += ["--J1", str(args.J1)]
    use_time = os.path.exists("/usr/bin/time")
    wrapped = (["/usr/bin/time", "-v"] + cmd) if use_time else cmd

    log_path = os.path.join(base_dir, "pipeline.log")
    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.run(wrapped, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0

    return {
        "method": method,
        "wall_seconds": dt,
        "peak_rss_mb": _parse_peak_rss_mb(log_path),
        "exit_code": proc.returncode,
        "log": log_path,
        "base_dir": base_dir,
        "nlc_dir": _find_nlc_results_dir(base_dir),
    }


def run_symmetry_ab(args) -> int:
    """A/B the dense vs symmetry-decomposed full_ed path on the same model:
    wall time, peak memory, and per-observable thermodynamic agreement
    (they should match to eigenvalue precision)."""
    os.makedirs(args.report_dir, exist_ok=True)
    print(f"[bench-ab] dense (FULL) vs symmetrized (FULL_SYMMETRIZED) "
          f"full_ed: geometry={args.geometry} max_order={args.max_order} "
          f"J1={args.J1}")

    variants = {
        "FULL": os.path.join(args.report_dir, "ab_dense"),
        "FULL_SYMMETRIZED": os.path.join(args.report_dir, "ab_symmetrized"),
    }
    runs: dict[str, dict] = {}
    for method, base in variants.items():
        print(f"[bench-ab] running full_ed --method={method} ...", flush=True)
        r = run_full_ed_method(method, base, args)
        runs[method] = r
        ok = r["exit_code"] == 0 and r["nlc_dir"]
        rss = f"{r['peak_rss_mb']:.1f} MB" if r["peak_rss_mb"] else "n/a"
        print(f"  -> {'OK' if ok else 'FAILED'}  "
              f"{r['wall_seconds']:.2f} s  peak={rss}")

    dense = runs["FULL"]
    sym = runs["FULL_SYMMETRIZED"]
    if dense["exit_code"] != 0 or sym["exit_code"] != 0 \
            or not dense["nlc_dir"] or not sym["nlc_dir"]:
        print("[bench-ab] FATAL: one of the runs failed", file=sys.stderr)
        return 2

    obs_files = {
        "energy": "nlc_energy.txt",
        "specific_heat": "nlc_specific_heat.txt",
        "entropy": "nlc_entropy.txt",
    }
    accuracy: dict = {}
    print("\n[bench-ab] thermodynamic agreement (symmetrized vs dense):")
    print(f"  {'observable':<14} {'max_abs':>12} {'mean_rel':>12}")
    for obs, fname in obs_files.items():
        cd = _load_curve(dense["nlc_dir"], fname)
        cs = _load_curve(sym["nlc_dir"], fname)
        if cd is None or cs is None:
            continue
        m = compare_curves(cd[0], cd[1], cs[0], cs[1])
        accuracy[obs] = m
        print(f"  {obs:<14} {m['max_abs']:>12.2e} {m['mean_rel']:>12.2e}")

    speedup = (dense["wall_seconds"] / sym["wall_seconds"]
               if sym["wall_seconds"] > 0 else None)
    mem_ratio = (dense["peak_rss_mb"] / sym["peak_rss_mb"]
                 if (dense["peak_rss_mb"] and sym["peak_rss_mb"]) else None)
    print("\n[bench-ab] summary:")
    print(f"  wall:  dense={dense['wall_seconds']:.2f}s  "
          f"sym={sym['wall_seconds']:.2f}s  "
          f"speedup={speedup:.2f}x" if speedup else "  wall: n/a")
    if mem_ratio:
        print(f"  peak:  dense={dense['peak_rss_mb']:.1f}MB  "
              f"sym={sym['peak_rss_mb']:.1f}MB  ratio={mem_ratio:.2f}x")

    report = {
        "mode": "symmetry_ab",
        "config": vars(args),
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "base_dir"}
                 for k, v in runs.items()},
        "accuracy_sym_vs_dense": accuracy,
        "speedup": speedup,
        "mem_ratio": mem_ratio,
    }
    out_json = os.path.join(args.report_dir, "symmetry_ab_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[bench-ab] wrote {out_json}")
    return 0


def _find_nlc_results_dir(base_dir: str) -> Optional[str]:
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if entry.startswith("nlc_results_order_") and os.path.isdir(full):
            return full
    return None


def _load_curve(d: str, fname: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    p = os.path.join(d, fname)
    if not os.path.isfile(p):
        return None
    try:
        data = np.loadtxt(p)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    return data[:, 0], data[:, 1]


def compare_curves(ref_T: np.ndarray, ref_Y: np.ndarray,
                   T: np.ndarray, Y: np.ndarray) -> dict:
    """Resample ``Y`` onto ``ref_T`` and compute per-T-band error metrics."""
    Y_on_ref = np.interp(ref_T, T, Y)
    diff = Y_on_ref - ref_Y
    abs_diff = np.abs(diff)
    denom = np.where(np.abs(ref_Y) > 1e-9, np.abs(ref_Y), np.nan)
    rel = abs_diff / denom

    bands = {
        "low (T<0.5)":  ref_T < 0.5,
        "mid (0.5-2)":  (ref_T >= 0.5) & (ref_T < 2.0),
        "high (T>=2)":  ref_T >= 2.0,
    }
    result: dict = {
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "mean_rel": float(np.nanmean(rel)),
        "max_rel": float(np.nanmax(rel)),
    }
    for band, mask in bands.items():
        if not np.any(mask):
            result[band] = None
            continue
        result[band] = {
            "mae": float(np.mean(abs_diff[mask])),
            "mean_rel": float(np.nanmean(rel[mask])),
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report_dir", required=True)
    p.add_argument("--geometry", default="triangular_site")
    p.add_argument("--max_order", type=int, default=5)
    p.add_argument("--J1", type=float, default=1.0)
    p.add_argument("--temp_min", type=float, default=0.1)
    p.add_argument("--temp_max", type=float, default=10.0)
    p.add_argument("--temp_bins", type=int, default=80)
    p.add_argument("--ftlm_samples", type=int, default=80)
    p.add_argument("--ftlm_krylov", type=int, default=200)
    p.add_argument("--symmetry_ab", action="store_true",
                   help="A/B the dense full_ed (--method FULL) against the "
                        "symmetry-decomposed full_ed (--method "
                        "FULL_SYMMETRIZED): wall time, peak RSS, and "
                        "thermodynamic agreement. Skips the FTLM/Lanczos "
                        "accuracy comparison.")
    args = p.parse_args()

    if args.symmetry_ab:
        return run_symmetry_ab(args)

    os.makedirs(args.report_dir, exist_ok=True)
    runs: dict[str, dict] = {}

    print(f"[bench] geometry={args.geometry}  max_order={args.max_order}  "
          f"J1={args.J1}  T in [{args.temp_min}, {args.temp_max}] x {args.temp_bins}")
    print(f"[bench] FTLM: samples={args.ftlm_samples}  krylov={args.ftlm_krylov}")
    print()

    # ---- Run all three pipelines ----
    for name in PIPELINES:
        base = os.path.join(args.report_dir, f"run_{name}")
        print(f"[bench] running {name} ...", flush=True)
        runs[name] = run_pipeline(name, base, args)
        runs[name]["base_dir"] = base
        nlc = _find_nlc_results_dir(base)
        runs[name]["nlc_dir"] = nlc
        status = "OK" if runs[name]["exit_code"] == 0 and nlc else "FAILED"
        print(f"  -> {status}  ({runs[name]['wall_seconds']:.2f} s, "
              f"nlc_dir={nlc})")

    # ---- Compare against full_ed reference ----
    ref = runs.get("full_ed")
    if ref is None or ref["exit_code"] != 0 or ref["nlc_dir"] is None:
        print("[bench] FATAL: full_ed reference run failed", file=sys.stderr)
        return 2

    obs_files = {
        "specific_heat": "nlc_specific_heat.txt",
        "energy":        "nlc_energy.txt",
        "entropy":       "nlc_entropy.txt",
    }

    report: dict = {
        "config": vars(args),
        "runs": {n: {k: v for k, v in r.items() if k != "base_dir"}
                 for n, r in runs.items()},
        "accuracy_vs_full_ed": {},
    }

    print("\n[bench] accuracy table (relative error vs full_ed):")
    print(f"  {'observable':<14} {'pipeline':<14} "
          f"{'mean_rel':>10} {'max_rel':>10} "
          f"{'low':>10} {'mid':>10} {'high':>10}")

    for obs, fname in obs_files.items():
        ref_curve = _load_curve(ref["nlc_dir"], fname)
        if ref_curve is None:
            print(f"  [{obs}] reference curve missing in {ref['nlc_dir']}")
            continue
        ref_T, ref_Y = ref_curve
        report["accuracy_vs_full_ed"][obs] = {}
        for name in ("lanczos_boost", "ftlm", "auto"):
            r = runs[name]
            if r["exit_code"] != 0 or r["nlc_dir"] is None:
                continue
            curve = _load_curve(r["nlc_dir"], fname)
            if curve is None:
                continue
            metrics = compare_curves(ref_T, ref_Y, *curve)
            report["accuracy_vs_full_ed"][obs][name] = metrics
            low = metrics["low (T<0.5)"]
            mid = metrics["mid (0.5-2)"]
            high = metrics["high (T>=2)"]
            print(f"  {obs:<14} {name:<14} "
                  f"{metrics['mean_rel']:>10.2e} "
                  f"{metrics['max_rel']:>10.2e} "
                  f"{(low['mean_rel'] if low else float('nan')):>10.2e} "
                  f"{(mid['mean_rel'] if mid else float('nan')):>10.2e} "
                  f"{(high['mean_rel'] if high else float('nan')):>10.2e}")

    # ---- Verdict ----
    print("\n[bench] runtime:")
    for name in PIPELINES:
        r = runs[name]
        print(f"  {name:<14}  {r['wall_seconds']:>8.2f} s  "
              f"(exit={r['exit_code']})")

    # Score: rank by mean_rel on specific_heat (most demanding observable),
    # plus runtime tiebreaker.
    sh = report["accuracy_vs_full_ed"].get("specific_heat", {})
    if sh:
        ranked = sorted(
            sh.items(),
            key=lambda kv: (kv[1]["mean_rel"], runs[kv[0]]["wall_seconds"]),
        )
        verdict_lines = []
        for i, (name, m) in enumerate(ranked):
            verdict_lines.append(
                f"  #{i + 1}  {name:<14}  C(T) mean_rel={m['mean_rel']:.2e}  "
                f"wall={runs[name]['wall_seconds']:.2f} s"
            )
        verdict = "Winner (vs full_ed reference): " + ranked[0][0]
        report["verdict"] = {
            "winner": ranked[0][0],
            "ranking": [{"pipeline": n, "mean_rel_C": m["mean_rel"],
                         "wall_seconds": runs[n]["wall_seconds"]}
                        for n, m in ranked],
        }
        print("\n[bench] verdict:")
        for line in verdict_lines:
            print(line)
        print(f"\n  *** {verdict} ***\n")

    out_json = os.path.join(args.report_dir, "benchmark_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[bench] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
