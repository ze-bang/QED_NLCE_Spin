"""Multi-resummer convergence cross-validation report.

Given a finished NLCE base directory (one for which the workflow has
already produced ED outputs), re-run the NLC summation step once per
resummer in :data:`SUPPORTED_RESUMMERS` and emit a report on how
much the final thermodynamic curves disagree across the resummers.

The inter-method spread (max - min, or std) at each temperature is an
*unbiased* estimate of the truncation-induced systematic error of
the NLCE: if all accelerators agree to within a few percent, the
series is well-converged at that T; if they disagree wildly, the
series is barely converging there and the user should either go to
higher orders or restrict the temperature window.

CLI:

    qed-nlce-convergence-report \
        --base_dir /path/to/finished_nlce_run \
        [--kernel auto|standard|triangular|ftlm] \
        [--temp_min ...] [--temp_max ...] [--temp_bins ...]

Outputs (under ``<base_dir>/convergence_report/``):

* ``nlc_convergence_report.json``: full per-method + spread payload.
* ``nlc_convergence_report.png``: comparison plot (4 panels:
  C, S, E, free energy F) of every resummer's curve.
* ``per_method/<method>/`` : raw NLC kernel outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np

from ..geometries import _paths


# Per-kernel metadata: (script_path, supported_resummers, flag_name).
# The triangular kernel uses ``--resummation`` with one set of options,
# the FTLM and full kernels use a different (overlapping) set.
KERNELS = {
    "triangular": {
        "script": _paths.NLC_SUM_TRIANGULAR,
        "flag": "--resummation",
        "resummers": ["none", "euler", "wynn", "wynn_multi",
                      "brezinski", "aitken", "pade"],
        "ed_flag": "--eigenvalue_dir",
        "supports_plot": False,
    },
    "ftlm": {
        "script": _paths.NLC_SUM_FTLM,
        "flag": "--resummation",
        "resummers": ["direct", "euler", "wynn", "robust"],
        "ed_flag": "--ftlm_dir",
        "supports_plot": True,
    },
    "standard": {
        "script": _paths.NLC_SUM_FULL,
        "flag": "--resummation_method",
        "resummers": ["direct", "euler", "wynn", "shanks", "pade"],
        "ed_flag": "--eigenvalue_dir",
        "supports_plot": True,
    },
}


def _detect_kernel(base_dir: str) -> str:
    """Best-effort kernel auto-detect.

    Looks for marker files / directory names in the NLCE base dir.
    Defaults to 'ftlm' (which handles mixed FULL+FTLM HDF5 inputs).
    """
    entries = set(os.listdir(base_dir)) if os.path.isdir(base_dir) else set()
    if any(name.startswith("triangular_") or "triangular" in name for name in entries):
        # Triangular workflow uses a distinct cluster_dir prefix.
        for entry in entries:
            if entry.startswith("clusters_order_") or "triangular" in entry.lower():
                return "triangular"
    return "ftlm"


def _find_subdir(base_dir: str, prefix: str) -> Optional[str]:
    candidates = sorted(
        d for d in os.listdir(base_dir)
        if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))
    )
    if not candidates:
        return None
    return os.path.join(base_dir, candidates[-1])


def _parse_curve(path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Parse a (T, value, [error]) text file emitted by the NLC kernels."""
    if not os.path.isfile(path):
        return None
    try:
        data = np.loadtxt(path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    return data[:, 0], data[:, 1]


def _run_kernel_one_method(
    kernel: dict,
    cluster_info_dir: str,
    ed_dir: str,
    out_dir: str,
    method: str,
    temp_min: float,
    temp_max: float,
    temp_bins: int,
) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        kernel["script"],
        f"--cluster_dir={cluster_info_dir}",
        f"{kernel['ed_flag']}={ed_dir}",
        f"--output_dir={out_dir}",
        f"--temp_min={temp_min}",
        f"--temp_max={temp_max}",
        f"--temp_bins={temp_bins}",
        f"{kernel['flag']}={method}",
    ]
    if kernel.get("supports_plot", False):
        cmd.append("--plot")
    log_path = os.path.join(out_dir, "kernel.log")
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        return res.returncode == 0
    except Exception as exc:
        logging.error("Kernel run for method=%s failed: %s", method, exc)
        return False


def _spread(curves: dict[str, np.ndarray]) -> dict:
    """Compute inter-method spread statistics at each temperature."""
    if not curves:
        return {}
    stack = np.stack(list(curves.values()), axis=0)  # (n_methods, n_T)
    return {
        "min": stack.min(axis=0).tolist(),
        "max": stack.max(axis=0).tolist(),
        "median": np.median(stack, axis=0).tolist(),
        "std": stack.std(axis=0).tolist(),
        "range": (stack.max(axis=0) - stack.min(axis=0)).tolist(),
    }


def _make_plot(report: dict, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable; skipping plot: %s", exc)
        return

    observables = [k for k in ("specific_heat", "energy", "entropy", "free_energy")
                   if k in report["observables"]]
    if not observables:
        return

    fig, axes = plt.subplots(1, len(observables), figsize=(5 * len(observables), 4))
    if len(observables) == 1:
        axes = [axes]

    for ax, obs in zip(axes, observables):
        block = report["observables"][obs]
        T = np.asarray(block["temperature"])
        for method, vals in block["per_method"].items():
            ax.plot(T, vals, label=method, alpha=0.85)
        if "spread" in block and block["spread"]:
            spread_min = np.asarray(block["spread"]["min"])
            spread_max = np.asarray(block["spread"]["max"])
            ax.fill_between(T, spread_min, spread_max, color="grey", alpha=0.15,
                            label="inter-method range")
        ax.set_xscale("log")
        ax.set_xlabel("T")
        ax.set_ylabel(obs)
        ax.set_title(obs)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("NLCE multi-resummer convergence report")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-resummer NLCE convergence cross-validation report.",
    )
    parser.add_argument("--base_dir", required=True,
                        help="A finished NLCE workflow directory.")
    parser.add_argument("--kernel", choices=["auto", "standard", "triangular", "ftlm"],
                        default="auto", help="NLC summation kernel to use.")
    parser.add_argument("--cluster_dir", default=None,
                        help="Override cluster_info dir auto-detection.")
    parser.add_argument("--ed_dir", default=None,
                        help="Override ED-results dir auto-detection.")
    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_max", type=float, default=10.0)
    parser.add_argument("--temp_bins", type=int, default=200)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of resummers to run (default: all kernel-supported).")
    parser.add_argument("--out_subdir", default="convergence_report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.path.isdir(args.base_dir):
        logging.error("base_dir not found: %s", args.base_dir)
        return 2

    # Resolve kernel and locate the cluster / ED directories.
    kernel_name = args.kernel
    if kernel_name == "auto":
        kernel_name = _detect_kernel(args.base_dir)
    kernel = KERNELS[kernel_name]
    logging.info("Using NLC kernel: %s (%s)", kernel_name, kernel["script"])

    cluster_dir = args.cluster_dir or _find_subdir(args.base_dir, "clusters_order_")
    ed_dir = args.ed_dir or _find_subdir(args.base_dir, "ed_results_order_")
    if cluster_dir is None or ed_dir is None:
        logging.error("Could not locate cluster_dir or ed_dir under %s", args.base_dir)
        return 2

    # The actual cluster_info subdirectory lives inside cluster_dir.
    cluster_info = None
    for candidate in (
        os.path.join(cluster_dir, f"cluster_info_order_{os.path.basename(cluster_dir).split('_')[-1]}"),
        cluster_dir,
    ):
        if os.path.isdir(candidate):
            cluster_info = candidate
            break
    if cluster_info is None:
        logging.error("Could not locate cluster_info dir under %s", cluster_dir)
        return 2

    methods = args.methods or kernel["resummers"]
    out_root = os.path.join(args.base_dir, args.out_subdir)
    os.makedirs(out_root, exist_ok=True)

    report: dict = {
        "base_dir": os.path.abspath(args.base_dir),
        "kernel": kernel_name,
        "methods_requested": methods,
        "methods_succeeded": [],
        "methods_failed": [],
        "temp_range": [args.temp_min, args.temp_max, args.temp_bins],
        "observables": {},
    }

    # ---- Run kernel per method ----
    per_method_outputs: dict[str, str] = {}
    for method in methods:
        sub = os.path.join(out_root, "per_method", method)
        ok = _run_kernel_one_method(
            kernel, cluster_info, ed_dir, sub, method,
            args.temp_min, args.temp_max, args.temp_bins,
        )
        if ok:
            report["methods_succeeded"].append(method)
            per_method_outputs[method] = sub
            logging.info("[%s] OK -> %s", method, sub)
        else:
            report["methods_failed"].append(method)
            logging.warning("[%s] FAILED (see %s/kernel.log)", method, sub)

    # ---- Aggregate per-observable curves + spread ----
    obs_files = {
        "specific_heat": "nlc_specific_heat.txt",
        "energy": "nlc_energy.txt",
        "entropy": "nlc_entropy.txt",
        "free_energy": "nlc_free_energy.txt",
    }
    for obs_name, fname in obs_files.items():
        per_method_curves: dict[str, list[float]] = {}
        T_ref: Optional[np.ndarray] = None
        for method, dirpath in per_method_outputs.items():
            parsed = _parse_curve(os.path.join(dirpath, fname))
            if parsed is None:
                continue
            T, V = parsed
            if T_ref is None:
                T_ref = T
            elif len(T_ref) != len(T):
                # Reinterpolate onto the reference grid.
                V = np.interp(T_ref, T, V)
            per_method_curves[method] = np.asarray(V).tolist()
        if not per_method_curves or T_ref is None:
            continue
        spread = _spread({m: np.asarray(v) for m, v in per_method_curves.items()})
        report["observables"][obs_name] = {
            "temperature": T_ref.tolist(),
            "per_method": per_method_curves,
            "spread": spread,
        }

    # ---- Persist report + plot ----
    json_path = os.path.join(out_root, "nlc_convergence_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote %s", json_path)

    plot_path = os.path.join(out_root, "nlc_convergence_report.png")
    _make_plot(report, plot_path)
    if os.path.isfile(plot_path):
        logging.info("Wrote %s", plot_path)

    # ---- Console summary ----
    n_ok = len(report["methods_succeeded"])
    n_fail = len(report["methods_failed"])
    logging.info("Convergence report: %d / %d resummers succeeded.",
                 n_ok, n_ok + n_fail)
    for obs_name, block in report["observables"].items():
        if not block["spread"]:
            continue
        rng = np.asarray(block["spread"]["range"])
        med = np.asarray(block["spread"]["median"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(rng) / np.where(np.abs(med) > 1e-12, np.abs(med), np.nan)
            rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            continue
        logging.info(
            "  %s : inter-method relative spread  median=%.2e  max=%.2e",
            obs_name, float(np.nanmedian(rel)), float(np.nanmax(rel)),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
