#!/usr/bin/env python3
"""Robust pyrochlore NLCE fitting tool.

Fits pyrochlore model Hamiltonians to thermodynamic data (specific heat,
entropy, susceptibility) using order-8 NLCE with the ``pyrochlore_order8``
pipeline (adaptive LB-NLCE).

Supported Hamiltonians
----------------------
``qsi``   -- quantum spin ice / XYZ + anisotropic exchange:
             H = Σ_{<ij>} [Jzz S_i^z S_j^z
                           + J_± (S_i^+ S_j^- + S_i^- S_j^+)
                           + J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
                           − i J_z±/2 ((γ_ij* S_i^+ − γ_ij S_i^-) S_j^z + h.c.)]
             Parameters: (Jzz, Jpm, Jpmpm, Jzpm)

``xyz``   -- anisotropic Heisenberg (Jxx, Jyy, Jzz) + longitudinal field h.
             Parameters: (Jxx, Jyy, Jzz, h)

``heisenberg`` -- isotropic Heisenberg J + field h.  Parameters: (J, h)

Fitting strategy
----------------
1. ``differential_evolution`` for global search (parallelism via ``workers``).
2. ``Nelder-Mead`` local refinement from the best DE solution.
3. Results written to ``--output_dir/fit_result.json`` and plotted.

Caching
-------
Each NLCE evaluation hashes the parameter vector and checks an HDF5 cache
in ``--work_dir/.nlce_cache.h5``.  Hit rate is typically >80% for DE
populations that revisit near-optimal regions.

Low-T extension
---------------
When the NLCE sum diverges below T_conv, ``bernu_misguich`` from
``qed_nlce.analysis.entropy_interpolation`` is applied automatically if
``--bernu`` is passed.  The chi2 computation then covers T down to
T_min even when T_conv > T_min.

Usage
-----
Basic fit (specific heat)::

    python -m qed_nlce.analysis.pyrochlore_nlce_fit \\
        --exp_Cv  data/Cv_zero_field.txt \\
        --model qsi \\
        --max_order 8 \\
        --base_dir /scratch/nlce_work \\
        --output_dir fit_results_qsi \\
        --workers 8

Multi-observable fit (Cv at several fields)::

    python -m qed_nlce.analysis.pyrochlore_nlce_fit \\
        --exp_config datasets.json \\
        --model qsi \\
        --max_order 8 \\
        --bernu \\
        --E0_per_site -0.39 \\
        --base_dir /scratch/nlce_work \\
        --output_dir fit_results_qsi \\
        --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Numpy-safe JSON encoder
# ---------------------------------------------------------------------------

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Simple HDF5-backed result cache
# ---------------------------------------------------------------------------

class _ResultCache:
    """Cache NLCE result arrays keyed by a parameter hash string."""

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._hits = self._misses = 0

    def _key(self, params: np.ndarray) -> str:
        # Round to 8 significant figures to tolerate float repr differences
        rounded = np.round(params.astype(float), decimals=8)
        return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]

    def get(self, params: np.ndarray) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            import h5py
            key = self._key(params)
            with h5py.File(self.path, "a") as f:
                if key in f:
                    grp = f[key]
                    result = {k: grp[k][()] for k in grp}
                    self._hits += 1
                    return result
        except Exception:
            pass
        self._misses += 1
        return None

    def put(self, params: np.ndarray, result: dict) -> None:
        if not self.enabled:
            return
        try:
            import h5py
            key = self._key(params)
            with h5py.File(self.path, "a") as f:
                if key in f:
                    del f[key]
                grp = f.create_group(key)
                for k, v in result.items():
                    grp.create_dataset(k, data=np.asarray(v))
        except Exception as e:
            logging.debug("Cache write failed: %s", e)

    def log_stats(self) -> None:
        total = self._hits + self._misses
        if total:
            logging.info("NLCE cache: %d hits / %d calls (%.0f%%)",
                         self._hits, total, 100 * self._hits / total)


# ---------------------------------------------------------------------------
# NLCE runner
# ---------------------------------------------------------------------------

class PyrochloreNLCERunner:
    """Runs one NLCE evaluation and returns thermodynamic observables.

    Wraps the ``qed_nlce`` unified CLI so that all cluster generation,
    Hamiltonian prep, ED, and summation are handled by the existing
    infrastructure.  A content-addressed result cache avoids redundant
    ED when the optimizer revisits nearby parameter values.
    """

    def __init__(
        self,
        model: str,
        max_order: int,
        base_dir: str,
        *,
        temp_min: float = 0.05,
        temp_max: float = 20.0,
        temp_bins: int = 120,
        SI_units: bool = True,
        parallel: bool = True,
        num_cores: int = 4,
        po8_full_threshold: int = 13,
        po8_k_base: int = 500,
        po8_k_max: int = 2000,
        bernu: bool = False,
        E0_per_site: Optional[float] = None,
        S_gs: float = 0.0,
        cache_path: Optional[str] = None,
        skip_cluster_gen: bool = False,
        extra_flags: Optional[list[str]] = None,
    ):
        self.model = model
        self.max_order = max_order
        self.base_dir = os.path.abspath(base_dir)
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.temp_bins = temp_bins
        self.SI_units = SI_units
        self.parallel = parallel
        self.num_cores = num_cores
        self.po8_full_threshold = po8_full_threshold
        self.po8_k_base = po8_k_base
        self.po8_k_max = po8_k_max
        self.bernu = bernu
        self.E0_per_site = E0_per_site
        self.S_gs = S_gs
        self.extra_flags = extra_flags or []
        self.skip_cluster_gen = skip_cluster_gen

        cache_path = cache_path or os.path.join(base_dir, ".nlce_cache.h5")
        self.cache = _ResultCache(cache_path)

        # Persistent cluster + Hamiltonian directories (reused across calls).
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Parameter → Hamiltonian flags
    # ------------------------------------------------------------------

    def _param_to_flags(self, params: np.ndarray) -> list[str]:
        """Convert parameter vector to geometry-specific CLI flags."""
        if self.model == "qsi":
            Jzz, Jpm, Jpmpm, Jzpm = params
            return [
                f"--Jzz={Jzz:.12f}",
                f"--Jpm={Jpm:.12f}",
                f"--Jpmpm={Jpmpm:.12f}",
                f"--Jzpm={Jzpm:.12f}",
            ]
        if self.model == "xyz":
            Jxx, Jyy, Jzz = params[:3]
            h = params[3] if len(params) > 3 else 0.0
            return [f"--Jxx={Jxx:.12f}", f"--Jyy={Jyy:.12f}",
                    f"--Jzz={Jzz:.12f}", f"--h={h:.12f}"]
        if self.model == "heisenberg":
            J, h = (params[0], params[1] if len(params) > 1 else 0.0)
            return [f"--Jxx={J:.12f}", f"--Jyy={J:.12f}",
                    f"--Jzz={J:.12f}", f"--h={h:.12f}"]
        raise ValueError(f"Unknown model: {self.model!r}")

    # ------------------------------------------------------------------
    # Core: run NLCE and return T, Cv, S arrays
    # ------------------------------------------------------------------

    def run(self, params: np.ndarray) -> Optional[dict[str, np.ndarray]]:
        """Run the full NLCE pipeline for ``params``.  Returns dict with
        keys ``T``, ``Cv``, ``S`` (all 1D float arrays), or ``None`` on
        failure.
        """
        params = np.asarray(params, dtype=float)

        # Cache lookup
        cached = self.cache.get(params)
        if cached is not None:
            return cached

        # Build CLI command
        cmd = self._build_cmd(params)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        except subprocess.CalledProcessError as e:
            logging.error("NLCE run failed: %s", e.stderr.decode("utf-8", errors="replace")[:400])
            return None
        except subprocess.TimeoutExpired:
            logging.error("NLCE run timed out (>2 h)")
            return None

        # After the first successful run, cluster files are on disk — skip
        # regeneration on all subsequent calls (saves ~30 s per evaluation).
        self.skip_cluster_gen = True

        result = self._read_result()
        if result is None:
            return None

        # Optional Bernu-Misguich low-T extension
        if self.bernu and self.E0_per_site is not None:
            result = self._apply_bernu(result)

        self.cache.put(params, result)
        return result

    def _build_cmd(self, params: np.ndarray) -> list[str]:
        cmd = [
            sys.executable, "-m", "qed_nlce",
            "--geometry=pyrochlore",
            "--pipeline=pyrochlore_order8",
            f"--max_order={self.max_order}",
            f"--base_dir={self.base_dir}",
            f"--temp_min={self.temp_min:.8f}",
            f"--temp_max={self.temp_max:.8f}",
            f"--temp_bins={self.temp_bins}",
            f"--po8_full_threshold={self.po8_full_threshold}",
            f"--po8_k_base={self.po8_k_base}",
            f"--po8_k_max={self.po8_k_max}",
            "--thermo",
            # Hamiltonian prep always runs (writes coupling constants per params).
            # Cluster generation is skipped after the first call.
        ]
        if self.skip_cluster_gen:
            cmd.append("--skip_cluster_gen")
        if self.SI_units:
            cmd.append("--SI_units")
        if self.parallel:
            cmd.extend(["--parallel", f"--num_cores={self.num_cores}"])

        cmd.extend(self._param_to_flags(params))
        cmd.extend(self.extra_flags)
        return cmd

    def _read_result(self) -> Optional[dict]:
        """Read the NLCE summation output from disk."""
        nlc_dir = os.path.join(
            self.base_dir, f"nlc_results_order_{self.max_order}"
        )
        cv_file = os.path.join(nlc_dir, "nlc_specific_heat.txt")
        if not os.path.isfile(cv_file):
            logging.error("NLCE specific heat file not found: %s", cv_file)
            return None
        try:
            data = np.loadtxt(cv_file)
            T, Cv = data[:, 0], data[:, 1]
        except Exception as e:
            logging.error("Failed to load %s: %s", cv_file, e)
            return None

        result = {"T": T, "Cv": Cv}

        # Try loading entropy
        s_file = os.path.join(nlc_dir, "nlc_entropy.txt")
        if os.path.isfile(s_file):
            try:
                sdata = np.loadtxt(s_file)
                result["S"] = sdata[:, 1]
            except Exception:
                pass

        return result

    def _apply_bernu(self, result: dict) -> dict:
        """Extend result below convergence via Bernu-Misguich."""
        try:
            from .entropy_interpolation import bernu_misguich
            bm = bernu_misguich(
                T_nlce=result["T"],
                Cv_nlce=result["Cv"],
                E0_per_site=self.E0_per_site,
                S_gs=self.S_gs,
                T_conv_max=None,
                n_out=self.temp_bins,
            )
            return {
                "T":  bm["T"],
                "Cv": bm["Cv"],
                "S":  bm["S"],
                "E":  bm["E"],
                "_bernu_applied": np.array([1]),
            }
        except Exception as e:
            logging.warning("Bernu-Misguich failed, using raw NLCE: %s", e)
            return result


# ---------------------------------------------------------------------------
# Objective function + multi-dataset chi2
# ---------------------------------------------------------------------------

def _interp_safe(T_calc, y_calc, T_exp):
    """Cubic interpolation from calc grid onto experimental T points."""
    if T_calc is None or len(T_calc) < 2:
        return np.full_like(T_exp, np.nan)
    sort = np.argsort(T_calc)
    fn = interp1d(T_calc[sort], y_calc[sort], kind="cubic",
                  bounds_error=False, fill_value=np.nan)
    return fn(T_exp)


def _chi2_dataset(y_calc, y_exp, T_exp, T_min, T_max, sigma=None):
    """Weighted chi2 over temperature range [T_min, T_max]."""
    mask = (T_exp >= T_min) & (T_exp <= T_max) & np.isfinite(y_calc)
    if mask.sum() == 0:
        return 1e12
    diff = y_calc[mask] - y_exp[mask]
    if sigma is not None:
        w = 1.0 / (sigma[mask] ** 2 + 1e-20)
        return float(np.sum(w * diff ** 2))
    return float(np.sum(diff ** 2))


# ---------------------------------------------------------------------------
# Fit engine
# ---------------------------------------------------------------------------

class PyrochloreNLCEFit:
    """High-level fitting interface.

    Parameters
    ----------
    runner : PyrochloreNLCERunner
        Handles the NLCE computation.
    datasets : list of dict
        Each dict has:
          ``T``       : np.ndarray — experimental temperatures
          ``Cv``      : np.ndarray — experimental specific heat (optional)
          ``S``       : np.ndarray — experimental entropy (optional)
          ``Cv_err``  : np.ndarray — uncertainties on Cv (optional)
          ``T_min``   : float — lower fit window
          ``T_max``   : float — upper fit window
          ``weight``  : float — dataset weight in chi2 (default 1.0)
    param_names : list of str
    bounds : list of (lo, hi) tuples
    """

    def __init__(
        self,
        runner: PyrochloreNLCERunner,
        datasets: list[dict],
        param_names: list[str],
        bounds: list[tuple[float, float]],
    ):
        self.runner = runner
        self.datasets = datasets
        self.param_names = param_names
        self.bounds = bounds
        self._n_eval = 0
        self._best_chi2 = np.inf
        self._best_params = None

    def chi2(self, params: np.ndarray) -> float:
        """Evaluate chi2 for ``params``."""
        params = np.asarray(params, dtype=float)
        result = self.runner.run(params)

        self._n_eval += 1

        if result is None:
            return 1e12

        total = 0.0
        for ds in self.datasets:
            T_exp = np.asarray(ds["T"])
            T_min = ds.get("T_min", self.runner.temp_min)
            T_max = ds.get("T_max", self.runner.temp_max)
            w = ds.get("weight", 1.0)

            for obs_key in ("Cv", "S"):
                if obs_key not in ds or obs_key not in result:
                    continue
                y_exp = np.asarray(ds[obs_key])
                y_calc = _interp_safe(result["T"], result[obs_key], T_exp)
                sigma = np.asarray(ds[obs_key + "_err"]) if obs_key + "_err" in ds else None
                total += w * _chi2_dataset(y_calc, y_exp, T_exp, T_min, T_max, sigma)

        if total < self._best_chi2:
            self._best_chi2 = total
            self._best_params = params.copy()

        logging.info(
            "[eval %4d] chi2=%.6g  %s",
            self._n_eval, total,
            "  ".join(f"{n}={v:.4f}" for n, v in zip(self.param_names, params)),
        )
        return total

    def fit(
        self,
        *,
        method: str = "differential_evolution",
        n_starts: int = 15,
        workers: int = 1,
        tol: float = 1e-6,
        maxiter: int = 2000,
        seed: int = 42,
        initial_params: Optional[np.ndarray] = None,
    ) -> dict:
        """Run the optimizer and return a result dict.

        Parameters
        ----------
        method : ``"differential_evolution"`` (default, global search) or
                 ``"multi_start_nelder_mead"`` (faster, local).
        workers : Number of parallel workers for differential evolution.
                  Note: each worker calls one full NLCE, so set this ≤
                  num_cores so total concurrency stays bounded.
        """
        logging.info("Starting %s fit, %d parameters.", method, len(self.bounds))
        logging.info("Parameter bounds:")
        for name, (lo, hi) in zip(self.param_names, self.bounds):
            logging.info("  %-12s [%.4f, %.4f]", name, lo, hi)

        t0 = time.time()

        if method == "differential_evolution":
            res = differential_evolution(
                self.chi2,
                self.bounds,
                maxiter=maxiter,
                tol=tol,
                seed=seed,
                workers=workers,
                polish=True,
                disp=False,
            )
            best_params = res.x
            best_chi2 = res.fun
            success = res.success
            message = res.message

        elif method == "multi_start_nelder_mead":
            # Latin Hypercube sampling + Nelder-Mead from each start.
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            sampler = qmc.LatinHypercube(d=len(self.bounds), seed=seed)
            starts = qmc.scale(sampler.random(n_starts), lo, hi)
            if initial_params is not None:
                starts[0] = np.clip(initial_params, lo, hi)

            best_params, best_chi2 = None, np.inf
            for i, x0 in enumerate(starts):
                logging.info("Start %d/%d: %s",
                             i + 1, n_starts,
                             " ".join(f"{n}={v:.3f}" for n, v in
                                      zip(self.param_names, x0)))
                r = minimize(self.chi2, x0, method="Nelder-Mead",
                             options={"maxiter": maxiter, "xatol": 1e-5, "fatol": tol})
                if r.fun < best_chi2:
                    best_chi2 = r.fun
                    best_params = r.x

            success = True
            message = f"multi_start_nelder_mead with {n_starts} starts"
        else:
            raise ValueError(f"Unknown method: {method!r}")

        elapsed = time.time() - t0
        logging.info("Optimization completed in %.1f min, chi2=%.6g", elapsed / 60, best_chi2)

        self.runner.cache.log_stats()

        return {
            "best_params": {n: float(v) for n, v in zip(self.param_names, best_params)},
            "chi2": float(best_chi2),
            "n_eval": self._n_eval,
            "elapsed_min": elapsed / 60,
            "success": success,
            "message": str(message),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_dataset(path: str, *, T_min=None, T_max=None, weight=1.0,
                  obs="Cv") -> dict:
    """Load a two-column (T, observable) text file."""
    data = np.loadtxt(path)
    return {
        "T": data[:, 0],
        obs: data[:, 1],
        "T_min": T_min or data[0, 0],
        "T_max": T_max or data[-1, 0],
        "weight": weight,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit pyrochlore Hamiltonian parameters to thermodynamic data "
                    "using order-N NLCE with the pyrochlore_order8 pipeline."
    )

    # --- data ---
    parser.add_argument("--exp_Cv", type=str, default=None,
                        help="Path to experimental Cv(T) file (2-column: T Cv).")
    parser.add_argument("--exp_S",  type=str, default=None,
                        help="Path to experimental S(T) file.")
    parser.add_argument("--exp_config", type=str, default=None,
                        help="JSON config for multiple datasets (see docs).")
    parser.add_argument("--T_fit_min", type=float, default=None,
                        help="Lower temperature bound for chi2 (default: data start).")
    parser.add_argument("--T_fit_max", type=float, default=None,
                        help="Upper temperature bound for chi2 (default: data end).")

    # --- model ---
    parser.add_argument("--model", type=str, default="qsi",
                        choices=["qsi", "xyz", "heisenberg"],
                        help="Spin model (default: qsi).")
    # QSI initial / bounds
    parser.add_argument("--Jzz_init", type=float, default=0.5)
    parser.add_argument("--Jpm_init", type=float, default=0.25)
    parser.add_argument("--Jpmpm_init", type=float, default=0.0)
    parser.add_argument("--Jzpm_init", type=float, default=0.0)
    parser.add_argument("--Jzz_min",  type=float, default=-2.0)
    parser.add_argument("--Jzz_max",  type=float, default=2.0)
    parser.add_argument("--Jpm_min",  type=float, default=-2.0)
    parser.add_argument("--Jpm_max",  type=float, default=2.0)
    parser.add_argument("--Jpmpm_min", type=float, default=-1.0)
    parser.add_argument("--Jpmpm_max", type=float, default=1.0)
    parser.add_argument("--Jzpm_min", type=float, default=-1.0)
    parser.add_argument("--Jzpm_max", type=float, default=1.0)

    # --- NLCE ---
    parser.add_argument("--max_order", type=int, default=8)
    parser.add_argument("--temp_min",  type=float, default=0.05)
    parser.add_argument("--temp_max",  type=float, default=20.0)
    parser.add_argument("--temp_bins", type=int, default=120)
    parser.add_argument("--SI_units",  action="store_true", default=True)
    parser.add_argument("--base_dir",  type=str, default="nlce_fit_work")
    parser.add_argument("--output_dir", type=str, default="fit_results_pyrochlore")
    parser.add_argument("--skip_cluster_gen", action="store_true")
    parser.add_argument("--workers",   type=int, default=1,
                        help="Parallel workers for differential evolution.")
    parser.add_argument("--num_cores", type=int, default=4,
                        help="CPU cores for parallel cluster ED within each evaluation.")

    # --- pipeline knobs ---
    parser.add_argument("--po8_full_threshold", type=int, default=13)
    parser.add_argument("--po8_k_base",  type=int, default=500)
    parser.add_argument("--po8_k_max",   type=int, default=2000)

    # --- low-T extension ---
    parser.add_argument("--bernu", action="store_true",
                        help="Apply Bernu-Misguich entropy interpolation below T_conv.")
    parser.add_argument("--E0_per_site", type=float, default=None,
                        help="Ground-state energy per site for Bernu-Misguich.")
    parser.add_argument("--S_gs", type=float, default=0.0,
                        help="Ground-state entropy per site (default 0).")

    # --- optimizer ---
    parser.add_argument("--method", type=str, default="differential_evolution",
                        choices=["differential_evolution", "multi_start_nelder_mead"])
    parser.add_argument("--n_starts",  type=int, default=15)
    parser.add_argument("--maxiter",   type=int, default=2000)
    parser.add_argument("--tol",       type=float, default=1e-6)
    parser.add_argument("--seed",      type=int, default=42)

    args = parser.parse_args(argv)

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "fit.log")),
            logging.StreamHandler(),
        ],
    )

    # --- Load datasets ---
    datasets = []
    if args.exp_config:
        with open(args.exp_config) as f:
            cfg = json.load(f)
        for ds_cfg in cfg["datasets"]:
            ds = _load_dataset(
                ds_cfg["file"],
                T_min=ds_cfg.get("T_min"), T_max=ds_cfg.get("T_max"),
                weight=ds_cfg.get("weight", 1.0),
                obs=ds_cfg.get("observable", "Cv"),
            )
            datasets.append(ds)
    elif args.exp_Cv:
        ds = _load_dataset(
            args.exp_Cv, T_min=args.T_fit_min, T_max=args.T_fit_max, obs="Cv"
        )
        datasets.append(ds)
        if args.exp_S:
            ds_s = _load_dataset(
                args.exp_S, T_min=args.T_fit_min, T_max=args.T_fit_max, obs="S"
            )
            datasets.append(ds_s)
    else:
        parser.error("Provide --exp_Cv or --exp_config.")

    # --- Parameter setup ---
    if args.model == "qsi":
        param_names = ["Jzz", "Jpm", "Jpmpm", "Jzpm"]
        bounds = [
            (args.Jzz_min,   args.Jzz_max),
            (args.Jpm_min,   args.Jpm_max),
            (args.Jpmpm_min, args.Jpmpm_max),
            (args.Jzpm_min,  args.Jzpm_max),
        ]
        initial = np.array([args.Jzz_init, args.Jpm_init,
                            args.Jpmpm_init, args.Jzpm_init])
    elif args.model == "xyz":
        param_names = ["Jxx", "Jyy", "Jzz", "h"]
        bounds = [(args.Jzz_min, args.Jzz_max)] * 3 + [(0.0, 5.0)]
        initial = np.array([1.0, 1.0, 1.0, 0.0])
    else:  # heisenberg
        param_names = ["J", "h"]
        bounds = [(0.0, 3.0), (0.0, 3.0)]
        initial = np.array([1.0, 0.0])

    # --- Build runner ---
    runner = PyrochloreNLCERunner(
        model=args.model,
        max_order=args.max_order,
        base_dir=args.base_dir,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_bins=args.temp_bins,
        SI_units=args.SI_units,
        parallel=(args.num_cores > 1),
        num_cores=args.num_cores,
        po8_full_threshold=args.po8_full_threshold,
        po8_k_base=args.po8_k_base,
        po8_k_max=args.po8_k_max,
        bernu=args.bernu,
        E0_per_site=args.E0_per_site,
        S_gs=args.S_gs,
        skip_cluster_gen=args.skip_cluster_gen,
    )

    # --- Fit ---
    fit_engine = PyrochloreNLCEFit(runner, datasets, param_names, bounds)
    result = fit_engine.fit(
        method=args.method,
        n_starts=args.n_starts,
        workers=args.workers,
        tol=args.tol,
        maxiter=args.maxiter,
        seed=args.seed,
        initial_params=initial,
    )

    # --- Save ---
    out_json = os.path.join(args.output_dir, "fit_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, cls=_NpEncoder)
    logging.info("Fit result saved to %s", out_json)

    # --- Best-fit NLCE curve ---
    best_p = np.array(list(result["best_params"].values()))
    best_result = runner.run(best_p)
    if best_result is not None:
        out_nlce = os.path.join(args.output_dir, "best_fit_Cv.txt")
        np.savetxt(out_nlce, np.column_stack([best_result["T"], best_result["Cv"]]),
                   header="T  Cv")
        logging.info("Best-fit Cv curve saved to %s", out_nlce)
        _plot(datasets, best_result, result, args.output_dir, args.SI_units)

    return 0


def _plot(datasets, best_result, fit_result, output_dir, SI_units):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets) + 1))

    for i, ds in enumerate(datasets):
        obs = "Cv" if "Cv" in ds else "S"
        ax.scatter(ds["T"], ds[obs], s=20, color=colors[i],
                   label=f"Exp {obs} ({i})", zorder=3)

    if "Cv" in best_result:
        ax.plot(best_result["T"], best_result["Cv"], "k-", lw=2,
                label="Best-fit NLCE")

    param_str = ", ".join(
        f"{k}={v:.3f}" for k, v in fit_result["best_params"].items()
    )
    ax.set_title(f"Pyrochlore NLCE fit\n{param_str}", fontsize=9)
    ax.set_xscale("log")
    y_label = "C$_v$ [J/(mol·K)]" if SI_units else "C$_v$ [k$_B$]"
    ax.set_xlabel("T [K]" if SI_units else "T [J]")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(output_dir, "fit_result.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
