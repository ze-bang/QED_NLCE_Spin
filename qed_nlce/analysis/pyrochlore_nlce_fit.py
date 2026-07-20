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

        # Per-process working root + cache. Parallel DE workers are separate
        # processes; each gets ``base_dir/w<pid>`` so their cluster/ED/result
        # trees and HDF5 caches never clobber one another. Set lazily on first
        # ``run()`` in each process (see ``_worker_dir``).
        self._active_base = self.base_dir
        self._wd_pid = None

        # Persistent cluster + Hamiltonian directories (reused across calls).
        os.makedirs(self.base_dir, exist_ok=True)

    def _worker_dir(self) -> str:
        """Return this process's isolated working directory (``base_dir/w<pid>``),
        (re)binding the result cache to it the first time each process calls in."""
        pid = os.getpid()
        if self._wd_pid != pid:
            wd = os.path.join(self.base_dir, f"w{pid}")
            os.makedirs(wd, exist_ok=True)
            self._active_base = wd
            self.cache = _ResultCache(os.path.join(wd, ".nlce_cache.h5"))
            self._wd_pid = pid
            self.skip_cluster_gen = False   # each worker generates clusters once
        return self._active_base

    # ------------------------------------------------------------------
    # Parameter → Hamiltonian flags
    # ------------------------------------------------------------------

    def _param_to_flags(self, params: np.ndarray,
                        h_override: Optional[float] = None) -> list[str]:
        """Convert parameter vector to geometry-specific CLI flags.

        ``h_override`` (when not None) supplies the Zeeman field for a
        per-field (multi-field) evaluation; the field is then NOT a fit
        parameter. The pyrochlore geometry accepts Jxx/Jyy/Jzz/h only, so the
        non-Kramers QSI model is fit in the XYZ frame (Jxx = -J± + J±±,
        Jyy = -J± - J±±, Jzz = Jzz; Jz± = 0 by time reversal).
        """
        if self.model == "qsi":
            # Local-frame QSI mapped to the XYZ couplings the ED code consumes.
            # A 2-parameter vector means J±± is FIXED at exactly 0 (--fix_Jpmpm):
            # Jxx == Jyy then conserves total S^z (the [111] Zeeman is
            # longitudinal in the local frames), so 16-site clusters block into
            # <=12870-dim sectors instead of a 65536-dim Sz-broken dense solve
            # (~1.3 GB/40 s vs ~34 GB/1 h) -- the difference between a
            # DE-viable order-5 fit and an OOM.
            Jzz, Jpm = params[0], params[1]
            Jpmpm = params[2] if len(params) >= 3 else 0.0
            Jxx, Jyy = -Jpm + Jpmpm, -Jpm - Jpmpm
            h = h_override if h_override is not None else 0.0
            return [f"--Jxx={Jxx:.12f}", f"--Jyy={Jyy:.12f}",
                    f"--Jzz={Jzz:.12f}", f"--h={h:.12f}"]
        if self.model == "xyz":
            Jxx, Jyy, Jzz = params[:3]
            h = (h_override if h_override is not None
                 else (params[3] if len(params) > 3 else 0.0))
            return [f"--Jxx={Jxx:.12f}", f"--Jyy={Jyy:.12f}",
                    f"--Jzz={Jzz:.12f}", f"--h={h:.12f}"]
        if self.model == "heisenberg":
            J = params[0]
            h = (h_override if h_override is not None
                 else (params[1] if len(params) > 1 else 0.0))
            return [f"--Jxx={J:.12f}", f"--Jyy={J:.12f}",
                    f"--Jzz={J:.12f}", f"--h={h:.12f}"]
        raise ValueError(f"Unknown model: {self.model!r}")

    # ------------------------------------------------------------------
    # Core: run NLCE and return T, Cv, S arrays
    # ------------------------------------------------------------------

    def run(self, params: np.ndarray,
            h: Optional[float] = None,
            field_dir: Optional[tuple] = None) -> Optional[dict[str, np.ndarray]]:
        """Run the full NLCE pipeline for ``params`` at Zeeman field ``h``
        (meV) along ``field_dir``. Returns dict with keys ``T``, ``Cv``, ``S``
        (all 1D float arrays), or ``None`` on failure.
        """
        params = np.asarray(params, dtype=float)
        self._worker_dir()   # bind this process's isolated base_dir + cache

        # Cache key folds in the field so multi-field evaluations don't collide.
        if h is None:
            cache_key = params
        else:
            fd = np.asarray(field_dir if field_dir is not None else (0.0, 0.0, 0.0),
                            dtype=float)
            cache_key = np.concatenate([params, [float(h)], fd])

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Build CLI command
        cmd = self._build_cmd(params, h, field_dir)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        except subprocess.CalledProcessError as e:
            # The workflow logs INFO to stderr, so the interesting part (the
            # traceback / abort reason) is at the END of the stream.
            err = e.stderr.decode("utf-8", errors="replace")
            logging.error("NLCE run failed (rc=%s): ...%s", e.returncode, err[-1500:])
            return None
        except subprocess.TimeoutExpired:
            logging.error("NLCE run timed out (>2 h): %s", " ".join(cmd[:8]))
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

        self.cache.put(cache_key, result)
        return result

    def _build_cmd(self, params: np.ndarray,
                   h: Optional[float] = None,
                   field_dir: Optional[tuple] = None) -> list[str]:
        # Pipeline: full_ed (the in-process symmetry-adapted solver). Large
        # clusters (Hilbert dim > EDOptions.oftlm_cutoff) auto-route to OFTLM
        # inside dense_ed.run_ed_in_process -- no extra flag needed.
        cmd = [
            sys.executable, "-m", "qed_nlce",
            "--geometry=pyrochlore",
            "--pipeline=full_ed",
            f"--max_order={self.max_order}",
            f"--base_dir={self._active_base}",
            f"--temp_min={self.temp_min:.8f}",
            f"--temp_max={self.temp_max:.8f}",
            f"--temp_bins={self.temp_bins}",
            "--thermo",
        ]
        if self.skip_cluster_gen:
            cmd.append("--skip_cluster_gen")
        if self.SI_units:
            cmd.append("--SI_units")
        if self.parallel:
            cmd.extend(["--parallel", f"--num_cores={self.num_cores}"])

        cmd.extend(self._param_to_flags(params, h))
        if field_dir is not None:
            cmd.extend(["--field_dir",
                        f"{float(field_dir[0]):.10f}",
                        f"{float(field_dir[1]):.10f}",
                        f"{float(field_dir[2]):.10f}"])
        cmd.extend(self.extra_flags)
        return cmd

    def _read_result(self) -> Optional[dict]:
        """Read the NLCE summation output from disk."""
        nlc_dir = os.path.join(
            self._active_base, f"nlc_results_order_{self.max_order}"
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
    """Linear interpolation from calc grid onto experimental T points.

    Linear, not cubic: gapped in-field C(T) is ~0 below the Zeeman gap and
    then rises steeply; a cubic spline RINGS negative between grid points
    on that shape (undershoot ~0.1 J/mol/K), which both distorts chi2 and
    false-triggers the negative-C divergence rejector. The model grid is a
    dense 40-point log grid -- linear error is negligible against it.
    """
    if T_calc is None or len(T_calc) < 2:
        return np.full_like(T_exp, np.nan)
    sort = np.argsort(T_calc)
    fn = interp1d(T_calc[sort], y_calc[sort], kind="linear",
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
        """Evaluate chi2 for ``params``, summing over datasets. Each dataset may
        carry its own Zeeman field (``h_meV`` / ``field_dir``); the NLCE is run
        once per distinct field (the runner caches on params+field)."""
        params = np.asarray(params, dtype=float)
        self._n_eval += 1

        # Floating g-factor: when "g_eff" is a fit parameter, the Zeeman
        # field of each dataset is recomputed per evaluation from its stored
        # B (Tesla): h[meV] = B * g_eff * MU_B_MEV. The model couplings passed to
        # the NLCE runner exclude g_eff (it only enters through h).
        g_idx = self.param_names.index("g_eff") if "g_eff" in self.param_names else None
        p_model = params if g_idx is None else np.delete(params, g_idx)

        total = 0.0
        for ds in self.datasets:
            # Per-field evaluation (multi-field fit); h=None -> zero/single field.
            h = ds.get("h_meV")
            if g_idx is not None and ds.get("B_tesla") is not None:
                h = float(ds["B_tesla"]) * float(params[g_idx]) * MU_B_MEV
            result = self.runner.run(
                p_model, h=h, field_dir=ds.get("field_dir"))
            if result is None:
                return 1e12

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
                # NLCE divergence rejector (two-sided physical bound on Cv).
                # Below the series' convergence radius (T <~ J, or large field)
                # the bare partial sums cancel catastrophically and the model
                # C(T) diverges to O(10)-O(1000) J/mol/K, EITHER sign -- and a
                # cancellation-artifact "fit" of such curves scores better than
                # a physical one, poisoning the DE (this is exactly what railed
                # the order-5 campaigns toward unphysical large Jzz). The true
                # magnetic C of a doublet is bounded: the two-level Schottky
                # peak is ~3.6 J/(mol-Pr K), and exchange only broadens/lowers
                # it. So any in-window C outside [-0.05, CV_CEILING] marks the
                # point as non-converged and it is rejected outright. CV_CEILING
                # = 8 sits far above every physical feature (<~4) and far below
                # the divergence onset (>~20), so no converged point is lost.
                if obs_key == "Cv":
                    in_win = (T_exp >= T_min) & (T_exp <= T_max) & np.isfinite(y_calc)
                    if np.any(y_calc[in_win] < -0.05) or np.any(y_calc[in_win] > CV_CEILING):
                        total = 1e10
                        break
                total += w * _chi2_dataset(y_calc, y_exp, T_exp, T_min, T_max, sigma)
            if total >= 1e10:
                break

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
        popsize: int = 15,
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
            # Build the initial population so the warm-start point is
            # guaranteed to be a member.  scipy DE normalises init to [0,1]
            # per-dimension; we undo that here then redo it.
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            n_dim  = len(self.bounds)
            n_pop  = popsize * n_dim
            rng    = np.random.default_rng(seed)
            samp   = qmc.LatinHypercube(d=n_dim, seed=rng)
            # unit-cube population (what scipy expects for `init`)
            init_pop = samp.random(n_pop)
            if initial_params is not None:
                # Replace first member with clipped warm start
                init_pop[0] = (np.clip(initial_params, lo, hi) - lo) / (hi - lo)
            res = differential_evolution(
                self.chi2,
                self.bounds,
                maxiter=maxiter,
                tol=tol,
                seed=seed,
                workers=workers,
                popsize=popsize,
                init=init_pop,
                polish=False,
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

# Zeeman conversion: pyrochlore couplings (Jxx/Jyy/Jzz) are in meV; h must
# therefore also be in meV so all Hamiltonian energies share the same unit.
# NLC_sum then applies the 1/k_B = 11.6045 K/meV scale factor via --energy_unit=meV.
#   h[meV] = B[T] * g_eff * mu_B[meV/T] = B * g_eff * 0.05788 meV/T
MU_B_MEV = 0.05788             # mu_B in meV/T (CODATA: 9.274e-24 J/T / 1.602e-22 J/meV)
G_EFF  = 5.4                  # effective g along local [111] (Kimura 2013 / Sibille 2018)
# In-window magnetic-Cv ceiling (J/mol-Pr/K) for the NLCE divergence rejector.
# Physical doublet features are <~4; NLCE non-convergence diverges to >~20.
CV_CEILING = 8.0


def _load_dataset(path: str, *, T_min=None, T_max=None, weight=1.0,
                  obs="Cv", h_meV=None, field_dir=None) -> dict:
    """Load a two-column (T, observable) text file (optionally field-tagged)."""
    data = np.loadtxt(path)
    ds = {
        "T": data[:, 0],
        obs: data[:, 1],
        "T_min": T_min if T_min is not None else data[0, 0],
        "T_max": T_max if T_max is not None else data[-1, 0],
        "weight": weight,
    }
    if h_meV is not None:
        ds["h_meV"] = float(h_meV)
    if field_dir is not None:
        ds["field_dir"] = tuple(float(x) for x in field_dir)
    return ds


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
    parser.add_argument("--fix_Jpmpm", action="store_true",
                        help="qsi model: pin J±± at exactly 0 (2 exchange "
                             "params). Keeps Jxx == Jyy, so total S^z stays "
                             "conserved and 16-site clusters stay Sz-blocked "
                             "(~40 s dense solves instead of ~1 h, 34 GB "
                             "Sz-broken ones) -- required for order-5 DE fits.")
    parser.add_argument("--float_g", action="store_true",
                        help="Fit the effective g-factor as an extra parameter "
                             "(multi-field configs only; initialized at --g_eff, "
                             "bounded by --g_min/--g_max).")
    parser.add_argument("--g_min", type=float, default=3.0,
                        help="Lower bound for a floating g_eff (default 3.0).")
    parser.add_argument("--g_max", type=float, default=7.0,
                        help="Upper bound for a floating g_eff (default 7.0).")
    parser.add_argument("--g_eff", type=float, default=G_EFF,
                        help="Effective g along local [111] for the Zeeman "
                             "conversion h[meV]=B[T]*g_eff*mu_B (default %(default)s).")
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
    parser.add_argument("--resummation", type=str, default="none",
                        choices=["auto", "direct", "none", "euler", "wynn",
                                 "shanks", "aitken", "pade", "theta",
                                 "robust", "wynn_multi", "brezinski"],
                        help="NLCE series-acceleration method forwarded to the "
                             "summation kernel. Default 'none' (BARE partial "
                             "sum). With all-exact clusters (order<=6 under "
                             "--fix_Jpmpm, no OFTLM), the bare series is already "
                             "converged in the physical regime -- verified at "
                             "order 6, moderate Jzz: S4=1.713, S5=1.714, "
                             "S6=1.747 -- so bare IS the truth and needs no "
                             "acceleration. Crucially it is DETERMINISTIC: "
                             "'auto' picks a method per-property from "
                             "convergence heuristics and, near the divergence "
                             "radius, the fit worker and a rerun can pick "
                             "DIFFERENT methods (2 vs 22 vs 850 J/mol/K), making "
                             "chi2 a lottery that railed the order-5 campaigns. "
                             "Where bare diverges (large Jzz / large field, "
                             "below the convergence radius) the two-sided "
                             "CV_CEILING rejector marks the point non-converged.")
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
    parser.add_argument("--popsize",   type=int, default=15,
                        help="DE population multiplier (population = popsize * "
                             "n_params). Lower it (e.g. 6) for expensive "
                             "order-6 evaluations.")
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
        # Accept both schemas: single-field ("datasets") and multi-field
        # ("experimental_data", each with a Tesla "h" + "field_dir").
        ds_list = cfg.get("datasets") or cfg.get("experimental_data") or []
        for ds_cfg in ds_list:
            B = ds_cfg.get("h")   # field in Tesla for a multi-field dataset
            # Field in meV (pyrochlore couplings are in meV; NLC_sum converts to K).
            h_meV = (float(B) * args.g_eff * MU_B_MEV) if B is not None else None
            ds = _load_dataset(
                ds_cfg["file"],
                T_min=ds_cfg.get("T_min", ds_cfg.get("temp_min")),
                T_max=ds_cfg.get("T_max", ds_cfg.get("temp_max")),
                weight=ds_cfg.get("weight", 1.0),
                obs=ds_cfg.get("observable", "Cv"),
                h_meV=h_meV,
                field_dir=ds_cfg.get("field_dir"),
            )
            if B is not None:
                ds["B_tesla"] = float(B)   # kept for a floating g_eff
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
    # Multi-field fits derive the Zeeman h per dataset (h_meV), so h is NOT a
    # fit parameter; single/zero-field fits keep h in the vector.
    multi_field = any("h_meV" in ds for ds in datasets)

    def _jb(lo, hi):  # coupling bound helper (fall back to the Jzz window)
        return (getattr(args, lo, args.Jzz_min), getattr(args, hi, args.Jzz_max))

    if args.model == "qsi":
        # Non-Kramers: Jz± = 0 by time reversal -> 3 exchange params
        # (Jzz, J±, J±±), mapped to the ED code's Jxx/Jyy/Jzz in _param_to_flags.
        if args.fix_Jpmpm:
            # J±± pinned at exactly 0 -> Jxx == Jyy -> U(1) S^z survives
            # (also in a [111] field) and large clusters stay Sz-blocked.
            param_names = ["Jzz", "Jpm"]
            bounds  = [(args.Jzz_min, args.Jzz_max), _jb("Jpm_min", "Jpm_max")]
            initial = np.array([args.Jzz_init, args.Jpm_init])
        else:
            param_names = ["Jzz", "Jpm", "Jpmpm"]
            bounds  = [(args.Jzz_min, args.Jzz_max), _jb("Jpm_min", "Jpm_max"),
                       _jb("Jpmpm_min", "Jpmpm_max")]
            initial = np.array([args.Jzz_init, args.Jpm_init, args.Jpmpm_init])
    elif args.model == "xyz":
        param_names = ["Jxx", "Jyy", "Jzz"]
        bounds  = [(args.Jzz_min, args.Jzz_max)] * 3
        initial = np.array([1.0, 1.0, 1.0])
    else:  # heisenberg
        param_names = ["J"]
        bounds  = [(0.0, 3.0)]
        initial = np.array([1.0])

    if not multi_field and args.model in ("xyz", "heisenberg"):
        # Fit the field too when it is not fixed per dataset.
        param_names = param_names + ["h"]
        bounds = bounds + [(0.0, 5.0)]
        initial = np.append(initial, 0.0)

    if args.float_g:
        if not multi_field:
            parser.error("--float_g requires a multi-field --exp_config "
                         "(datasets with Tesla 'h' fields).")
        # g-factor calibration: the Zeeman scale of every in-field dataset is
        # recomputed per evaluation as h = B * g_eff * MU_B_MEV (see chi2).
        param_names = param_names + ["g_eff"]
        bounds = bounds + [(args.g_min, args.g_max)]
        initial = np.append(initial, args.g_eff)

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
        extra_flags=["--resummation", args.resummation],
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
        popsize=args.popsize,
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
