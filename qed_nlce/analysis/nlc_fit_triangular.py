#!/usr/bin/env python3
"""
NLCE Fitting Tool for Triangular Lattice

This script fits Numerical Linked Cluster Expansion (NLCE) calculations
for triangular lattice models to experimental or target specific heat data.

Supports:
- J1-J2 Heisenberg model
- XXZ model with anisotropy
- Kitaev-Heisenberg model
- Anisotropic exchange model (YbMgGaO4-type) with bond-dependent phases:
  H = Σ_{⟨ij⟩} [J_zz S_i^z S_j^z + J_± (S_i^+ S_j^- + S_i^- S_j^+)
               + J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
               - i J_z±/2 ((γ_ij* S_i^+ - γ_ij S_i^-) S_j^z + h.c.)]
  where γ_ij = 1, e^{i2π/3}, e^{-i2π/3} for bonds along a1, a2, a3 directions.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import logging
import json
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.stats import qmc
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.callbacks import CheckpointSaver
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
import time


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CostLandscapeLogger:
    """Accumulates every (params, chi²) evaluation and DE population state for
    cost-function landscape analysis and optimizer checkpointing.

    Usage:
        logger = CostLandscapeLogger(param_names, output_dir)
        # inside objective: logger.log_evaluation(params, chi2)
        # as DE callback:  callback=logger.de_callback
        # at end:          logger.save()
    """

    def __init__(self, param_names, output_dir, flush_interval=100):
        self.param_names = list(param_names)
        self.output_dir = output_dir
        self.flush_interval = flush_interval
        self._evals = []          # list of [*params, chi2]
        self._gen_best = []       # list of [gen, chi2_best, *xk]
        self._start_time = time.time()
        self._timestamps = []     # wall-clock seconds since start per eval
        self._generation = 0

    # ---- called every objective evaluation ----
    def log_evaluation(self, params, chi2):
        self._evals.append(list(params) + [float(chi2)])
        self._timestamps.append(time.time() - self._start_time)
        if len(self._evals) % self.flush_interval == 0:
            self._flush()

    # ---- DE callback (called once per generation) ----
    def de_callback(self, xk, convergence=0.0):
        self._generation += 1
        best_chi2 = float(self._evals[-1][-1]) if self._evals else np.nan
        # xk is current best; find its chi2 from recent evals
        for row in reversed(self._evals):
            if np.allclose(row[:-1], xk, atol=1e-12):
                best_chi2 = row[-1]
                break
        self._gen_best.append([self._generation, best_chi2, convergence]
                              + list(xk))
        logging.info(f"DE generation {self._generation}: best χ²={best_chi2:.6f}, "
                     f"convergence={convergence:.6e}")
        self._flush()  # save after every generation

    # ---- Bayesian optimization callback (called after each evaluation) ----
    def bo_callback(self, opt_result):
        """scikit-optimize callback; opt_result is a partial OptimizeResult."""
        n = len(opt_result.func_vals)
        best_idx = int(np.argmin(opt_result.func_vals))
        best_chi2 = opt_result.func_vals[best_idx]
        best_x = opt_result.x_iters[best_idx]
        self._generation = n
        self._gen_best.append([n, best_chi2, 0.0] + list(best_x))
        logging.info(f"BO iteration {n}: current χ²={opt_result.func_vals[-1]:.6f}, "
                     f"best χ²={best_chi2:.6f}")
        self._flush()

    # ---- periodic flush ----
    def _flush(self):
        self._save_npz(os.path.join(self.output_dir, 'cost_landscape.npz'))

    # ---- final save ----
    def save(self):
        npz_path = self._save_npz(os.path.join(self.output_dir,
                                                'cost_landscape.npz'))
        csv_path = self._save_csv(os.path.join(self.output_dir,
                                                'cost_landscape.csv'))
        logging.info(f"Cost landscape saved: {npz_path} ({len(self._evals)} evaluations)")
        logging.info(f"Cost landscape CSV:   {csv_path}")

    def _save_npz(self, path):
        evals = np.array(self._evals) if self._evals else np.empty((0,))
        timestamps = np.array(self._timestamps)
        gen_best = np.array(self._gen_best) if self._gen_best else np.empty((0,))
        np.savez_compressed(path,
                            evaluations=evals,
                            timestamps=timestamps,
                            gen_best=gen_best,
                            param_names=np.array(self.param_names))
        return path

    def _save_csv(self, path):
        header = ','.join(self.param_names + ['chi_squared', 'wall_time_s'])
        evals = np.array(self._evals) if self._evals else np.empty((0,))
        timestamps = np.array(self._timestamps)
        if evals.size == 0:
            with open(path, 'w') as f:
                f.write(header + '\n')
        else:
            data = np.column_stack([evals, timestamps])
            np.savetxt(path, data, delimiter=',', header=header, comments='')
        return path

    @staticmethod
    def load(path):
        """Load a saved landscape for post-hoc analysis."""
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}


def setup_logging(log_file):
    """Set up logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_experimental_data(file_path):
    """Load experimental data from the specified file"""
    data = np.loadtxt(file_path)
    temp = data[:, 0]
    spec_heat = data[:, 1]
    return temp, spec_heat


def load_multiple_experimental_data(exp_data_configs):
    """Load experimental data from multiple files"""
    exp_datasets = []
    for config in exp_data_configs:
        temp, spec_heat = load_experimental_data(config['file'])
        dataset = {
            'temp': temp,
            'spec_heat': spec_heat,
            'h': config.get('h', 0.0),
            'field_dir': config.get('field_dir', [0, 0, 1]),
            'weight': config.get('weight', 1.0)
        }
        if 'temp_min' in config:
            dataset['temp_min'] = config['temp_min']
        if 'temp_max' in config:
            dataset['temp_max'] = config['temp_max']
        exp_datasets.append(dataset)
    return exp_datasets


def apply_gaussian_broadening(temp, spec_heat, sigma, broadening_type='linear'):
    """Apply Gaussian broadening to specific heat data"""
    if sigma <= 0:
        return spec_heat
    
    if broadening_type == 'log':
        log_temp = np.log(temp)
        dt = np.mean(np.diff(log_temp))
        sigma_pixels = sigma / dt
        broadened = gaussian_filter1d(spec_heat, sigma=sigma_pixels, mode='nearest')
    else:
        dt = np.mean(np.diff(temp))
        sigma_pixels = sigma / dt
        broadened = gaussian_filter1d(spec_heat, sigma=sigma_pixels, mode='nearest')
    
    return broadened


def _n_lead_params(fixed_params, model):
    """Number of leading (non-broadening) fit dimensions in the parameter
    vector: the model couplings (possibly frozen via --fix_exchange) plus an
    optional free g_c that is always appended LAST.
    """
    if model == 'anisotropic':
        if fixed_params.get("fix_exchange", False):
            n = 0  # exchange held fixed (Stage-1 g_c-only fit)
        else:
            n = 5 if fixed_params.get("fit_J2", False) else 4
    elif model == 'kitaev':
        n = 4
    else:
        n = 3 if fixed_params.get("fit_Jz_ratio", False) else 2
    if fixed_params.get("fit_g_c", False):
        n += 1
    return n


def _unpack_g_c(params, fixed_params, model):
    """g_c to use this evaluation: the fitted value (last leading dim) when
    --fit_g_c is set, otherwise the fixed g_c from fixed_params."""
    if fixed_params.get("fit_g_c", False):
        return float(params[_n_lead_params(fixed_params, model) - 1])
    return float(fixed_params.get("g_c", 2.0))


def run_nlce_triangular(params, fixed_params, exp_temp, work_dir, h_field=None, temp_range=None, field_dir=None):
    """
    Run triangular lattice NLCE with the given parameters.

    Args:
        params: Parameter array depending on model type:
                - xxz_j1j2/kitaev: [J1, J2, ...]
                - anisotropic: [Jzz, Jpm, Jpmpm, Jzpm, ...]
        fixed_params: Dictionary of fixed parameters
        exp_temp: Experimental temperature array
        work_dir: Working directory for NLCE calculations
        h_field: Override magnetic field strength
        temp_range: Override temperature range
        field_dir: Override field direction (default: from fixed_params)
    
    Returns:
        calc_temp, calc_spec_heat arrays (in SI units J/(mol·K) if SI_units=True)
    """
    model = fixed_params.get("model", "xxz_j1j2")
    
    # Extract parameters based on model type
    if model == 'anisotropic':
        fit_J2_aniso = fixed_params.get("fit_J2", False)
        if fixed_params.get("fix_exchange", False):
            # Stage-1 g_c fit: exchange is held fixed at its frozen values.
            Jzz = fixed_params["fixed_Jzz"]
            Jpm = fixed_params["fixed_Jpm"]
            Jpmpm = fixed_params["fixed_Jpmpm"]
            Jzpm = fixed_params["fixed_Jzpm"]
            J2 = fixed_params.get("J2", 0.0)
        elif fit_J2_aniso:
            n_model_params = 5
            Jzz, Jpm, Jpmpm, Jzpm, J2 = params[:5]
        else:
            n_model_params = 4
            Jzz, Jpm, Jpmpm, Jzpm = params[:4]
            J2 = fixed_params.get("J2", 0.0)  # fixed NNN coupling (default 0)
        J1 = 1.0
        Gamma, Gamma_prime = None, None
    elif model == 'kitaev':
        n_model_params = 4
        J_H, J_K, Gamma, Gamma_prime = params[:4]
        J1, J2 = J_H, J_K  # Map to J1/J2 for NLCE runner
        Jzz, Jpm, Jpmpm, Jzpm = None, None, None, None
    else:
        fit_Jz_ratio = fixed_params.get("fit_Jz_ratio", False)
        if fit_Jz_ratio:
            n_model_params = 3
            J1, J2, Jz_ratio = params[:3]
        else:
            Jz_ratio = fixed_params.get("Jz_ratio", 1.0)
            n_model_params = 2
            J1, J2 = params[:2]
        Jzz, Jpm, Jpmpm, Jzpm = None, None, None, None
        Gamma, Gamma_prime = None, None
    
    # g_c is fitted (last leading dim) when --fit_g_c, else the fixed value.
    # It scales the Zeeman term (hz = h * field_dir_z * g_c), so no change to
    # the bare h (= mu_B B / k_B) is needed.
    g_c_value = _unpack_g_c(params, fixed_params, model)

    h_value = h_field if h_field is not None else fixed_params.get("h", 0.0)
    if field_dir is None:
        field_dir = fixed_params["field_dir"]
    
    temp_min = temp_range.get('temp_min', fixed_params["temp_min"]) if temp_range else fixed_params["temp_min"]
    temp_max = temp_range.get('temp_max', fixed_params["temp_max"]) if temp_range else fixed_params["temp_max"]
    
    # Write experimental temperature points to a file so the NLCE evaluates
    # C(T) at exactly the measurement temperatures — no interpolation needed.
    # Filter to [temp_min, temp_max] before writing.
    temp_mask = (exp_temp >= temp_min) & (exp_temp <= temp_max)
    filtered_temps = np.sort(exp_temp[temp_mask])
    
    temp_points_file = os.path.join(work_dir, 'temp_points.txt')
    np.savetxt(temp_points_file, filtered_temps, fmt='%.12e')
    
    # Build NLCE command for triangular lattice
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nlce_script = os.path.join(script_dir, '..', 'run', 'nlce_triangular.py')
    
    cmd = [
        sys.executable,
        nlce_script,
        '--max_order', str(fixed_params["max_order"]),
        '--h', f'{h_value:.12f}',
        '--ed_executable', str(fixed_params["ED_path"]),
        '--field_dir', f'{field_dir[0]:.12f}', f'{field_dir[1]:.12f}', f'{field_dir[2]:.12f}',
        '--temp_points_file', temp_points_file,
        '--temp_min', f'{temp_min:.8f}',
        '--temp_max', f'{temp_max:.8f}',
        '--temp_bins', str(fixed_params["temp_bins"]),
        '--model', model,
        '--method', fixed_params.get("ed_method", "FULL"),
        '--thermo',
        '--base_dir', work_dir,
        '--g_ab', f'{fixed_params.get("g_ab", 2.0):.12f}',
        '--g_c', f'{g_c_value:.12f}',
        '--resummation', fixed_params.get("resummation", "euler")
    ]
    
    # Symmetry threshold for --symm auto-selection
    if fixed_params.get("symm_threshold") is not None:
        cmd.extend(['--symm_threshold', str(fixed_params["symm_threshold"])])
    
    # Add model-specific parameters
    if model == 'anisotropic':
        cmd.extend(['--Jzz', f'{Jzz:.12f}'])
        cmd.extend(['--Jpm', f'{Jpm:.12f}'])
        cmd.extend(['--Jpmpm', f'{Jpmpm:.12f}'])
        cmd.extend(['--Jzpm', f'{Jzpm:.12f}'])
        if abs(J2) > 1e-10 or fixed_params.get('fit_J2', False):
            cmd.extend(['--J2', f'{J2:.12f}'])  # NNN isotropic Heisenberg
    elif model == 'kitaev':
        cmd.extend(['--J1', f'{J1:.12f}'])
        cmd.extend(['--J2', f'{J2:.12f}'])
        if Gamma is not None:
            cmd.extend(['--Gamma', f'{Gamma:.12f}'])
        if Gamma_prime is not None:
            cmd.extend(['--Gamma_prime', f'{Gamma_prime:.12f}'])
    else:
        cmd.extend(['--J1', f'{J1:.12f}'])
        cmd.extend(['--J2', f'{J2:.12f}'])
        cmd.extend(['--Jz_ratio', f'{Jz_ratio:.12f}'])
    
    if fixed_params.get("skip_cluster_gen", True):
        cmd.append('--skip_cluster_gen')
    
    # SI units for comparison with experimental data (J/(mol·K))
    if fixed_params.get("SI_units", True):
        cmd.append('--SI_units')
    
    # Parallel ED across clusters within this NLCE run
    if fixed_params.get("parallel_ed", False):
        cmd.append('--parallel')
        ed_cores = fixed_params.get("ed_num_cores", 0)
        if ed_cores > 0:
            cmd.extend(['--num_cores', str(ed_cores)])
    
    # Streaming-symmetry diagonalization: pass flag to NLCE runner.
    # The orbit basis is cached inside each cluster's ham dir (basis_cache/).
    # IMPORTANT: The orbit basis depends on which operator types are PRESENT
    # on each bond (encoded via edge labels in the automorphism finder).
    # If a coupling is zero, the corresponding bond term vanishes, which can
    # enlarge the automorphism group and produce an INVALID basis for
    # non-zero values of that coupling.  To guard against this, the fitter
    # runs a dedicated basis-seeding pass (before the optimizer loop) using
    # guaranteed-nonzero couplings, so the cached basis is valid for ALL
    # parameter combinations the optimizer may explore.
    if fixed_params.get("streaming_symmetry", False):
        cmd.append('--streaming-symmetry')
    
    if not fixed_params.get("skip_ham_prep", False):
        # Clean up old results
        ed_dir = os.path.join(work_dir, f'ed_results_order_{fixed_params["max_order"]}')
        if os.path.exists(ed_dir):
            shutil.rmtree(ed_dir)
        ham_dir = os.path.join(work_dir, f'hamiltonians_order_{fixed_params["max_order"]}')
        if os.path.exists(ham_dir):
            if fixed_params.get("streaming_symmetry", False):
                # Preserve basis_cache/ subdirectories inside each cluster's ham dir.
                # The basis was pre-seeded with all-nonzero couplings before the
                # optimizer loop, so it is valid for any coupling combination.
                for entry in os.listdir(ham_dir):
                    entry_path = os.path.join(ham_dir, entry)
                    if os.path.isdir(entry_path):
                        for sub in os.listdir(entry_path):
                            sub_path = os.path.join(entry_path, sub)
                            if sub == 'basis_cache':
                                continue  # keep cached orbit basis
                            if os.path.isdir(sub_path):
                                shutil.rmtree(sub_path)
                            else:
                                os.remove(sub_path)
                    else:
                        os.remove(entry_path)
            else:
                shutil.rmtree(ham_dir)
    else:
        cmd.append('--skip_ham_prep')
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        nlc_dir = os.path.join(work_dir, f'nlc_results_order_{fixed_params["max_order"]}')
        spec_heat_file = os.path.join(nlc_dir, 'nlc_specific_heat.txt')
        
        if not os.path.exists(spec_heat_file):
            logging.error(f"Specific heat file not found: {spec_heat_file}")
            return None, None
        
        calc_data = np.loadtxt(spec_heat_file)
        calc_temp = calc_data[:, 0]
        calc_spec_heat = calc_data[:, 1]
        
        return calc_temp, calc_spec_heat
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running NLCE: {e}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return None, None


def interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp):
    """Interpolate calculated data to match experimental temperature points"""
    if calc_temp is None or calc_spec_heat is None:
        return np.zeros_like(exp_temp)
    
    sort_idx = np.argsort(calc_temp)
    calc_temp = calc_temp[sort_idx]
    calc_spec_heat = calc_spec_heat[sort_idx]
    
    interp_func = interp1d(calc_temp, calc_spec_heat, kind='cubic', 
                          bounds_error=False, fill_value=np.nan)
    
    interp_spec_heat = interp_func(exp_temp)
    interp_spec_heat = np.nan_to_num(interp_spec_heat)
    
    return interp_spec_heat


def _ensure_field_work_dir(work_dir, field_idx, max_order):
    """Create per-field work directory with symlinked shared cluster directory.
    
    Each field value needs its own work directory to avoid conflicts in
    Hamiltonian/ED/NLC result files when running in parallel.
    The cluster directory is shared (read-only) via symlink.
    """
    field_dir_path = os.path.join(work_dir, f'field_{field_idx}')
    os.makedirs(field_dir_path, exist_ok=True)
    
    # Symlink shared cluster directory so nlce_triangular.py finds it
    src_cluster = os.path.abspath(
        os.path.join(work_dir, f'clusters_order_{max_order}'))
    dst_cluster = os.path.join(field_dir_path, f'clusters_order_{max_order}')
    
    if os.path.exists(src_cluster) and not os.path.lexists(dst_cluster):
        try:
            os.symlink(src_cluster, dst_cluster)
        except OSError:
            shutil.copytree(src_cluster, dst_cluster)
    
    return field_dir_path


def _compute_single_field(args_tuple):
    """Worker function: run NLCE for one field value.
    
    Designed to be called from ThreadPoolExecutor or sequentially.
    Returns (field_idx, calc_temp, calc_spec_heat).
    """
    field_idx, model_params, fixed_params, exp_temp, field_work_dir, \
        h_field, temp_range, field_dir = args_tuple
    calc_temp, calc_spec_heat = run_nlce_triangular(
        model_params, fixed_params, exp_temp, field_work_dir,
        h_field=h_field, temp_range=temp_range, field_dir=field_dir
    )
    return field_idx, calc_temp, calc_spec_heat


def calc_chi_squared(params, fixed_params, exp_datasets, work_dir):
    """Calculate chi-squared between experimental and calculated specific heat"""
    total_chi_squared = 0.0
    
    n_datasets = len(exp_datasets)
    fit_broadening = fixed_params.get("fit_broadening", False)
    model = fixed_params.get("model", "xxz_j1j2")
    save_snapshots = fixed_params.get("save_snapshots", False)
    snapshot_dir = fixed_params.get("snapshot_dir", work_dir)
    iteration_counter = fixed_params.get("iteration_counter", [0])
    
    # Leading (non-broadening) fit dims = model couplings (possibly frozen)
    # plus an optional free g_c. Everything past that is broadening sigmas.
    fit_Jz_ratio = fixed_params.get("fit_Jz_ratio", False)
    n_lead = _n_lead_params(fixed_params, model)

    model_params = params[:n_lead]
    if fit_broadening:
        sigmas = params[n_lead:n_lead+n_datasets]
    else:
        sigmas = [0.0] * n_datasets
    
    all_calc_temps = []
    all_calc_heats = []
    all_exp_temps = []
    all_exp_heats = []
    all_interp_heats = []
    
    # --- Phase 1: Run NLCE for all field values (parallel or sequential) ---
    parallel_fields = fixed_params.get("parallel_fields", False)
    max_workers = fixed_params.get("max_parallel_fields", 0)
    if max_workers <= 0:
        max_workers = n_datasets
    
    # Build task list for all datasets
    tasks = []
    for i, dataset in enumerate(exp_datasets):
        h_field = dataset.get('h', 0.0)
        ds_field_dir = dataset.get('field_dir', None)
        temp_range = {}
        if 'temp_min' in dataset:
            temp_range['temp_min'] = dataset['temp_min']
        if 'temp_max' in dataset:
            temp_range['temp_max'] = dataset['temp_max']
        
        # Each field gets its own work directory when running in parallel
        if parallel_fields and n_datasets > 1:
            field_work = _ensure_field_work_dir(
                work_dir, i, fixed_params["max_order"])
        else:
            field_work = work_dir
        
        tasks.append((
            i, model_params, fixed_params, dataset['temp'],
            field_work, h_field,
            temp_range if temp_range else None, ds_field_dir
        ))
    
    # Execute: parallel or sequential
    field_results = {}  # idx -> (calc_temp, calc_spec_heat)
    if parallel_fields and n_datasets > 1:
        n_workers = min(max_workers, n_datasets)
        logging.debug(f"Running {n_datasets} field values in parallel "
                      f"({n_workers} workers)")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_compute_single_field, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, ct, cs = future.result()
                    field_results[idx] = (ct, cs)
                except Exception as exc:
                    logging.error(f"Field {idx} NLCE failed: {exc}")
                    field_results[idx] = (None, None)
    else:
        for task in tasks:
            idx, ct, cs = _compute_single_field(task)
            field_results[idx] = (ct, cs)
    
    # --- Phase 2: Compute chi-squared from collected results ---
    for i, dataset in enumerate(exp_datasets):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        weight = dataset.get('weight', 1.0)
        
        calc_temp, calc_spec_heat = field_results[i]
        
        if calc_temp is None:
            total_chi_squared += 1e10 * weight
            continue
        
        all_calc_temps.append(calc_temp)
        all_calc_heats.append(calc_spec_heat)
        all_exp_temps.append(exp_temp)
        all_exp_heats.append(exp_spec_heat)
        
        # Match NLCE output to experimental temperatures.
        # When --temp_points_file is used, the NLCE evaluates C(T) directly
        # at the (filtered) experimental T values, so no interpolation needed.
        # Check if grids match; fall back to interpolation if they don't.
        temp_mask = np.ones_like(exp_temp, dtype=bool)
        if 'temp_min' in dataset:
            temp_mask &= exp_temp >= dataset['temp_min']
        if 'temp_max' in dataset:
            temp_mask &= exp_temp <= dataset['temp_max']
        
        if np.sum(temp_mask) == 0:
            continue
        
        if len(calc_temp) == np.sum(temp_mask) and np.allclose(calc_temp, np.sort(exp_temp[temp_mask]), rtol=1e-8):
            # Direct comparison — no interpolation
            matched_spec_heat = calc_spec_heat[np.argsort(calc_temp)]
            all_interp_heats.append(matched_spec_heat)
        else:
            # Fallback: interpolate (legacy path or mismatched grids)
            interp_spec_heat = interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp)
            matched_spec_heat = interp_spec_heat[temp_mask]
            all_interp_heats.append(interp_spec_heat)
        
        # Apply broadening if fitting
        if fit_broadening and sigmas[i] > 0:
            matched_spec_heat = apply_gaussian_broadening(
                exp_temp[temp_mask], matched_spec_heat, sigmas[i])
        
        # Calculate chi-squared
        diff = (exp_spec_heat[temp_mask] - matched_spec_heat)
        chi_sq = np.sum(diff**2) * weight
        total_chi_squared += chi_sq
    
    # Save snapshot for debugging
    if save_snapshots:
        iteration_counter[0] += 1
        iter_num = iteration_counter[0]
        snapshot_file = os.path.join(snapshot_dir, f'snapshot_iter_{iter_num:04d}.txt')
        with open(snapshot_file, 'w') as f:
            f.write(f"# Iteration {iter_num}\n")
            f.write(f"# Chi-squared: {total_chi_squared:.6f}\n")
            if model == 'anisotropic':
                _fit_J2 = fixed_params.get('fit_J2', False)
                if fixed_params.get("fix_exchange", False):
                    _base = (f"# Jzz={fixed_params['fixed_Jzz']:.6f} (fixed), "
                             f"Jpm={fixed_params['fixed_Jpm']:.6f} (fixed), "
                             f"Jpmpm={fixed_params['fixed_Jpmpm']:.6f} (fixed), "
                             f"Jzpm={fixed_params['fixed_Jzpm']:.6f} (fixed)")
                    if fixed_params.get("fit_g_c", False):
                        _base += f", g_c={params[_n_lead_params(fixed_params, model)-1]:.6f}"
                    f.write(_base + "\n")
                else:
                    _base = f"# Jzz={params[0]:.6f}, Jpm={params[1]:.6f}, Jpmpm={params[2]:.6f}, Jzpm={params[3]:.6f}"
                    if _fit_J2 and len(params) > 4:
                        _base += f", J2={params[4]:.6f}"
                    if fixed_params.get("fit_g_c", False):
                        _base += f", g_c={params[_n_lead_params(fixed_params, model)-1]:.6f}"
                    f.write(_base + "\n")
            elif model == 'kitaev':
                f.write(f"# J={params[0]:.6f}, K={params[1]:.6f}, Gamma={params[2]:.6f}, Gamma'={params[3]:.6f}\n")
            else:
                f.write(f"# J1={params[0]:.6f}, J2={params[1]:.6f}\n")
            f.write("#\n# Calculated specific heat:\n")
            f.write("# T(K)  C_calc(J/mol/K)\n")
            if len(all_calc_temps) > 0:
                for T, C in zip(all_calc_temps[0], all_calc_heats[0]):
                    f.write(f"{T:.6e} {C:.6e}\n")
            f.write("#\n# Experimental vs Calculated (at matched temperatures):\n")
            f.write("# T_exp  C_exp  C_calc  diff\n")
            if len(all_exp_temps) > 0 and len(all_interp_heats) > 0:
                exp_t = all_exp_temps[0]
                exp_c = all_exp_heats[0]
                calc_c = all_interp_heats[0]
                # Align arrays: calc_c may match a temp-masked subset
                if len(calc_c) == len(exp_t):
                    for T, C_exp, C_int in zip(exp_t, exp_c, calc_c):
                        f.write(f"{T:.6e} {C_exp:.6e} {C_int:.6e} {C_exp-C_int:.6e}\n")
                elif len(all_calc_temps) > 0:
                    # Direct-match path: calc arrays correspond to NLCE output temps
                    for T, C_int in zip(all_calc_temps[0], all_calc_heats[0]):
                        f.write(f"{T:.6e} {'':10s} {C_int:.6e}\n")
    
    # Log progress with appropriate parameter names
    model = fixed_params.get("model", "xxz_j1j2")
    if model == 'anisotropic':
        parts = []
        if not fixed_params.get("fix_exchange", False):
            parts += [f"Jzz={params[0]:.4f}", f"Jpm={params[1]:.4f}",
                      f"Jpmpm={params[2]:.4f}", f"Jzpm={params[3]:.4f}"]
            if fixed_params.get('fit_J2', False) and len(params) > 4:
                parts.append(f"J2={params[4]:.4f}")
        if fixed_params.get("fit_g_c", False):
            parts.append(f"g_c={params[_n_lead_params(fixed_params, model)-1]:.4f}")
        logging.info(f"Parameters: {', '.join(parts)}, Chi²={total_chi_squared:.4f}")
    elif model == 'kitaev':
        log_msg = f"J={params[0]:.4f}, K={params[1]:.4f}, Γ={params[2]:.4f}, Γ'={params[3]:.4f}"
        logging.info(f"Parameters: {log_msg}, Chi²={total_chi_squared:.4f}")
    else:
        log_msg = f"J1={params[0]:.4f}, J2={params[1]:.4f}"
        if fit_Jz_ratio:
            log_msg += f", Jz_ratio={params[2]:.4f}"
        logging.info(f"Parameters: {log_msg}, Chi²={total_chi_squared:.4f}")
    
    # Log to cost landscape accumulator if present
    landscape_logger = fixed_params.get("landscape_logger", None)
    if landscape_logger is not None:
        landscape_logger.log_evaluation(params, total_chi_squared)
    
    return total_chi_squared


def plot_results(exp_datasets, fixed_params, best_params, work_dir, output_dir):
    """Plot experimental data and best fit"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_datasets)))
    model = fixed_params.get("model", "xxz_j1j2")
    fit_Jz_ratio = fixed_params.get("fit_Jz_ratio", False)
    # Leading fit dims (model couplings possibly frozen + optional free g_c).
    n_model_params = _n_lead_params(fixed_params, model)

    # Use parallel fields for final plot if enabled
    n_datasets = len(exp_datasets)
    parallel_fields = fixed_params.get("parallel_fields", False)
    
    # Build and run all NLCE tasks
    tasks = []
    for i, dataset in enumerate(exp_datasets):
        h_field = dataset.get('h', 0.0)
        ds_field_dir = dataset.get('field_dir', None)
        temp_range = {}
        if 'temp_min' in dataset:
            temp_range['temp_min'] = dataset['temp_min']
        if 'temp_max' in dataset:
            temp_range['temp_max'] = dataset['temp_max']
        
        if parallel_fields and n_datasets > 1:
            field_work = _ensure_field_work_dir(
                work_dir, i, fixed_params["max_order"])
        else:
            field_work = work_dir
        
        tasks.append((
            i, best_params[:n_model_params], fixed_params, dataset['temp'],
            field_work, h_field,
            temp_range if temp_range else None, ds_field_dir
        ))
    
    # Execute in parallel or sequentially
    plot_results_map = {}
    if parallel_fields and n_datasets > 1:
        max_workers = fixed_params.get("max_parallel_fields", 0)
        if max_workers <= 0:
            max_workers = n_datasets
        with ThreadPoolExecutor(max_workers=min(max_workers, n_datasets)) as executor:
            futures = {
                executor.submit(_compute_single_field, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, ct, cs = future.result()
                    plot_results_map[idx] = (ct, cs)
                except Exception:
                    plot_results_map[idx] = (None, None)
    else:
        for task in tasks:
            idx, ct, cs = _compute_single_field(task)
            plot_results_map[idx] = (ct, cs)
    
    for i, (dataset, color) in enumerate(zip(exp_datasets, colors)):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset.get('h', 0.0)
        label_str = dataset.get('label', f'h={h_field:.3f}')
        
        calc_temp, calc_spec_heat = plot_results_map[i]
        
        # Plot experimental data
        plt.scatter(exp_temp, exp_spec_heat, c=[color], s=30, alpha=0.7, 
                   label=f'Exp ({label_str})')
        
        # Plot calculated data
        if calc_temp is not None:
            plt.plot(calc_temp, calc_spec_heat, c=color, lw=2, 
                    label=f'NLCE ({label_str})')
    
    if model == 'anisotropic':
        if fixed_params.get("fix_exchange", False):
            g_c_lbl = best_params[-1] if fixed_params.get("fit_g_c", False) else fixed_params.get("g_c", 2.0)
            title_str = (f'Triangular Lattice Fit (Anisotropic, exchange fixed): '
                         f'g_c={g_c_lbl:.3f}')
        else:
            title_str = (f'Triangular Lattice Fit (Anisotropic): '
                        f'Jzz={best_params[0]:.3f}, Jpm={best_params[1]:.3f}, '
                        f'Jpmpm={best_params[2]:.3f}, Jzpm={best_params[3]:.3f}')
    elif model == 'kitaev':
        title_str = (f'Triangular Lattice Fit (JK\u0393\u0393\'): '
                    f'J={best_params[0]:.3f}, K={best_params[1]:.3f}, '
                    f'\u0393={best_params[2]:.3f}, \u0393\'={best_params[3]:.3f}')
    else:
        title_str = f'Triangular Lattice Fit: J1={best_params[0]:.3f}, J2={best_params[1]:.3f}'
        if fit_Jz_ratio:
            title_str += f', Jxy/Jz={best_params[2]:.3f}'
    
    # Set labels based on SI units setting
    use_SI_units = fixed_params.get("SI_units", True)
    if use_SI_units:
        plt.xlabel('Temperature (K)')
        plt.ylabel('Specific Heat (J/(mol·K))')
    else:
        plt.xlabel('Temperature (J₁)')
        plt.ylabel('Specific Heat')
    plt.title(title_str)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_file = os.path.join(output_dir, 'specific_heat_fit.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_file}")
    plt.close()


def multi_start_optimization(obj_func, initial_params, bounds, n_starts=30, 
                            method='COBYLA', args=(), **kwargs):
    """Perform multi-start optimization"""
    best_result = None
    best_score = np.inf
    all_results = []
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # Use Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=len(bounds))
    samples = sampler.random(n=n_starts)
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    
    logging.info(f"Starting multi-start optimization with {n_starts} starts using {method}")
    
    for i, start_point in enumerate(scaled_samples):
        try:
            result = minimize(
                obj_func,
                start_point,
                method=method,
                bounds=bounds,
                args=args,
                options=kwargs.get('options', {})
            )
            all_results.append(result)
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
            
            logging.info(f"Start {i+1}/{n_starts}: chi²={result.fun:.4f}, params={result.x}")
            
        except Exception as e:
            logging.warning(f"Start {i+1}/{n_starts} failed: {e}")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description='Fit triangular lattice NLCE to specific heat data')
    
    # Input/output
    parser.add_argument('--exp_data', type=str, help='Path to experimental specific heat data')
    parser.add_argument('--exp_config', type=str, default=None,
                       help='Path to JSON config for multiple datasets')
    parser.add_argument('--output_dir', type=str, default='./fit_results_triangular',
                       help='Output directory')
    parser.add_argument('--work_dir', type=str, default='nlc_fit_triangular_work',
                       help='Working directory')
    parser.add_argument('--ed_executable', type=str, default='./build/ED',
                       help='Path to ED executable')
    
    # Initial guess and bounds
    parser.add_argument('--initial_J1', type=float, default=1.0, help='Initial J1')
    parser.add_argument('--initial_J2', type=float, default=0.0, help='Initial J2')
    parser.add_argument('--J1_min', type=float, default=-5.0, help='Min J1')
    parser.add_argument('--J1_max', type=float, default=5.0, help='Max J1')
    parser.add_argument('--J2_min', type=float, default=-5.0, help='Min J2')
    parser.add_argument('--J2_max', type=float, default=5.0, help='Max J2')
    
    # XXZ anisotropy ratio Jxy/Jz (for xxz_j1j2 model)
    # Convention: Jz = J1 (fixed), Jxy = Jz_ratio * J1
    parser.add_argument('--fit_Jz_ratio', action='store_true',
                       help='Fit Jxy/Jz ratio as a free parameter (adds Jz_ratio to fit params). '
                            'Convention: Jz=J1, Jxy=Jz_ratio*J1. '
                            'When not set, Jz_ratio is fixed at --Jz_ratio value.')
    parser.add_argument('--Jz_ratio', type=float, default=1.0,
                       help='Jxy/Jz ratio (default: 1.0 = isotropic Heisenberg). '
                            'Convention: Jz=J1, Jxy=Jz_ratio*J1. '
                            'Used as initial guess when --fit_Jz_ratio is set.')
    parser.add_argument('--Jz_ratio_min', type=float, default=0.0, help='Min Jz_ratio for fitting')
    parser.add_argument('--Jz_ratio_max', type=float, default=1.0, help='Max Jz_ratio for fitting')
    
    # Anisotropic model: optional NNN isotropic Heisenberg J2
    parser.add_argument('--fit_J2', action='store_true', default=False,
                       help='Fit NNN isotropic Heisenberg J2 (anisotropic model only). '
                            'Adds J2 as 5th free parameter bounded by [--J2_min, --J2_max].')
    parser.add_argument('--initial_J2_aniso', type=float, default=0.0,
                       help='Initial J2 for anisotropic+J2 fit (default: 0.0)')
    parser.add_argument('--J2_aniso_min', type=float, default=0.0,
                       help='Lower bound for J2 in anisotropic model (default: 0.0, i.e. AFM NNN only)')
    parser.add_argument('--J2_aniso_max', type=float, default=0.5,
                       help='Upper bound for J2 in anisotropic model (default: 0.5 K)')

    # Anisotropic model initial values and bounds
    parser.add_argument('--initial_Jzz', type=float, default=2, help='Initial Jzz')
    parser.add_argument('--initial_Jpm', type=float, default=1, help='Initial Jpm')
    parser.add_argument('--initial_Jpmpm', type=float, default=1, help='Initial Jpmpm')
    parser.add_argument('--initial_Jzpm', type=float, default=1, help='Initial Jzpm')
    parser.add_argument('--Jzz_min', type=float, default=-5.0, help='Min Jzz')
    parser.add_argument('--Jzz_max', type=float, default=5.0, help='Max Jzz')
    parser.add_argument('--Jpm_min', type=float, default=-5.0, help='Min Jpm')
    parser.add_argument('--Jpm_max', type=float, default=5.0, help='Max Jpm')
    parser.add_argument('--Jpmpm_min', type=float, default=-5.0, help='Min Jpmpm')
    parser.add_argument('--Jpmpm_max', type=float, default=5.0, help='Max Jpmpm')
    parser.add_argument('--Jzpm_min', type=float, default=-5.0, help='Min Jzpm')
    parser.add_argument('--Jzpm_max', type=float, default=5.0, help='Max Jzpm')

    # g-factor tuning (anisotropic model). For B||c data (field_dir=[0,0,1])
    # only g_c enters the Zeeman term, so g_ab is left fixed. g_c scales the
    # field: hz = h * field_dir_z * g_c, with h = mu_B B / k_B (bare).
    parser.add_argument('--fit_g_c', action='store_true', default=False,
                       help='Fit the out-of-plane g-factor g_c as a free parameter '
                            '(appended LAST to the fit vector), bounded by '
                            '[--g_c_min, --g_c_max], initial from --g_c. '
                            'High-field (B||c) Schottky peaks constrain g_c.')
    parser.add_argument('--g_c_min', type=float, default=3.0,
                       help='Lower bound for g_c when --fit_g_c (default 3.0)')
    parser.add_argument('--g_c_max', type=float, default=4.5,
                       help='Upper bound for g_c when --fit_g_c (default 4.5)')
    parser.add_argument('--fix_exchange', action='store_true', default=False,
                       help='Hold the anisotropic exchange (Jzz,Jpm,Jpmpm,Jzpm) '
                            'FIXED at their --initial_* values and do not fit them '
                            '(Stage-1 g_c-only fit). Requires --model anisotropic; '
                            'typically combined with --fit_g_c.')

    # JKΓΓ' (Kitaev) model initial values and bounds
    parser.add_argument('--initial_J_H', type=float, default=0.0, help='Initial Heisenberg coupling J')
    parser.add_argument('--initial_J_K', type=float, default=1.0, help='Initial Kitaev coupling K')
    parser.add_argument('--initial_Gamma', type=float, default=0.0, help='Initial Γ coupling')
    parser.add_argument('--initial_Gamma_prime', type=float, default=0.0, help="Initial Γ' coupling")
    parser.add_argument('--J_H_min', type=float, default=-5.0, help='Min J (Heisenberg)')
    parser.add_argument('--J_H_max', type=float, default=5.0, help='Max J (Heisenberg)')
    parser.add_argument('--J_K_min', type=float, default=-5.0, help='Min K (Kitaev)')
    parser.add_argument('--J_K_max', type=float, default=5.0, help='Max K (Kitaev)')
    parser.add_argument('--Gamma_min', type=float, default=-5.0, help='Min Γ')
    parser.add_argument('--Gamma_max', type=float, default=5.0, help='Max Γ')
    parser.add_argument('--Gamma_prime_min', type=float, default=-5.0, help="Min Γ'")
    parser.add_argument('--Gamma_prime_max', type=float, default=5.0, help="Max Γ'")
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=5, help='Maximum NLCE order (default: 5 for triangular lattice)')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1],
                       help='Field direction')
    parser.add_argument('--temp_bins', type=int, default=100, help='Temperature bins')
    parser.add_argument('--temp_min', type=float, default=0.01, help='Min temperature')
    parser.add_argument('--temp_max', type=float, default=10.0, help='Max temperature')
    parser.add_argument('--model', type=str, default='xxz_j1j2',
                       choices=['xxz_j1j2', 'kitaev', 'anisotropic'],
                       help='Spin model type')
    
    # Skip flags
    parser.add_argument('--skip_cluster_gen', action='store_true')
    parser.add_argument('--skip_ham_prep', action='store_true')
    
    # Anisotropic g-tensor for Zeeman coupling
    parser.add_argument('--g_ab', type=float, default=2.0,
                       help='In-plane g-factor (default: 2.0; NdMgAl11O19 magnetization: 1.54)')
    parser.add_argument('--g_c', type=float, default=2.0,
                       help='Out-of-plane g-factor (default: 2.0; NdMgAl11O19 magnetization: 3.75)')
    
    # SI units (default: True for comparison with experimental data)
    parser.add_argument('--SI_units', action='store_true', default=True,
                       help='Use SI units (J/(mol·K)) for specific heat output (default: True)')
    parser.add_argument('--no_SI_units', action='store_true',
                       help='Disable SI units (use natural units instead)')
    
    # Optimization
    parser.add_argument('--method', type=str, default='multi_start',
                       choices=['multi_start', 'differential_evolution',
                                'dual_annealing', 'bayesian'])
    parser.add_argument('--ed_method', type=str, default='FULL',
                       help='ED solver method passed to the NLCE runner '
                            '(FULL, FULL_GPU, SCALAPACK_MIXED, etc. Default: FULL)')
    parser.add_argument('--resummation', type=str, default='euler',
                       choices=['none', 'euler', 'wynn', 'wynn_multi',
                                'brezinski', 'aitken', 'pade'],
                       help='NLCE resummation method (default: euler).')
    parser.add_argument('--streaming-symmetry', action='store_true',
                       help='Use streaming-symmetry diagonalization (exploits spatial automorphisms). '
                            'The orbit basis is precomputed once and cached, then reused '
                            'across all fitting iterations since it only depends on cluster '
                            'geometry and operator structure, not coupling values.')
    parser.add_argument('--symm_threshold', type=int, default=None,
                       help='Site count threshold for --symm auto-selection in ED. '
                            'Clusters with num_sites > threshold get --symm. '
                            'Default: runner default (13). Set to 9 to cover all 10+ site clusters.')
    parser.add_argument('--n_starts', type=int, default=20, help='Number of random starts')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations')

    # Bayesian optimization settings
    parser.add_argument('--n_initial', type=int, default=20,
                       help='Number of initial random points for Bayesian optimization '
                            '(default: 20). These are evaluated before the GP model '
                            'starts guiding the search.')
    parser.add_argument('--acq_func', type=str, default='EI',
                       choices=['EI', 'LCB', 'PI', 'gp_hedge'],
                       help='Acquisition function for Bayesian optimization. '
                            'EI=Expected Improvement, LCB=Lower Confidence Bound, '
                            'PI=Probability of Improvement, '
                            'gp_hedge=auto-select (default: EI)')
    parser.add_argument('--xi', type=float, default=0.01,
                       help='Exploration-exploitation trade-off for EI/PI '
                            'acquisition (default: 0.01). Larger values explore more.')
    parser.add_argument('--kappa', type=float, default=1.96,
                       help='Exploration parameter for LCB acquisition '
                            '(default: 1.96). Larger values explore more.')
    parser.add_argument('--bo_log_transform', action='store_true',
                       help='Use log(chi²) as the BO objective. Compresses the '
                            'dynamic range so the GP surrogate fits much better '
                            'when chi² spans orders of magnitude.')
    parser.add_argument('--bo_noise', type=str, default='gaussian',
                       choices=['gaussian', '0'],
                       help='GP noise model: "gaussian" lets the GP estimate '
                            'observation noise (recommended for NLCE truncation '
                            'noise); "0" assumes noise-free (default: gaussian)')
    parser.add_argument('--bo_inject_x0', action='store_true', default=True,
                       help='Inject the user-supplied initial guess as the first '
                            'evaluation point instead of pure random init '
                            '(default: True)')
    parser.add_argument('--no_bo_inject_x0', action='store_false',
                       dest='bo_inject_x0',
                       help='Disable initial-guess injection')
    
    # Optional fitting parameters
    parser.add_argument('--fit_broadening', action='store_true')
    
    # Debug snapshots
    parser.add_argument('--save_snapshots', action='store_true',
                       help='Save diagnostic snapshots at each iteration for debugging')
    
    # Parallelism
    parser.add_argument('--parallel_fields', action='store_true',
                       help='Run NLCE for different field values in parallel. '
                            'Each field gets its own work directory to avoid conflicts. '
                            'Gives up to N× speedup for N field values.')
    parser.add_argument('--max_parallel_fields', type=int, default=0,
                       help='Max number of field values to compute simultaneously '
                            '(0 = number of datasets, i.e. all in parallel)')
    parser.add_argument('--parallel_ed', action='store_true',
                       help='Also parallelize ED across clusters within each NLCE run. '
                            'Combines with --parallel_fields for two-level parallelism.')
    parser.add_argument('--ed_num_cores', type=int, default=0,
                       help='Number of CPU cores for parallel ED per field value '
                            '(0 = auto: total_cores / num_parallel_fields)')
    
    # Cost landscape logging
    parser.add_argument('--save_landscape', action='store_true', default=True,
                       help='Save cost function landscape (all evaluations) to .npz and .csv '
                            '(default: True)')
    parser.add_argument('--no_save_landscape', action='store_true',
                       help='Disable cost landscape saving')

    # Warm-start (Bayesian optimization only): seed the GP with previously
    # evaluated (params, chi2) points from one or more cost_landscape.npz files
    # so prior compute is reused as already-evaluated x0/y0 (NOT re-run).
    parser.add_argument('--warm_start_npz', type=str, default=None,
                       help='Comma-separated path(s) to prior cost_landscape.npz file(s). '
                            'Their (params, chi2) evaluations are fed to the Bayesian '
                            'optimizer as known x0/y0 points (matched by parameter name, '
                            'filtered to the current bounds), so the GP is guided by them '
                            'without re-evaluating. Sets n_initial_points=0.')
    parser.add_argument('--warm_start_params_only', action='store_true',
                       help='Use only the PARAMETER locations from --warm_start_npz as '
                            'seed points (x0 with NO y0), so they are RE-EVALUATED under '
                            'the current objective. Use this when the objective changed '
                            '(e.g. different dataset weights) so the stored chi2 values '
                            'are no longer valid. The seeds form the initial design and '
                            'the GP refines from there.')
    parser.add_argument('--warm_start_top_k', type=int, default=0,
                       help='Keep only the K best prior points (lowest stored chi2) from '
                            '--warm_start_npz (0 = keep all). Useful with '
                            '--warm_start_params_only to cap re-evaluation cost while '
                            'still seeding the promising regions.')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'nlc_fit_triangular.log')
    setup_logging(log_file)
    
    # Set up working directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Load experimental data
    if args.exp_config:
        with open(args.exp_config, 'r') as f:
            config = json.load(f)
        exp_datasets = load_multiple_experimental_data(config['datasets'])
    elif args.exp_data:
        temp, spec_heat = load_experimental_data(args.exp_data)
        exp_datasets = [{
            'temp': temp,
            'spec_heat': spec_heat,
            'h': args.h,
            'field_dir': args.field_dir,
            'weight': 1.0
        }]
    else:
        logging.error("Must provide --exp_data or --exp_config")
        sys.exit(1)
    
    # Determine SI_units setting (--no_SI_units overrides default True)
    use_SI_units = args.SI_units and not args.no_SI_units
    
    # Set up snapshot directory if saving snapshots
    snapshot_dir = os.path.join(args.output_dir, 'snapshots')
    if args.save_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)
        logging.info(f"Saving diagnostic snapshots to: {snapshot_dir}")
    
    # Compute ED core allocation for parallel execution
    import multiprocessing as _mp
    _total_cores = _mp.cpu_count()
    _n_fields = len(exp_datasets)
    if args.ed_num_cores > 0:
        _ed_cores = args.ed_num_cores
    elif args.parallel_fields and args.parallel_ed and _n_fields > 1:
        # Split cores across parallel fields
        _n_parallel = min(args.max_parallel_fields or _n_fields, _n_fields)
        _ed_cores = max(1, _total_cores // _n_parallel)
    elif args.parallel_ed:
        _ed_cores = _total_cores
    else:
        _ed_cores = 0
    
    if args.parallel_fields:
        logging.info(f"Parallel fields enabled: up to "
                     f"{args.max_parallel_fields or _n_fields} field values simultaneously")
    if args.parallel_ed:
        logging.info(f"Parallel ED enabled: {_ed_cores} cores per field")
    
    # Fixed parameters
    fixed_params = {
        "max_order": args.max_order,
        "h": args.h,
        "field_dir": args.field_dir,
        "temp_bins": args.temp_bins,
        "temp_min": args.temp_min,
        "temp_max": args.temp_max,
        "model": args.model,
        "ED_path": args.ed_executable,
        "skip_cluster_gen": args.skip_cluster_gen,
        "skip_ham_prep": args.skip_ham_prep,
        "fit_broadening": args.fit_broadening,
        "n_datasets": len(exp_datasets),
        "SI_units": use_SI_units,
        "g_ab": args.g_ab,
        "g_c": args.g_c,
        "save_snapshots": args.save_snapshots,
        "snapshot_dir": snapshot_dir,
        "iteration_counter": [0],  # Mutable list to track iteration count
        "parallel_fields": args.parallel_fields,
        "max_parallel_fields": args.max_parallel_fields,
        "parallel_ed": args.parallel_ed,
        "ed_num_cores": _ed_cores,
        "ed_method": args.ed_method,
        "fit_Jz_ratio": args.fit_Jz_ratio,
        "Jz_ratio": args.Jz_ratio,
        "streaming_symmetry": args.streaming_symmetry,
        "symm_threshold": args.symm_threshold,
        "resummation": args.resummation,
    }
    
    # Generate clusters first if needed
    if not args.skip_cluster_gen:
        logging.info("Generating triangle-based NLCE clusters...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_gen_script = os.path.join(script_dir, '..', 'prep', 'generate_triangle_nlce_clusters.py')
        cluster_gen_cmd = [
            sys.executable,
            cluster_gen_script,
            '--max_order', str(args.max_order),
            '--output_dir', os.path.join(args.work_dir, f'clusters_order_{args.max_order}'),
            '--no_visualize',
        ]
        if args.model in ('kitaev', 'anisotropic'):
            # Direction-dependent couplings need bond-colored cluster
            # dedup: topologically isomorphic embeddings (straight vs
            # bent chains, ...) are NOT isospectral for these models.
            cluster_gen_cmd.append('--bond_colored')
        subprocess.run(cluster_gen_cmd, check=True)
        fixed_params["skip_cluster_gen"] = True
    
    # Set up parameter bounds based on model type
    if args.model == 'anisotropic':
        fit_exchange = not args.fix_exchange
        # --fix_exchange (Stage-1 g_c fit) implies J2 is not fit either.
        fit_J2 = args.fit_J2 and fit_exchange
        if fit_exchange:
            bounds = [
                (args.Jzz_min, args.Jzz_max),
                (args.Jpm_min, args.Jpm_max),
                (args.Jpmpm_min, args.Jpmpm_max),
                (args.Jzpm_min, args.Jzpm_max)
            ]
            initial_params = [args.initial_Jzz, args.initial_Jpm, args.initial_Jpmpm, args.initial_Jzpm]
            names_list = ["Jzz", "Jpm", "Jpmpm", "Jzpm"]
            if fit_J2:
                bounds.append((args.J2_aniso_min, args.J2_aniso_max))
                initial_params.append(args.initial_J2_aniso)
                names_list.append("J2")
        else:
            # Exchange frozen at the --initial_* values.
            bounds = []
            initial_params = []
            names_list = []
            fixed_params["fix_exchange"] = True
            fixed_params["fixed_Jzz"]   = args.initial_Jzz
            fixed_params["fixed_Jpm"]   = args.initial_Jpm
            fixed_params["fixed_Jpmpm"] = args.initial_Jpmpm
            fixed_params["fixed_Jzpm"]  = args.initial_Jzpm

        fixed_params["fit_J2"] = fit_J2
        fixed_params["J2"]     = args.initial_J2_aniso  # used when not fitting

        # g_c tuning: append as the LAST fit dimension (anisotropic only).
        fixed_params["fit_g_c"] = args.fit_g_c
        if args.fit_g_c:
            bounds.append((args.g_c_min, args.g_c_max))
            initial_params.append(args.g_c)
            names_list.append("g_c")

        if not bounds:
            logging.error("No free parameters: --fix_exchange without --fit_g_c "
                          "leaves nothing to optimize.")
            sys.exit(1)

        param_names = ", ".join(names_list)
        logging.info(f"Fitting anisotropic model: {param_names}")
        logging.info(
            f"Exchange {'FIXED' if not fit_exchange else 'free'} "
            f"(Jzz={args.initial_Jzz}, Jpm={args.initial_Jpm}, "
            f"Jpmpm={args.initial_Jpmpm}, Jzpm={args.initial_Jzpm})"
            + (f", fit g_c in [{args.g_c_min}, {args.g_c_max}] (init {args.g_c})"
               if args.fit_g_c else f", g_c fixed at {args.g_c}"))
    elif args.model == 'kitaev':
        bounds = [
            (args.J_H_min, args.J_H_max),
            (args.J_K_min, args.J_K_max),
            (args.Gamma_min, args.Gamma_max),
            (args.Gamma_prime_min, args.Gamma_prime_max)
        ]
        initial_params = [args.initial_J_H, args.initial_J_K, args.initial_Gamma, args.initial_Gamma_prime]
        param_names = "J, K, Gamma, Gamma_prime"
        
        logging.info(f"Fitting JKΓΓ' (Kitaev) model: {param_names}")
        logging.info(f"Initial: J={args.initial_J_H}, K={args.initial_J_K}, "
                    f"Γ={args.initial_Gamma}, Γ'={args.initial_Gamma_prime}")
    else:
        bounds = [(args.J1_min, args.J1_max), (args.J2_min, args.J2_max)]
        initial_params = [args.initial_J1, args.initial_J2]
        param_names = "J1, J2"
        
        # Add Jz_ratio as fitting parameter if requested
        if args.fit_Jz_ratio:
            bounds.append((args.Jz_ratio_min, args.Jz_ratio_max))
            initial_params.append(args.Jz_ratio)
            param_names += ", Jz_ratio"
        
        logging.info(f"Fitting {args.model} model: {param_names}")
        logging.info(f"Initial: J1={args.initial_J1}, J2={args.initial_J2}" +
                    (f", Jz_ratio={args.Jz_ratio}" if args.fit_Jz_ratio else ""))
    
    if args.fit_broadening:
        for _ in exp_datasets:
            bounds.append((0.0, 1.0))
            initial_params.append(0.1)
    
    # Set up cost landscape logger
    save_landscape = args.save_landscape and not args.no_save_landscape
    landscape_logger = None
    if save_landscape:
        landscape_param_names = [p.strip() for p in param_names.split(',')]
        landscape_logger = CostLandscapeLogger(
            param_names=landscape_param_names,
            output_dir=args.output_dir,
            flush_interval=50
        )
        fixed_params["landscape_logger"] = landscape_logger
        logging.info("Cost landscape logging enabled — "
                     "evaluations will be saved to cost_landscape.npz/csv")
    
    # --- Basis seeding pass (streaming-symmetry only) ---
    # The orbit basis depends on which operator types are PRESENT on each bond.
    # If a coupling parameter is exactly zero, the automorphism group enlarges
    # and the cached basis becomes incompatible with nonzero values of that
    # coupling.  To prevent this, we run a single NLCE pass with guaranteed-
    # nonzero "seed" couplings (ham prep + basis precompute only, no ED/NLC)
    # BEFORE the optimizer loop.  The cached basis is then valid for any
    # coupling combination the optimizer explores.
    if args.streaming_symmetry:
        logging.info("="*80)
        logging.info("Basis seeding pass: precomputing orbit basis with all couplings nonzero")
        logging.info("="*80)
        
        # Choose small nonzero seed values for ALL coupling parameters.
        # Exact values don't matter — only the operator structure (which bond
        # types are present) affects the automorphism group.
        _SEED = 0.12345  # arbitrary nonzero value
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        nlce_script = os.path.join(script_dir, '..', 'run', 'nlce_triangular.py')
        
        def _seed_basis_in_work_dir(seed_work_dir):
            """Run ham-prep + basis-precompute in a single work directory."""
            seed_cmd = [
                sys.executable, nlce_script,
                '--max_order', str(args.max_order),
                '--h', '0.0',
                '--ed_executable', str(args.ed_executable),
                '--field_dir', '0', '0', '1',
                '--temp_min', '1.0', '--temp_max', '2.0', '--temp_bins', '2',
                '--model', args.model,
                '--method', fixed_params.get("ed_method", "FULL"),
                '--base_dir', seed_work_dir,
                '--g_ab', f'{args.g_ab:.12f}',
                '--g_c', f'{args.g_c:.12f}',
                '--streaming-symmetry',
                '--skip_cluster_gen',
                '--skip_ed',     # don't run ED
                '--skip_nlc',    # don't run NLC summation
            ]
            # Add model-specific seed couplings (ALL nonzero)
            if args.model == 'anisotropic':
                seed_cmd.extend(['--Jzz', f'{_SEED}', '--Jpm', f'{_SEED}',
                                 '--Jpmpm', f'{_SEED}', '--Jzpm', f'{_SEED}'])
            elif args.model == 'kitaev':
                seed_cmd.extend(['--J1', f'{_SEED}', '--J2', f'{_SEED}',
                                 '--Gamma', f'{_SEED}', '--Gamma_prime', f'{_SEED}'])
            else:
                seed_cmd.extend(['--J1', f'{_SEED}', '--J2', f'{_SEED}',
                                 '--Jz_ratio', f'{_SEED}'])
            
            try:
                subprocess.run(seed_cmd, check=True, capture_output=True)
                logging.info(f"Basis seeded in {seed_work_dir}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Basis seeding failed in {seed_work_dir}: "
                              f"{e.stderr.decode('utf-8')}")
                raise
        
        # Seed basis in the directories that the optimizer will use
        if args.parallel_fields and len(exp_datasets) > 1:
            # Each field gets its own work dir; seed basis in each
            for i in range(len(exp_datasets)):
                field_work = _ensure_field_work_dir(
                    args.work_dir, i, args.max_order)
                _seed_basis_in_work_dir(field_work)
        else:
            _seed_basis_in_work_dir(args.work_dir)
        
        logging.info("Basis seeding complete — cached orbit basis will be reused "
                     "across all fitting iterations")
        logging.info("="*80)
    
    # Run optimization
    logging.info("Starting optimization...")
    
    if args.method == 'multi_start':
        result = multi_start_optimization(
            calc_chi_squared,
            initial_params,
            bounds,
            n_starts=args.n_starts,
            method='COBYLA',
            args=(fixed_params, exp_datasets, args.work_dir),
            options={'maxiter': args.max_iter}
        )
    elif args.method == 'differential_evolution':
        de_kwargs = dict(
            args=(fixed_params, exp_datasets, args.work_dir),
            maxiter=args.max_iter,
            seed=42,
            disp=True
        )
        if landscape_logger is not None:
            de_kwargs['callback'] = landscape_logger.de_callback
        result = differential_evolution(
            calc_chi_squared,
            bounds,
            **de_kwargs
        )
    elif args.method == 'dual_annealing':
        result = dual_annealing(
            calc_chi_squared,
            bounds,
            args=(fixed_params, exp_datasets, args.work_dir),
            maxiter=args.max_iter,
            seed=42
        )
    elif args.method == 'bayesian':
        if not HAS_SKOPT:
            logging.error("scikit-optimize is required for Bayesian optimization. "
                          "Install it with: pip install scikit-optimize")
            sys.exit(1)
        # Build skopt search space from bounds, skipping frozen dims (lo==hi)
        all_names = [n.strip() for n in param_names.split(',')]
        free_idx = [i for i, (lo, hi) in enumerate(bounds) if lo != hi]
        frozen_idx = [i for i, (lo, hi) in enumerate(bounds) if lo == hi]
        frozen_vals = {i: bounds[i][0] for i in frozen_idx}
        if frozen_idx:
            logging.info(f"Frozen parameters (lo==hi): "
                         f"{', '.join(all_names[i] + f'={frozen_vals[i]}' for i in frozen_idx)}")
        dimensions = [Real(bounds[i][0], bounds[i][1], name=all_names[i])
                      for i in free_idx]
        n_calls = args.max_iter
        n_initial_points = min(args.n_initial, n_calls)
        # xi/kappa are top-level kwargs in skopt's gp_minimize
        extra_kwargs = {}
        if args.acq_func in ('EI', 'PI'):
            extra_kwargs['xi'] = args.xi
        elif args.acq_func == 'LCB':
            extra_kwargs['kappa'] = args.kappa

        # GP noise model
        bo_noise = args.bo_noise
        if bo_noise == '0':
            bo_noise = 1e-10  # effectively noiseless
        extra_kwargs['noise'] = bo_noise

        bo_callbacks = []
        if landscape_logger is not None:
            bo_callbacks.append(landscape_logger.bo_callback)

        # Optionally log-transform the objective so the GP surrogate
        # models log(χ²) instead of raw χ². This compresses the
        # huge dynamic range (e.g. 20k–3.6M) into ~3–15, giving the
        # GP a much easier function to fit and dramatically improving
        # acquisition quality.
        use_log = args.bo_log_transform

        # ── Warm start: reuse prior (params, chi2) evaluations ──────────────
        # If one or more prior cost_landscape.npz files are supplied, feed
        # their evaluations to the GP as already-known x0/y0 points (matched
        # to the current free dimensions by parameter name, filtered to the
        # current bounds). These are NOT re-evaluated. This takes precedence
        # over --bo_inject_x0 (the injected guess is already covered).
        x0_list = None
        y0_list = None
        if args.warm_start_npz:
            all_names_bo = [n.strip() for n in param_names.split(',')]
            warm_pts = []  # list of (xfree, chi2_old)
            for npz_path in args.warm_start_npz.split(','):
                npz_path = npz_path.strip()
                if not npz_path:
                    continue
                if not os.path.exists(npz_path):
                    logging.warning(f"Warm-start file not found, skipping: {npz_path}")
                    continue
                d = np.load(npz_path, allow_pickle=True)
                ev = d['evaluations']
                names = [str(x) for x in d['param_names']]
                if ev.size == 0:
                    continue
                col = {nm: k for k, nm in enumerate(names)}
                if not all(all_names_bo[i] in col for i in free_idx):
                    logging.warning(f"Warm-start {npz_path}: param names "
                                    f"{names} do not cover free dims "
                                    f"{[all_names_bo[i] for i in free_idx]}; skipping.")
                    continue
                kept = 0
                for row in ev:
                    chi2 = float(row[-1])
                    if not np.isfinite(chi2) or chi2 <= 0:
                        continue
                    xfree = [float(row[col[all_names_bo[i]]]) for i in free_idx]
                    if not all(bounds[i][0] <= xfree[j] <= bounds[i][1]
                               for j, i in enumerate(free_idx)):
                        continue
                    warm_pts.append((xfree, chi2))
                    kept += 1
                logging.info(f"Warm-start {os.path.basename(npz_path)}: "
                             f"loaded {kept}/{len(ev)} usable points.")

            # Optionally keep only the K best (lowest stored chi2) points.
            if args.warm_start_top_k and args.warm_start_top_k > 0 \
                    and len(warm_pts) > args.warm_start_top_k:
                warm_pts.sort(key=lambda t: t[1])
                warm_pts = warm_pts[:args.warm_start_top_k]
                logging.info(f"Warm-start: kept top {len(warm_pts)} points "
                             f"by stored chi2.")

            if warm_pts:
                warm_x0 = [p[0] for p in warm_pts]
                if args.warm_start_params_only:
                    # Objective changed: stored chi2 is invalid. Re-evaluate the
                    # seed locations under the current objective; they form the
                    # initial design and the GP refines from there. With x0 and
                    # no y0, skopt evaluates the seeds AND counts them toward
                    # n_calls, requiring n_calls >= len(x0) + n_initial_points;
                    # we use n_initial_points=0 so there are no extra random
                    # points beyond the seeds.
                    if len(warm_x0) >= n_calls:
                        warm_x0 = warm_x0[:max(1, n_calls - 1)]
                        logging.warning(f"Warm-start seeds ({len(warm_pts)}) >= "
                                        f"n_calls ({n_calls}); truncating seeds to "
                                        f"{len(warm_x0)} to leave room for GP calls.")
                    x0_list = warm_x0
                    y0_list = None
                    n_initial_points = 0
                    logging.info(f"Warm-start (params-only): {len(warm_x0)} prior "
                                 f"parameter locations will be RE-EVALUATED as the "
                                 f"initial design (stored chi2 ignored); "
                                 f"{n_calls - len(warm_x0)} further model-guided "
                                 f"calls after that.")
                else:
                    x0_list = warm_x0
                    y0_list = [float(np.log(p[1])) if use_log else p[1]
                               for p in warm_pts]
                    n_initial_points = 0
                    logging.info(f"Warm-start: seeding GP with {len(warm_x0)} prior "
                                 f"evaluations (reusing stored chi2); "
                                 f"n_initial_points set to 0, "
                                 f"{n_calls} new model-guided calls to run.")
            else:
                logging.warning("Warm-start requested but no usable prior "
                                "points were loaded; falling back to normal init.")

        # Inject user-supplied initial guess as x0 so the GP has a
        # strong anchor near the expected optimum instead of relying
        # entirely on random initialization. (Skipped when warm-starting.)
        if x0_list is None and args.bo_inject_x0:
            x0_free = [initial_params[i] for i in free_idx]
            x0_list = [x0_free]
            logging.info(f"Injecting initial guess as x0: {x0_free}")

        def _expand_params(x_free):
            """Expand reduced BO vector back to full parameter vector."""
            full = np.empty(len(bounds))
            for j, idx in enumerate(free_idx):
                full[idx] = x_free[j]
            for idx, val in frozen_vals.items():
                full[idx] = val
            return full

        def _bo_objective(x):
            full_x = _expand_params(x)
            chi2 = calc_chi_squared(full_x, fixed_params,
                                    exp_datasets, args.work_dir)
            return float(np.log(chi2)) if use_log else chi2

        logging.info(f"Bayesian optimization: {n_calls} calls "
                     f"({n_initial_points} initial random, "
                     f"acq_func={args.acq_func})"
                     + (" [log-transformed objective]" if use_log else "")
                     + f" noise={bo_noise}")
        bo_result = gp_minimize(
            _bo_objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=args.acq_func,
            x0=x0_list,
            y0=y0_list,
            random_state=42,
            verbose=True,
            callback=bo_callbacks if bo_callbacks else None,
            **extra_kwargs,
        )
        # Wrap into a scipy-like result for downstream code
        class _BOResult:
            pass
        result = _BOResult()
        result.x = _expand_params(bo_result.x)
        # Convert back from log space if needed
        result.fun = float(np.exp(bo_result.fun)) if use_log else bo_result.fun
        result.nfev = len(bo_result.func_vals)
        logging.info(f"BO finished: {result.nfev} evaluations, "
                     f"best χ²={result.fun:.6f}")
    
    if result is None:
        logging.error("Optimization failed!")
        sys.exit(1)
    
    best_params = result.x
    best_chi_sq = result.fun
    
    logging.info("="*80)
    logging.info("Optimization completed!")
    
    if args.model == 'anisotropic':
        # Map the (variable-length) fitted vector back onto named params,
        # then fill in any frozen exchange / fixed g_c so the record is complete.
        _names = [n.strip() for n in param_names.split(',')] if param_names else []
        _fitted = {nm: float(v) for nm, v in zip(_names, best_params)}
        bp = {
            'Jzz':   _fitted.get('Jzz',   float(fixed_params.get('fixed_Jzz',   args.initial_Jzz))),
            'Jpm':   _fitted.get('Jpm',   float(fixed_params.get('fixed_Jpm',   args.initial_Jpm))),
            'Jpmpm': _fitted.get('Jpmpm', float(fixed_params.get('fixed_Jpmpm', args.initial_Jpmpm))),
            'Jzpm':  _fitted.get('Jzpm',  float(fixed_params.get('fixed_Jzpm',  args.initial_Jzpm))),
        }
        if 'J2' in _fitted:
            bp['J2'] = _fitted['J2']
        bp['g_c'] = _fitted.get('g_c', float(args.g_c))
        for _k, _v in bp.items():
            logging.info(f"Best {_k}: {_v:.6f}")
        results_dict = {
            'best_params': bp,
            'chi_squared': float(best_chi_sq),
            'fixed_params': fixed_params
        }
    elif args.model == 'kitaev':
        logging.info(f"Best J (Heisenberg): {best_params[0]:.6f}")
        logging.info(f"Best K (Kitaev): {best_params[1]:.6f}")
        logging.info(f"Best Γ: {best_params[2]:.6f}")
        logging.info(f"Best Γ': {best_params[3]:.6f}")
        results_dict = {
            'best_params': {
                'J': float(best_params[0]),
                'K': float(best_params[1]),
                'Gamma': float(best_params[2]),
                'Gamma_prime': float(best_params[3])
            },
            'chi_squared': float(best_chi_sq),
            'fixed_params': fixed_params
        }
    else:
        logging.info(f"Best J1: {best_params[0]:.6f}")
        logging.info(f"Best J2: {best_params[1]:.6f}")
        results_dict = {
            'best_params': {
                'J1': float(best_params[0]),
                'J2': float(best_params[1])
            },
            'chi_squared': float(best_chi_sq),
            'fixed_params': fixed_params
        }
    
    logging.info(f"Best chi-squared: {best_chi_sq:.6f}")
    logging.info("="*80)
    
    # Save cost landscape data
    if landscape_logger is not None:
        landscape_logger.save()
    
    # Remove non-serializable objects from fixed_params for JSON output
    serializable_fixed_params = {k: v for k, v in fixed_params.items()
                                  if k not in ('landscape_logger',)}
    results_dict['fixed_params'] = serializable_fixed_params
    
    # Save results
    with open(os.path.join(args.output_dir, 'fit_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2, cls=NumpyJSONEncoder)
    
    # Plot results
    plot_results(exp_datasets, fixed_params, best_params, args.work_dir, args.output_dir)
    
    logging.info("Fitting completed successfully!")


if __name__ == "__main__":
    main()
