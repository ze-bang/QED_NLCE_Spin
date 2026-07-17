import os
import numpy as np
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
import argparse

#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation utility.
Calculates thermodynamic properties of a lattice using cluster expansion.
"""

import matplotlib.pyplot as plt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. HDF5 file reading will not be available.")


# Module-level helper so it is picklable for multiprocessing.
def _parse_cluster_file(file_path):
    """Parse one cluster header file. Returns a dict or None on failure."""
    match = re.search(r'cluster_(\d+)_order_(\d+)', file_path)
    if not match:
        return None
    cluster_id, order = int(match.group(1)), int(match.group(2))
    multiplicity = None
    num_vertices = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("# Multiplicity") and ":" in line:
                    multiplicity = float(line.split(":")[-1].strip())
                elif line.startswith("# Number of vertices:"):
                    num_vertices = int(line.split(":")[1].strip())
                if multiplicity is not None and num_vertices is not None:
                    break
    except OSError:
        return None
    if multiplicity is None or num_vertices is None:
        return None
    return {
        'cluster_id': cluster_id,
        'order': order,
        'multiplicity': multiplicity,
        'num_vertices': num_vertices,
        'file_path': file_path,
    }


class NLCExpansion:
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min, temp_max, num_temps, measure_spin, SI_units=False,
                 energy_unit="dimensionless"):
        """
        Initialize the NLC expansion calculator.

        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_values: Array of inverse temperature values to evaluate
            energy_unit: Unit of the ED eigenvalues / P(T) energies relative
                to the KELVIN temperature grid. The kernel is otherwise
                unit-agnostic (kB = 1: T and E in the same units).
                "dimensionless" / "K" (default): no conversion.
                "meV": couplings were written in meV -> every energy is
                multiplied by 1/kB = 11.6045 K/meV (and stored P(T) grids
                rescaled likewise) so exp(-E/kB T) is dimensionless against
                a Kelvin grid.
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir

        self.SI = SI_units  # Flag for SI units
        # CODATA: kB = 8.617333262e-2 meV/K.
        if energy_unit not in ("dimensionless", "K", "meV"):
            raise ValueError(f"energy_unit must be dimensionless/K/meV, "
                             f"got {energy_unit!r}")
        self.energy_unit = energy_unit
        self.e_scale = (1.0 / 8.617333262e-2) if energy_unit == "meV" else 1.0
        if self.e_scale != 1.0:
            print(f"Energy unit: {energy_unit} -> scaling all ED energies "
                  f"by 1/kB = {self.e_scale:.6f} K/{energy_unit}")
        self.measure_spin = measure_spin  # Flag for measuring spin expectation values
        self.temp_values = np.logspace(np.log(temp_min)/np.log(10), np.log(temp_max)/np.log(10), num_temps)  # Default temperature range

        self.clusters = {}  # Will store {cluster_id: {order, multiplicity, eigenvalues, etc.}}
        self.weights = {}   # Will store calculated weights for each cluster and property
        
        # Lanczos-boost tracking
        self.truncated_clusters = {}  # {cluster_id: {num_eigenvalues, hilbert_dim, energy_cutoff}}
        self.lanczos_boost_active = False
        self.lanczos_boost_temp_max = None  # Maximum valid temperature for Lanczos-boosted results
        
    def read_clusters(self):
        """Read all cluster information from files in the cluster directory.

        Parsing each ``cluster_*_order_*.dat`` file is independent (read +
        regex on a few header lines), so we parallelise across files when
        the cluster set is large enough to amortise pool startup. Falls
        back to a serial loop for small inputs and on any error.
        """
        pattern = os.path.join(self.cluster_dir, "cluster_*_order_*.dat")
        cluster_files = glob.glob(pattern)

        # Small inputs: serial is faster than starting workers.
        PARALLEL_THRESHOLD = 64
        if len(cluster_files) < PARALLEL_THRESHOLD:
            results = [_parse_cluster_file(p) for p in cluster_files]
        else:
            try:
                import concurrent.futures
                max_workers = min(os.cpu_count() or 1, 8)
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                    # chunksize amortises IPC overhead for thousands of small files
                    results = list(ex.map(_parse_cluster_file, cluster_files, chunksize=32))
            except Exception as e:  # pragma: no cover - defensive
                print(f"Warning: parallel cluster parse failed ({e}); falling back to serial")
                results = [_parse_cluster_file(p) for p in cluster_files]

        for parsed in results:
            if parsed is None:
                continue
            cluster_id = parsed['cluster_id']
            self.clusters[cluster_id] = {
                'order': parsed['order'],
                'multiplicity': parsed['multiplicity'],
                'num_vertices': parsed['num_vertices'],
                'file_path': parsed['file_path'],
                'eigenvalues': None,
                'sp': None,
                'sm': None,
            }
    def read_eigenvalues(self):
        """Read eigenvalues for each cluster from ED output files (HDF5 format).
        
        Also detects Lanczos-boosted clusters with truncated spectra and tracks
        their metadata for temperature validity estimation.
        """
        for cluster_id in self.clusters:
            cluster_base_dir = os.path.join(
                self.eigenvalue_dir, 
                f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}"
            )
            cluster_output_dir = os.path.join(cluster_base_dir, "output")
            
            # Check for Lanczos-boost metadata file
            lanczos_boost_file = os.path.join(cluster_base_dir, "lanczos_boost_info.txt")
            if os.path.exists(lanczos_boost_file):
                self.lanczos_boost_active = True
                metadata = {}
                with open(lanczos_boost_file, 'r') as f:
                    for line in f:
                        if line.startswith('#') or ':' not in line:
                            continue
                        key, value = line.strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
                
                self.truncated_clusters[cluster_id] = {
                    'num_eigenvalues': int(metadata.get('num_eigenvalues', 0)),
                    'hilbert_dim': int(metadata.get('hilbert_dim', 0)),
                    'order': int(metadata.get('order', 0)),
                    'num_sites': int(metadata.get('num_sites', 0)),
                }
            
            # Try HDF5 file first (new format)
            h5_file = os.path.join(cluster_output_dir, "ed_results.h5")
            if HAS_H5PY and os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        if ('/eigendata/eigenvalues' in f
                                and len(f['/eigendata/eigenvalues']) > 0):
                            eigenvalues = f['/eigendata/eigenvalues'][:]
                            self.clusters[cluster_id]['eigenvalues'] = np.array(eigenvalues)

                            # For truncated clusters, calculate energy cutoff
                            if cluster_id in self.truncated_clusters:
                                E_min = np.min(eigenvalues)
                                E_max = np.max(eigenvalues)
                                self.truncated_clusters[cluster_id]['energy_cutoff'] = E_max - E_min
                            continue
                        # Large clusters solved by OFTLM carry no eigenvalue
                        # spectrum -- only the P(T) curves. Read them directly;
                        # the summation consumes them via _cluster_thermo_quantities.
                        if '/thermodynamics/temperatures' in f:
                            tg = f['/thermodynamics']
                            pt = {
                                'temperatures': tg['temperatures'][:],
                                'energy': tg['energy'][:],
                                'specific_heat': tg['specific_heat'][:],
                                'entropy': tg['entropy'][:],
                            }
                            # Independent-seed resampling errors (OFTLM):
                            # propagated through the weight subtraction so
                            # the final sum carries an honest stochastic band.
                            for prop in ('energy', 'specific_heat', 'entropy'):
                                key = f'std_error_{prop}'
                                if key in tg:
                                    pt[key] = tg[key][:]
                            self.clusters[cluster_id]['thermo_pt'] = pt
                            self.clusters[cluster_id]['eigenvalues'] = None
                            # Track stochastic provenance: OFTLM P(T) carries
                            # sample noise, which the weight subtraction can
                            # amplify -- see the cancellation diagnostic in
                            # calculate_weights.
                            method = f['/eigendata'].attrs.get('method', b'')
                            if isinstance(method, bytes):
                                method = method.decode()
                            self.clusters[cluster_id]['is_stochastic'] = (
                                str(method).upper() == 'OFTLM')
                            continue
                except Exception as e:
                    print(f"Warning: Error reading HDF5 file for cluster {cluster_id}: {e}")
            
            # Fall back to legacy text file format
            eigenvalue_file = os.path.join(cluster_output_dir, "eigenvalues.txt")
            if os.path.exists(eigenvalue_file):
                with open(eigenvalue_file, 'r') as f:
                    eigenvalues = [float(line.strip()) for line in f if line.strip()]
                eigenvalues = np.array(eigenvalues)
                self.clusters[cluster_id]['eigenvalues'] = eigenvalues
                
                # For truncated clusters, calculate energy cutoff
                if cluster_id in self.truncated_clusters:
                    E_min = np.min(eigenvalues)
                    E_max = np.max(eigenvalues)
                    self.truncated_clusters[cluster_id]['energy_cutoff'] = E_max - E_min
                continue
            
            print(f"Warning: Eigenvalue data not found for cluster {cluster_id}")
        
        # Estimate maximum valid temperature for Lanczos-boosted results
        if self.lanczos_boost_active and self.truncated_clusters:
            self._estimate_lanczos_boost_temp_validity()
    
    def _estimate_lanczos_boost_temp_validity(self):
        """
        Estimate the maximum valid temperature for Lanczos-boosted NLCE results.
        
        The truncation error is approximately:
            error ≈ exp(-E_cutoff / T) × (remaining density of states)
        
        For T << E_cutoff/5, the error is negligible (< 1%).
        We use the minimum energy cutoff across all truncated clusters.
        """
        energy_cutoffs = []
        for cluster_id, info in self.truncated_clusters.items():
            if 'energy_cutoff' in info and info['energy_cutoff'] > 0:
                energy_cutoffs.append(info['energy_cutoff'])
        
        if energy_cutoffs:
            # Use the minimum energy cutoff (most restrictive)
            min_cutoff = min(energy_cutoffs)
            # Conservative estimate: T_max ≈ E_cutoff / 5 for < 1% error
            self.lanczos_boost_temp_max = min_cutoff / 5.0
            
            print(f"\n{'='*70}")
            print(f"LANCZOS-BOOST MODE DETECTED")
            print(f"{'='*70}")
            print(f"  Truncated clusters: {len(self.truncated_clusters)}")
            print(f"  Minimum energy cutoff: {min_cutoff:.4f}")
            print(f"  Estimated valid temperature range: T < {self.lanczos_boost_temp_max:.4f}")
            print(f"  (Results for T > {self.lanczos_boost_temp_max:.4f} may have significant errors)")
            print(f"{'='*70}\n")
            
            # Detailed info for each truncated cluster
            for cluster_id, info in sorted(self.truncated_clusters.items()):
                num_eig = info.get('num_eigenvalues', '?')
                dim = info.get('hilbert_dim', '?')
                cutoff = info.get('energy_cutoff', 0)
                fraction = num_eig / dim if isinstance(num_eig, int) and isinstance(dim, int) else '?'
                print(f"  Cluster {cluster_id} (order {info.get('order', '?')}): "
                      f"{num_eig}/{dim} eigenvalues ({fraction*100:.1f}%), "
                      f"E_cutoff = {cutoff:.4f}")
    
    def _cluster_thermo_quantities(self, cluster_id):
        """Per-cluster {energy, specific_heat, entropy} on ``self.temp_values``.

        Dense clusters: computed from the eigenvalue spectrum. OFTLM clusters
        (no spectrum): the pre-computed P(T) curves interpolated onto the
        summation grid (both grids are log-spaced from temp_min/max, so this is
        near-exact).
        """
        ev = self.clusters[cluster_id].get('eigenvalues')
        if ev is not None:
            return self.calculate_thermodynamic_quantities(ev)
        pt = self.clusters[cluster_id].get('thermo_pt')
        if pt is None:
            raise ValueError(
                f"cluster {cluster_id}: neither eigenvalues nor P(T) available")
        # energy_unit=meV: the stored P(T) was computed with T on the
        # COUPLING (meV) scale -- rescale its temperature axis AND its
        # energy values by 1/kB; Cv and S are dimensionless, unchanged.
        Tc = np.asarray(pt['temperatures'], dtype=float) * self.e_scale
        # np.interp clamps (constant-extrapolates) outside the stored grid;
        # silently doing that produces flat, wrong P(T) tails if the
        # summation was invoked with a wider temp range than the ED step.
        t_lo, t_hi = float(np.min(Tc)), float(np.max(Tc))
        if (self.temp_values.min() < t_lo * (1 - 1e-9)
                or self.temp_values.max() > t_hi * (1 + 1e-9)):
            print(f"  WARNING: cluster {cluster_id}: summation temperature "
                  f"grid [{self.temp_values.min():.4g}, "
                  f"{self.temp_values.max():.4g}] extends beyond the stored "
                  f"P(T) grid [{t_lo:.4g}, {t_hi:.4g}]; values outside are "
                  f"CLAMPED (flat) and wrong. Rerun the ED step with a "
                  f"matching --temp_min/--temp_max.")
        order = np.argsort(Tc)
        def _itp(y):
            return np.interp(self.temp_values, Tc[order],
                             np.asarray(y, dtype=float)[order])
        e = _itp(pt['energy']) * self.e_scale
        c = _itp(pt['specific_heat']); s = _itp(pt['entropy'])
        if self.SI:
            R = 6.02214076e23 * 1.380649e-23
            e, c, s = e * R, c * R, s * R
        return {'energy': e, 'specific_heat': c, 'entropy': s}

    def _cluster_thermo_errors(self, cluster_id):
        """Per-cluster stochastic standard errors on ``self.temp_values``.

        Exact (full-spectrum) clusters return None; OFTLM clusters return
        the independent-seed resampling errors interpolated onto the
        summation grid (SI-scaled to match the quantities).
        """
        pt = self.clusters[cluster_id].get('thermo_pt')
        if pt is None or 'std_error_specific_heat' not in pt:
            return None
        Tc = np.asarray(pt['temperatures'], dtype=float) * self.e_scale
        order = np.argsort(Tc)
        out = {}
        for prop in ('energy', 'specific_heat', 'entropy'):
            key = f'std_error_{prop}'
            if key not in pt:
                return None
            err = np.interp(self.temp_values, Tc[order],
                            np.asarray(pt[key], dtype=float)[order])
            if prop == 'energy':
                err = err * self.e_scale
            if self.SI:
                err = err * (6.02214076e23 * 1.380649e-23)
            out[prop] = err
        return out

    def calculate_thermodynamic_quantities(self, eigenvalues):
        """
        Calculate thermodynamic quantities from eigenvalues.
        Uses a numerically stable approach to handle large beta values (low temperatures).

        Returns:
            Dictionary with 'energy', 'specific_heat', and 'entropy' as keys
        """
        # energy_unit=meV: scale eigenvalues onto the Kelvin grid (1.0 for
        # the default dimensionless/K convention -- see __init__).
        if self.e_scale != 1.0:
            eigenvalues = np.asarray(eigenvalues, dtype=float) * self.e_scale
        ground_state_energy = np.min(eigenvalues)
        shifted = eigenvalues - ground_state_energy

        T = self.temp_values
        # Use T_safe to avoid division by zero for very low T points;
        # those results are overridden by the ground-state limit below.
        T_safe = np.maximum(T, 1e-10)

        # exp_terms[j, i] = exp(-shifted[j] / T[i])   shape: (n_eigs, n_temps)
        exp_terms = np.exp(-shifted[:, np.newaxis] / T_safe[np.newaxis, :])
        Z = exp_terms.sum(axis=0)                                          # (n_temps,)

        energy = (eigenvalues[:, np.newaxis] * exp_terms).sum(axis=0) / Z
        energy_sq = ((eigenvalues**2)[:, np.newaxis] * exp_terms).sum(axis=0) / Z
        specific_heat = (energy_sq - energy**2) / T_safe**2
        entropy = np.log(Z) + (energy - ground_state_energy) / T_safe

        # T→0 limit: ground-state energy, Cv→0, S→0 (third law)
        low_T = T < 1e-5
        energy = np.where(low_T, ground_state_energy, energy)
        specific_heat = np.where(low_T, 0.0, specific_heat)
        entropy = np.where(low_T, 0.0, entropy)

        if self.SI:
            R = 6.02214076e23 * 1.380649e-23  # ≈ 8.314 J/(mol·K)
            specific_heat = specific_heat * R  # dimensionless → J/(mol·K)
            entropy = entropy * R              # dimensionless → J/(mol·K)
            energy = energy * R               # K → J/mol

        return {
            'energy': energy,
            'specific_heat': specific_heat,
            'entropy': entropy,
        }
    
    def read_spin_expectations(self, spin_exp_dir):
        """
        Read spin expectation values from files in the specified directory.
        
        Args:
            spin_exp_dir: Directory containing spin expectation files
        
        Returns:
            Dictionary mapping temperature to spin expectation values
        """

        spin_expectations = {
            'sp': np.zeros_like(self.temp_values),
            'sm': np.zeros_like(self.temp_values),
            'sz': np.zeros_like(self.temp_values)
        }
        # Find all spin expectation files
        pattern = os.path.join(spin_exp_dir, "spin_expectations_T*.dat")
        spin_files = glob.glob(pattern)
        
        if not spin_files:
            print(f"No spin expectation files found in {spin_exp_dir}")
            return
            
        print(f"Found {len(spin_files)} spin expectation files")
        
        # Process each file
        for file_path in spin_files:
            # Extract temperature from filename
            match = re.search(r'T(\d+\.\d+)\.dat', file_path)
            if not match:
                print(f"Could not extract temperature from filename: {file_path}")
                continue
                
            temperature = float(match.group(1))
            print(f"Processing spin expectations for T = {temperature}")
            
            # Read file data
            try:
                data = np.loadtxt(file_path, skiprows=1)
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            # Extract site indices and spin expectations
            sites = data[:, 0].astype(int)
            sp_real = np.mean(data[:, 1])
            sp_imag = np.mean(data[:, 2])
            sm_real = np.mean(data[:, 3])
            sm_imag = np.mean(data[:, 4])
            sz_real = np.mean(data[:, 5])
            sz_imag = np.mean(data[:, 6])

            # Find the index corresponding to the temperature
            temp_index = np.argmin(np.abs(self.temp_values - temperature))
            if temp_index >= len(spin_expectations['sp']):
                print(f"Warning: Temperature index {temp_index} out of bounds for spin expectations")
                continue
            # Store the spin expectation values
            
            # Store as complex numbers in a dictionary
            spin_expectations['sp'][temp_index] = sp_real + 1j * sp_imag
            spin_expectations['sm'][temp_index] = sm_real + 1j * sm_imag
            spin_expectations['sz'][temp_index] = sz_real + 1j * sz_imag
            
        return spin_expectations


    def read_subcluster_info(self):
        """Read subcluster information from the provided file."""

        self.subcluster_info = {}
        
        subcluster_file = 'subclusters_info.txt'  # Default subcluster info file

        # Check if file exists in the specified directory
        filepath = os.path.join(self.cluster_dir, subcluster_file)
        print(f"Looking for subcluster info file at: {filepath}")
        if not os.path.exists(filepath):
            filepath = subcluster_file  # Try relative path
            if not os.path.exists(filepath):
                print(f"Warning: Subcluster info file not found at {subcluster_file}")
                return
                
        with open(filepath, 'r') as f:
            current_cluster = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith('Cluster'):
                    # Parse cluster header: "Cluster X (Order Y, L(c) = ...):" or "Cluster X (Order Y):"
                    match = re.match(r'Cluster (\d+) \(Order (\d+)', line)
                    if match:
                        current_cluster = int(match.group(1))
                        self.subcluster_info[current_cluster] = {'subclusters': {}}
                        
                elif 'No subclusters' in line:
                    # This cluster has no subclusters
                    continue
                        
                elif 'Subclusters:' in line:
                    if current_cluster is None:
                        print(f"Warning: Found Subclusters line without cluster header: {line}")
                        continue
                    # Parse subclusters: "(1, 2), (3, 4), ..."
                    subclusters_str = line.split('Subclusters:')[-1].strip()
                    if not subclusters_str:
                        continue
                        
                    # Extract all (id, multiplicity) pairs
                    pairs = re.findall(r'\((\d+),\s*(\d+)\)', subclusters_str)
                    for subcluster_id, multiplicity in pairs:
                        self.subcluster_info[current_cluster]['subclusters'][int(subcluster_id)] = int(multiplicity)
    
    def get_subclusters(self, cluster_id):
        """
        Get all subclusters of a given cluster with their multiplicities.
        
        Returns:
            Dictionary mapping subcluster_id to its multiplicity in this cluster
        """
        # Explicit subcluster multiplicities (Y_cs) are mandatory for a
        # correct NLCE sum: there is no sound way to reconstruct them from
        # cluster orders alone (same-order subclusters, repeated topologies,
        # and non-unit multiplicities are all lost). We therefore refuse to
        # guess -- a missing/incomplete subclusters_info.txt is a hard error
        # rather than a silently wrong answer.
        if not getattr(self, 'subcluster_info', None):
            raise RuntimeError(
                "Subcluster information is unavailable. Generate "
                "'subclusters_info.txt' (the cluster-prep step writes it into "
                f"the cluster directory '{self.cluster_dir}') before summing; "
                "NLCE weights cannot be computed without explicit Y_cs "
                "multiplicities."
            )

        if cluster_id in self.subcluster_info:
            return self.subcluster_info[cluster_id]['subclusters']

        # The file was loaded but this cluster is absent. Order-1 clusters
        # legitimately have no subclusters; anything larger means the file is
        # incomplete and the recursion would be wrong.
        if self.clusters[cluster_id]['order'] <= 1:
            return {}
        raise RuntimeError(
            f"Cluster {cluster_id} (order {self.clusters[cluster_id]['order']}) "
            "is missing from subclusters_info.txt; its NLCE weight cannot be "
            "computed. Regenerate the subcluster table for this cluster set."
        )
    
    def _topological_sort_clusters(self):
        """
        Sort clusters in dependency order using topological sort (Kahn's algorithm).

        A cluster must be processed AFTER all its subclusters, regardless of order.
        This handles cases where a cluster of order N contains subclusters of the same order N.

        Returns:
            List of (cluster_id, order) tuples in processing order
        """
        import heapq

        deps = {}
        for cluster_id in self.clusters:
            subclusters = self.get_subclusters(cluster_id)
            deps[cluster_id] = set(subclusters.keys())

        # Reverse adjacency built ONCE: dependents[s] = clusters that
        # contain s as a subcluster. Each dependency edge is then visited
        # exactly once below (true Kahn's algorithm), instead of scanning
        # every cluster's dep-set on every pop -- that scan was O(n^2)
        # over the cluster count and dominated at order-8 cluster counts.
        dependents: dict = {cid: [] for cid in self.clusters}
        for cid, dep_set in deps.items():
            for s in dep_set:
                if s in dependents:  # tolerate dangling subcluster ids
                    dependents[s].append(cid)

        in_degree = {cid: len(d) for cid, d in deps.items()}
        # Min-heap keyed on (order, cid): pops in deterministic
        # (order, id) sequence without re-sorting the queue every pop.
        heap = [(self.clusters[cid]['order'], cid)
                for cid, deg in in_degree.items() if deg == 0]
        heapq.heapify(heap)
        processed = set()
        sorted_clusters = []

        while heap:
            order, cluster_id = heapq.heappop(heap)
            sorted_clusters.append((cluster_id, order))
            processed.add(cluster_id)

            for cid in dependents[cluster_id]:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    heapq.heappush(heap, (self.clusters[cid]['order'], cid))

        if len(sorted_clusters) != len(self.clusters):
            remaining = set(self.clusters.keys()) - processed
            # Almost always DANGLING subcluster ids (a subclusters_info.txt
            # entry referencing a cluster with no data on disk -- e.g. a
            # partially deleted cluster set), not a genuine cycle. Their
            # weights will be SKIPPED downstream, silently truncating every
            # branch of the sum that depends on them -- say so loudly.
            dangling = set()
            for cid in remaining:
                dangling |= {s for s in deps.get(cid, ())
                             if s not in self.clusters}
            print(f"WARNING: {len(remaining)} clusters have unsatisfiable "
                  f"dependencies and CANNOT get valid NLCE weights: "
                  f"{sorted(remaining)}")
            if dangling:
                print(f"WARNING: root cause -- subcluster ids referenced but "
                      f"absent from the cluster set: {sorted(dangling)}. "
                      f"The cluster directory is INCOMPLETE (partial "
                      f"deletion / stale subclusters_info.txt); regenerate "
                      f"it. The NLCE sum below silently EXCLUDES every "
                      f"affected branch.")
            else:
                print("WARNING: genuine dependency cycle (should be "
                      "impossible for subcluster relations) -- inspect "
                      "subclusters_info.txt.")
            print("Falling back to order-based sort for remaining clusters.")
            for cid in sorted(remaining, key=lambda c: (self.clusters[c]['order'], c)):
                sorted_clusters.append((cid, self.clusters[cid]['order']))

        return sorted_clusters
    

    def print_subcluster_info(self):
        """Print out the subcluster information for all clusters."""
        # Read subcluster information if not already loaded
        if not hasattr(self, 'subcluster_info'):
            self.read_subcluster_info()
            
        if not hasattr(self, 'subcluster_info') or not self.subcluster_info:
            print("No subcluster information available.")
            return
            
        print("Subcluster Information:")
        print("----------------------")
        
        for cluster_id, info in sorted(self.subcluster_info.items()):
            print(f"Cluster {cluster_id}:")
            if 'subclusters' in info and info['subclusters']:
                print("  Subclusters (ID, Multiplicity):")
                for subcluster_id, multiplicity in sorted(info['subclusters'].items()):
                    print(f"    {subcluster_id}: {multiplicity}")
            else:
                print("  No subclusters")
            print()
    
    def calculate_weights(self):
        """Calculate weights for all clusters using the NLC principle."""
        # Read subcluster information if not already loaded
        if not hasattr(self, 'subcluster_info'):
            self.read_subcluster_info()

        self.print_subcluster_info()
            
        # Sort clusters in dependency order (handles same-order subclusters)
        sorted_clusters = self._topological_sort_clusters()
        
        print(f"\nProcessing {len(sorted_clusters)} clusters in dependency order")
        
        # Initialize weights dictionary
        if self.measure_spin:
            self.weights = {
                'energy': {},
                'specific_heat': {},
                'entropy': {},
                'sp': {},
                'sm': {},
                'sz': {}
            }
        else:
            self.weights = {
                'energy': {},
                'specific_heat': {},
                'entropy': {}   
            }
        
        # Track which clusters have valid weights (all required subclusters available)
        self.valid_weights = set()

        # Squared stochastic errors of the weights, propagated in
        # quadrature through the subtraction:
        #   err2_W(c) = err2_P(c) + sum_s Y_cs^2 * err2_W(s)
        # Exact clusters contribute err2_P = 0 but still inherit their
        # stochastic subclusters' variance.
        self.weight_errs2 = {p: {} for p in ('energy', 'specific_heat', 'entropy')}
        
        # Calculate weights for each cluster
        for cluster_id, order in sorted_clusters:
            # OFTLM (large) clusters have no eigenvalue spectrum but DO carry
            # pre-computed P(T) in 'thermo_pt' -- only skip if neither is present.
            if (self.clusters[cluster_id]['eigenvalues'] is None
                    and 'thermo_pt' not in self.clusters[cluster_id]):
                print(f"  Cluster {cluster_id} (order {order}): SKIPPED (no data)")
                continue
            
            # Get subclusters with their multiplicities
            subclusters = self.get_subclusters(cluster_id)
            
            # Check if ALL required subclusters have valid weights
            # This is CRITICAL for correct recursive weight calculation
            missing_subclusters = []
            for sub_id in subclusters.keys():
                if sub_id not in self.valid_weights:
                    missing_subclusters.append(sub_id)
            
            if missing_subclusters:
                print(f"  Cluster {cluster_id} (order {order}): SKIPPED - missing weights for subclusters {missing_subclusters}")
                print(f"    Cannot compute weight because W(c) = P(c) - Σ Y_cs × W(s) requires ALL subcluster weights")
                continue
                
            # Get thermodynamic quantities for this cluster (dense spectrum or
            # OFTLM P(T) for large clusters).
            quantities = self._cluster_thermo_quantities(cluster_id)
            errors = self._cluster_thermo_errors(cluster_id)

            # Calculate weights for energy and specific heat
            for prop in ['energy', 'specific_heat', 'entropy']:
                # Property of the cluster
                property_value = quantities[prop]
                err2 = (np.square(errors[prop]) if errors is not None
                        else np.zeros_like(self.temp_values))
                # Subtract contributions from all subclusters with correct multiplicities
                for subcluster_id, multiplicity in subclusters.items():
                    if subcluster_id in self.weights[prop]:
                        property_value = property_value - \
                            self.weights[prop][subcluster_id] * multiplicity
                        sub_err2 = self.weight_errs2[prop].get(subcluster_id)
                        if sub_err2 is not None:
                            err2 = err2 + (multiplicity ** 2) * sub_err2
                # Store the weight (+ its propagated squared error when nonzero)
                self.weights[prop][cluster_id] = property_value
                if np.any(err2 > 0):
                    self.weight_errs2[prop][cluster_id] = err2

            # Stochastic-noise amplification diagnostic. The weight
            # W = P - sum(Y*W_sub) inherits the ABSOLUTE noise of the
            # stochastic P(T) inputs; when the subtraction cancels strongly
            # the contribution is noise-dominated -- the classic silent
            # failure mode of stochastic solvers at high NLCE order.
            cv_err2 = self.weight_errs2['specific_heat'].get(cluster_id)
            if cv_err2 is not None:
                w_scale = float(np.max(np.abs(
                    self.weights['specific_heat'][cluster_id])))
                e_scale = float(np.max(np.sqrt(cv_err2)))
                if w_scale > 0 and e_scale > 0.5 * w_scale:
                    print(f"  WARNING: cluster {cluster_id} (order {order}): "
                          f"propagated stochastic error on W_Cv "
                          f"(max sigma = {e_scale:.3g}) is "
                          f"{e_scale/w_scale:.1%} of max|W_Cv| = {w_scale:.3g} "
                          f"-- this cluster's NLCE contribution is "
                          f"noise-dominated. Raise --oftlm_num_samples "
                          f"(error ~ 1/sqrt(R)) or --oftlm_cutoff to solve "
                          f"it exactly.")
            elif self.clusters[cluster_id].get('is_stochastic'):
                # Legacy OFTLM file without std_error datasets: fall back
                # to the cancellation-ratio heuristic.
                p_scale = float(np.max(np.abs(quantities['specific_heat'])))
                w_scale = float(np.max(np.abs(
                    self.weights['specific_heat'][cluster_id])))
                if p_scale > 0 and w_scale < 0.05 * p_scale:
                    amp = p_scale / max(w_scale, 1e-300)
                    print(f"  WARNING: cluster {cluster_id} (order {order}, "
                          f"OFTLM, no std_error data): weight cancellation "
                          f"amplifies the stochastic sample noise ~{amp:.0f}x "
                          f"(max|W_Cv|/max|P_Cv| = {w_scale/p_scale:.3g}). "
                          f"Re-run the ED step with --oftlm_num_seeds >= 2 "
                          f"for a quantitative error band.")

            # Mark this cluster as having valid weights for thermodynamic properties
            self.valid_weights.add(cluster_id)


            if self.measure_spin:
                # Read spin expectation values if needed
                spin_exp_dir = os.path.join(self.eigenvalue_dir, f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}/output/spin_expectations")
                quantities_spin =  self.read_spin_expectations(spin_exp_dir)
                
                print(f"Spin expectation values for cluster {cluster_id}: {quantities_spin}")

                for prop in ['sp', 'sm', 'sz']:
                    # Get the spin expectation value
                    spin_value = quantities_spin[prop]
                    # Subtract contributions from all subclusters with correct multiplicities
                    for subcluster_id, multiplicity in subclusters.items():
                        if subcluster_id in self.weights[prop]:
                            spin_value -= self.weights[prop][subcluster_id] * multiplicity
                    # Store the weight
                    self.weights[prop][cluster_id] = spin_value
            



    def euler_resummation(self, partial_sums, l=3):
        """
        Apply Euler transformation for series acceleration.
        
        Implementation follows Tang–Khatami–Rigol (arXiv:1207.3366):
        Keep first (n-l) bare terms, apply Euler transform to the tail using
        forward differences with 2^{-(k+1)} weights.
        
        Args:
            partial_sums: Array of partial sums S_n = sum_{k=0}^n a_k
            l: Number of highest-order terms to keep "as is" (default: 3)
            
        Returns:
            Accelerated series approximation
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else 0.0
        
        # Convert partial sums to increments (the a_n terms)
        increments = [partial_sums[0]]
        for i in range(1, n):
            increments.append(partial_sums[i] - partial_sums[i-1])
        
        # Choose l adaptively if sequence is short
        l_use = min(l, max(2, n - 3))
        
        if n <= l_use:
            # Not enough terms for Euler, return last partial sum
            return partial_sums[-1]
        
        # Keep first (n - l_use) bare terms as-is
        bare_sum = partial_sums[n - l_use - 1] if n - l_use - 1 >= 0 else (
            np.zeros_like(partial_sums[-1]) if len(partial_sums.shape) > 1 
            else 0.0
        )
        
        # Apply Euler transform to the last l_use increments
        tail_increments = increments[n - l_use:]
        
        # Build forward difference triangle
        diff_triangle = [tail_increments]
        for k in range(len(tail_increments) - 1):
            prev_row = diff_triangle[-1]
            next_row = [prev_row[i+1] - prev_row[i] for i in range(len(prev_row) - 1)]
            if len(next_row) == 0:
                break
            diff_triangle.append(next_row)
        
        # Accumulate Euler tail: sum_k Δ^k a_{n-l}/2^{k+1}
        euler_tail = np.zeros_like(partial_sums[-1]) if len(partial_sums.shape) > 1 else 0.0
        for k, diff_row in enumerate(diff_triangle):
            if len(diff_row) > 0:
                euler_tail += diff_row[0] / (2**(k+1))
        
        return bare_sum + euler_tail
    
    def wynn_epsilon(self, sequence):
        """Wynn epsilon algorithm for series acceleration.

        sequence: list or 2-D array; each element may be a scalar or a
        1-D array of length n_temps.  Element-wise sentinel handling ensures
        no blow-up in one temperature contaminates others.
        """
        n = len(sequence)
        if n < 3:
            return sequence[-1] if n > 0 else 0.0

        seq = [np.asarray(s, dtype=float) for s in sequence]
        SENTINEL = 1e50
        is_array = seq[0].ndim > 0
        if is_array:
            shape = seq[0].shape
            eps = np.zeros((n + 1, n) + shape)
            for i in range(n):
                eps[1, i] = seq[i]
            for j in range(2, n + 1):
                for i in range(n - j + 1):
                    diff = eps[j-1, i+1] - eps[j-1, i]
                    result = np.full(shape, SENTINEL)
                    ok = np.abs(diff) > 1e-15 * (1.0 + np.abs(eps[j-1, i]))
                    if np.any(ok):
                        result[ok] = eps[j-2, i+1][ok] + 1.0 / diff[ok]
                    eps[j, i] = result
            best = seq[-1].copy()
            for j in range(3, n + 1, 2):
                ok = np.abs(eps[j, 0]) < SENTINEL * 0.1
                best[ok] = eps[j, 0][ok]
            return best
        else:
            eps_s = np.zeros((n + 1, n))
            for i in range(n):
                eps_s[1, i] = seq[i].item() if seq[i].ndim == 0 else seq[i]
            for j in range(2, n + 1):
                for i in range(n - j + 1):
                    diff = eps_s[j-1, i+1] - eps_s[j-1, i]
                    if abs(diff) < 1e-15 * (1.0 + abs(eps_s[j-1, i])):
                        eps_s[j, i] = SENTINEL
                    else:
                        eps_s[j, i] = eps_s[j-2, i+1] + 1.0 / diff
            best_s = float(seq[-1])
            for j in range(3, n + 1, 2):
                if abs(eps_s[j, 0]) < SENTINEL * 0.1:
                    best_s = eps_s[j, 0]
            return best_s

    def brezinski_theta(self, partial_sums):
        """Brezinski theta algorithm for series acceleration.

        Iteratively applies the Wynn-epsilon recurrence and collects even-column
        estimates (θ₂, θ₄, …), returning the last stable one.  Falls back to
        the last partial sum if no even column converges.

        partial_sums: list or 2-D array; each element may be a scalar or a
        1-D array of length n_temps.
        """
        n = len(partial_sums)
        if n < 3:
            return partial_sums[-1] if n > 0 else np.zeros_like(self.temp_values)
        seq_arr = np.array(partial_sums, dtype=float)
        seq_range = np.max(np.abs(seq_arr), axis=0) + 1e-10
        max_allowed = 100 * seq_range
        eps_den = 1e-14
        theta_prev = np.zeros_like(seq_arr)
        theta_curr = seq_arr.copy()
        evens = []
        for iteration in range(1, n):
            n_curr = theta_curr.shape[0]
            if n_curr < 2:
                break
            theta_next = np.zeros((n_curr - 1,) + theta_curr.shape[1:])
            for i in range(n_curr - 1):
                denom = theta_curr[i+1] - theta_curr[i]
                mask_small = np.abs(denom) < eps_den
                raw = np.where(mask_small, theta_curr[i+1],
                               theta_prev[i+1] + 1.0 / np.where(mask_small, 1.0, denom))
                theta_next[i] = np.where(np.abs(raw) > max_allowed, theta_curr[i+1], raw)
            theta_prev = theta_curr
            theta_curr = theta_next
            if iteration % 2 == 0 and theta_curr.shape[0] > 0:
                candidate = theta_curr[0].copy()
                if not np.any(np.abs(candidate) > max_allowed):
                    evens.append(candidate)
        if len(evens) >= 1:
            result = evens[-1]
            return result.item() if result.ndim == 0 else result
        last = np.asarray(partial_sums[-1])
        return last.item() if last.ndim == 0 else last.copy()

    def pade_approximant(self, coefficients, m=None, n=None):
        """
        Construct Padé approximant [m/n] from series coefficients.
        
        Args:
            coefficients: Series coefficients a_k
            m: Degree of numerator (default: len(coeffs)//2)
            n: Degree of denominator (default: len(coeffs)//2)
            
        Returns:
            Function representing the Padé approximant
        """
        L = len(coefficients)
        if m is None:
            m = L // 2
        if n is None:
            n = L - m - 1
            
        if m + n + 1 > L:
            m = L // 2
            n = L - m - 1
            
        # Set up linear system for denominator coefficients
        # c_k + sum_{j=1}^n b_j * c_{k-j} = 0 for k = m+1, ..., m+n
        if n > 0:
            A = np.zeros((n, n))
            b_vec = np.zeros(n)
            
            for i in range(n):
                k = m + 1 + i
                if k < L:
                    b_vec[i] = -coefficients[k]
                    for j in range(n):
                        if k - 1 - j >= 0 and k - 1 - j < L:
                            A[i, j] = coefficients[k - 1 - j]
                            
            # Solve for denominator coefficients
            try:
                q_coeffs = np.linalg.solve(A, b_vec)
                q_coeffs = np.concatenate([[1.0], q_coeffs])
            except np.linalg.LinAlgError:
                q_coeffs = np.array([1.0])
                n = 0
        else:
            q_coeffs = np.array([1.0])
            
        # Calculate numerator coefficients
        p_coeffs = np.zeros(m + 1)
        for k in range(min(m + 1, L)):
            p_coeffs[k] = coefficients[k]
            for j in range(1, min(k + 1, len(q_coeffs))):
                if k - j >= 0 and k - j < L:
                    p_coeffs[k] += q_coeffs[j] * coefficients[k - j]
                    
        return p_coeffs, q_coeffs
    
    def evaluate_pade(self, p_coeffs, q_coeffs, x):
        """Evaluate Padé approximant at point x."""
        p_val = np.polyval(p_coeffs[::-1], x)
        q_val = np.polyval(q_coeffs[::-1], x)
        
        # Avoid division by zero
        mask = np.abs(q_val) > 1e-15
        result = np.zeros_like(x)
        result[mask] = p_val[mask] / q_val[mask]
        result[~mask] = p_val[~mask]  # Use polynomial approximation when denominator is small
        
        return result
    
    def shanks_transform(self, sequence):
        """
        Apply Shanks transformation for series acceleration.
        
        Args:
            sequence: Array of partial sums
            
        Returns:
            Accelerated approximation
        """
        n = len(sequence)
        if n < 3:
            return sequence[-1] if n > 0 else 0.0
            
        # Shanks transformation: S' = (S_{n+1}*S_{n-1} - S_n^2) / (S_{n+1} - 2*S_n + S_{n-1})
        s_prev = sequence[-3]
        s_curr = sequence[-2] 
        s_next = sequence[-1]
        
        denominator = s_next - 2*s_curr + s_prev
        
        # Avoid division by zero
        if np.any(np.abs(denominator) < 1e-15):
            return s_next
        else:
            return (s_next * s_prev - s_curr**2) / denominator
    
    def aitken_delta2(self, sequence):
        """
        Apply Aitken's Δ² method for series acceleration.
        
        Args:
            sequence: Array of partial sums
            
        Returns:
            Accelerated approximation
        """
        return self.shanks_transform(sequence)  # Aitken's Δ² is equivalent to Shanks
    
    def analyze_convergence(self, partial_sums):
        """
        Analyze convergence properties of the series.
        
        Args:
            partial_sums: Array of partial sums
            
        Returns:
            Dictionary with convergence metrics
        """
        n = len(partial_sums)
        if n < 3:
            return {'converged': False, 'oscillatory': False, 'ratio_test': None}
            
        # Calculate differences
        diffs = np.diff(partial_sums, axis=0)
        
        # Check for convergence (differences getting smaller)
        if n > 3:
            recent_diffs = np.abs(diffs[-3:])
            converged = np.all(recent_diffs[-1] <= recent_diffs[-2]) and np.all(recent_diffs[-2] <= recent_diffs[-3])
        else:
            converged = False
            
        # Check for oscillatory behavior
        if n > 4:
            signs = np.sign(diffs[-4:])
            oscillatory = np.all(signs[1:] != signs[:-1])  # Alternating signs
        else:
            oscillatory = False
            
        # Ratio test for convergence
        if n > 2:
            ratios = np.abs(diffs[1:] / diffs[:-1])
            avg_ratio = np.mean(ratios[-min(3, len(ratios)):])  # Average of last few ratios
        else:
            avg_ratio = None
            
        return {
            'converged': converged,
            'oscillatory': oscillatory, 
            'ratio_test': avg_ratio
        }
    
    def select_resummation_method(self, partial_sums):
        """
        Automatically select the best resummation method based on convergence analysis.
        
        Args:
            partial_sums: Array of partial sums for analysis
            
        Returns:
            String indicating the selected method
        """
        convergence = self.analyze_convergence(partial_sums)
        n = len(partial_sums)
        
        # Count the actual number of distinct partial sums (non-zero increments)
        # This is the true number of NLCE terms
        # Note: partial_sums[0] is often 0 (order 0 placeholder), so we count non-zero increments
        if len(partial_sums.shape) > 1:
            increments = np.diff(partial_sums[:, 0])
        else:
            increments = np.diff(partial_sums)
        actual_terms = np.sum(np.abs(increments) > 1e-15)  # Number of actual NLCE orders with contributions
        
        # For very few terms, resummation is unreliable - use direct summation
        # Need at least 4 actual NLCE orders for reliable Shanks
        if actual_terms <= 3:
            print(f"  → Using DIRECT summation (only {actual_terms} NLCE orders)")
            return 'direct'
        
        # If already converged and not oscillatory, use direct sum
        if convergence['converged'] and not convergence['oscillatory']:
            return 'direct'
            
        # For oscillatory series, Euler works well
        if convergence['oscillatory']:
            return 'euler'

        # For slowly converging series, try Wynn's epsilon
        if convergence['ratio_test'] is not None and convergence['ratio_test'] > 0.5:
            if actual_terms >= 5:  # Wynn needs at least 5 terms for stability
                return 'wynn'
            else:
                return 'euler'
                

            
        # Default to Wynn's epsilon for general case
        return 'wynn'
    
    def apply_resummation(self, partial_sums, method='auto'):
        """
        Apply the specified resummation method.
        
        Args:
            partial_sums: Array of partial sums
            method: Resummation method ('auto', 'direct', 'euler', 'wynn', 'shanks', 'pade')
            
        Returns:
            Accelerated series approximation
        """
        if len(partial_sums) == 0:
            return np.zeros_like(self.temp_values)
            
        if method == 'auto':
            method = self.select_resummation_method(partial_sums)
            
        print(f"Using resummation method: {method}")
        
        # Normalise aliases so callers can use the same names as other kernels
        _ALIAS = {'none': 'direct', 'theta': 'brezinski', 'wynn_multi': 'wynn',
                  'entropy_derived': 'euler'}
        method = _ALIAS.get(method, method)

        if method == 'direct':
            return partial_sums[-1]
        elif method == 'euler':
            return self.euler_resummation(partial_sums)
        elif method == 'wynn':
            return self.wynn_epsilon(partial_sums)
        elif method == 'shanks':
            return self.shanks_transform(partial_sums)
        elif method == 'aitken':
            return self.aitken_delta2(partial_sums)
        elif method == 'brezinski':
            return self.brezinski_theta(partial_sums)
        elif method == 'pade':
            # WARNING: Padé approximants are designed for power series in a variable x.
            # NLCE partial sums are NOT power series coefficients - they are cumulative
            # sums over cluster contributions. Using Padé here may give inconsistent results.
            # Consider using 'euler', 'wynn', or 'shanks' instead.
            print("WARNING: Padé method is not well-suited for NLCE summation. Results may be unreliable.")
            print("         Consider using --resummation_method euler/wynn/shanks instead.")
            # For Padé, we need series coefficients, not partial sums
            # Convert partial sums to coefficients
            coeffs = np.diff(np.concatenate([[np.zeros_like(partial_sums[0])], partial_sums]), axis=0)
            p_coeffs, q_coeffs = self.pade_approximant(coeffs)
            # Evaluate at x=1 (summing the series)
            return self.evaluate_pade(p_coeffs, q_coeffs, np.ones_like(self.temp_values))
        else:
            print(f"Unknown method {method}, using direct summation")
            return partial_sums[-1]
    
    def sum_nlc(self, resummation_method='auto', order_cutoff=None, fix_negative_cv=False):
        """
        Perform the NLC summation with automatic resummation method selection.
        
        For Lanczos-boosted NLCE:
        - At low T (< T_max_valid): use all orders including truncated clusters
        - At high T (> T_max_valid): only use orders with full spectra
        - This ensures resummation operates on valid data at all temperatures
        
        Args:
            resummation_method: Method for series acceleration ('auto', 'direct', 'euler', 'wynn', 'shanks', 'pade')
            order_cutoff: Maximum order to include in the summation
            fix_negative_cv: If True, replace negative specific heat with derivative-based estimate.
                            This is a workaround for resummation artifacts. Default False.
            
        Returns:
            Dictionary with summed properties
        """
        
        # Determine which orders have truncated (Lanczos-boosted) clusters
        truncated_orders = set()
        if self.lanczos_boost_active:
            for cluster_id in self.truncated_clusters:
                if cluster_id in self.clusters:
                    truncated_orders.add(self.clusters[cluster_id]['order'])
            
            if truncated_orders:
                max_full_order = min(truncated_orders) - 1
                print(f"\nLanczos-boost temperature-adaptive summation:")
                print(f"  - Orders with full spectrum: 1-{max_full_order}")
                print(f"  - Orders with truncated spectrum: {sorted(truncated_orders)}")
                print(f"  - Low-T (T < {self.lanczos_boost_temp_max:.4f}): use all orders")
                print(f"  - High-T (T >= {self.lanczos_boost_temp_max:.4f}): use orders 1-{max_full_order} only")

        if self.measure_spin:
            # Initialize results for spin expectation values
            results = {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
                'entropy': np.zeros_like(self.temp_values),
                'sp': np.zeros_like(self.temp_values),
                'sm': np.zeros_like(self.temp_values),
                'sz': np.zeros_like(self.temp_values)
            }
            properties = ['energy', 'specific_heat', 'entropy', 'sp', 'sm', 'sz']
        else:
            results = {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
                'entropy': np.zeros_like(self.temp_values)
            }
            properties = ['energy', 'specific_heat', 'entropy']
    
        # Calculate the NLC sum for each property
        for prop in properties:
            if prop not in self.weights:
                continue
                
            print(f"Processing property: {prop}")
            
            # For Lanczos-boost: compute two sums and blend them
            if self.lanczos_boost_active and truncated_orders and self.lanczos_boost_temp_max is not None:
                # Sum by order, tracking full-spectrum vs truncated contributions separately
                sum_by_order_full = defaultdict(lambda: np.zeros_like(self.temp_values))
                sum_by_order_all = defaultdict(lambda: np.zeros_like(self.temp_values))
                
                for cluster_id, weight in self.weights[prop].items():
                    order = self.clusters[cluster_id]['order']
                    if order_cutoff is not None and order > order_cutoff:
                        continue
                    
                    contribution = weight * self.clusters[cluster_id]['multiplicity']
                    sum_by_order_all[order] += contribution
                    
                    # Only include in full-spectrum sum if not a truncated cluster
                    if cluster_id not in self.truncated_clusters:
                        sum_by_order_full[order] += contribution
                
                # Compute partial sums for both
                if sum_by_order_all:
                    max_order = max(sum_by_order_all.keys())
                    
                    # Full-spectrum partial sums (valid all T)
                    partial_sums_full = []
                    cumsum_full = np.zeros_like(self.temp_values)
                    
                    # All-orders partial sums (valid low T only)
                    partial_sums_all = []
                    cumsum_all = np.zeros_like(self.temp_values)
                    
                    for order in range(max_order + 1):
                        if order in sum_by_order_full:
                            cumsum_full += sum_by_order_full[order]
                        if order in sum_by_order_all:
                            cumsum_all += sum_by_order_all[order]
                        partial_sums_full.append(cumsum_full.copy())
                        partial_sums_all.append(cumsum_all.copy())
                    
                    partial_sums_full = np.array(partial_sums_full)
                    partial_sums_all = np.array(partial_sums_all)
                    
                    # Apply resummation to both
                    result_full = self.apply_resummation(partial_sums_full, method=resummation_method)
                    result_all = self.apply_resummation(partial_sums_all, method=resummation_method)
                    
                    # Blend: use all-orders for low T, full-spectrum for high T
                    # Smooth transition around T_max_valid
                    T_valid = self.lanczos_boost_temp_max
                    transition_width = T_valid * 0.2  # 20% transition zone
                    
                    # Sigmoid blending: 0 at low T (use all), 1 at high T (use full only)
                    blend_factor = 1.0 / (1.0 + np.exp(-(self.temp_values - T_valid) / transition_width))
                    
                    results[prop] = (1 - blend_factor) * result_all + blend_factor * result_full
                    
                    print(f"  Blended low-T (all orders) and high-T (full spectrum only) results")
                else:
                    print(f"Warning: No weights found for property {prop}")
                    results[prop] = np.zeros_like(self.temp_values)
            else:
                # Standard summation (no Lanczos-boost)
                sum_by_order = defaultdict(lambda: np.zeros_like(self.temp_values))

                for cluster_id, weight in self.weights[prop].items():
                    order = self.clusters[cluster_id]['order']
                    if order_cutoff is not None and order > order_cutoff:
                        continue

                    sum_by_order[order] += weight * self.clusters[cluster_id]['multiplicity']

                # Create array of partial sums
                if sum_by_order:
                    max_order = max(sum_by_order.keys())
                    partial_sums = []
                    cumulative_sum = np.zeros_like(self.temp_values)

                    for order in range(max_order + 1):
                        if order in sum_by_order:
                            cumulative_sum += sum_by_order[order]
                        partial_sums.append(cumulative_sum.copy())

                    partial_sums = np.array(partial_sums)

                    # Apply resummation
                    results[prop] = self.apply_resummation(partial_sums, method=resummation_method)

                    # SOTA convergence bookkeeping (Tang-Khatami-Rigol;
                    # Schaefer et al. PRB 102, 054408): ALWAYS carry both
                    # standard accelerations plus the last two bare orders.
                    # Their mutual agreement region -- not any single
                    # scheme's smoothness -- is the trusted-T criterion,
                    # and the Euler-Wynn spread is the quoted error bar.
                    if prop in ('energy', 'specific_heat', 'entropy'):
                        results[f'{prop}_euler'] = self.apply_resummation(
                            partial_sums, method='euler')
                        results[f'{prop}_wynn'] = self.apply_resummation(
                            partial_sums, method='wynn')
                        results[f'{prop}_bare_last'] = partial_sums[-1].copy()
                        results[f'{prop}_bare_prev'] = (
                            partial_sums[-2].copy() if len(partial_sums) >= 2
                            else partial_sums[-1].copy())
                else:
                    print(f"Warning: No weights found for property {prop}")
                    results[prop] = np.zeros_like(self.temp_values)
        
        # Stochastic error band of the BARE sum: err^2 = sum_c (L_c sigma_Wc)^2
        # over the included clusters (independent seeds per cluster). This is
        # the standard quoted scale for typicality-in-NLCE (Richter/
        # Steinigeweg PRB 99, 144422); resummation reweights the tail by
        # O(1) factors so the bare band is a faithful upper estimate.
        for prop in ('energy', 'specific_heat', 'entropy'):
            errs2 = getattr(self, 'weight_errs2', {}).get(prop)
            if not errs2:
                continue
            tot2 = np.zeros_like(self.temp_values)
            for cluster_id, err2 in errs2.items():
                order = self.clusters[cluster_id]['order']
                if order_cutoff is not None and order > order_cutoff:
                    continue
                mult = self.clusters[cluster_id]['multiplicity']
                tot2 = tot2 + (mult ** 2) * err2
            if np.any(tot2 > 0):
                results[f'{prop}_stderr'] = np.sqrt(tot2)
                print(f"  {prop}: stochastic std-error band "
                      f"max = {float(np.max(results[f'{prop}_stderr'])):.4g} "
                      f"(from {len(errs2)} stochastic-lineage clusters)")

        # ---- Automated trusted-T criterion (standard NLCE practice) ----
        # T*(prop) = lowest temperature above which BOTH
        #   (a) the last two bare orders agree, and
        #   (b) the Euler and Wynn accelerations agree,
        # to within rel_tol (1%). Below T* the expansion has left its
        # convergence region and NO resummed curve should be trusted,
        # however smooth it looks. (Tang-Khatami-Rigol arXiv:1207.3366;
        # Schaefer et al. PRB 102, 054408 quote exactly this criterion.)
        rel_tol = 0.01
        trusted = {}
        for prop in ('energy', 'specific_heat', 'entropy'):
            if f'{prop}_euler' not in results:
                continue
            scale = np.maximum(np.abs(results[f'{prop}_bare_last']), 1e-12)
            dev_orders = np.abs(results[f'{prop}_bare_last']
                                - results[f'{prop}_bare_prev']) / scale
            dev_schemes = np.abs(results[f'{prop}_euler']
                                 - results[f'{prop}_wynn']) / scale
            ok = (dev_orders < rel_tol) & (dev_schemes < rel_tol)
            # NLCE converges from high T downward: find the lowest T such
            # that every grid point above it satisfies both checks.
            t_star = None
            idx = np.argsort(self.temp_values)
            ok_sorted = ok[idx]
            # walk from the top down while converged
            j = len(ok_sorted) - 1
            while j >= 0 and ok_sorted[j]:
                j -= 1
            if j < len(ok_sorted) - 1:
                t_star = float(self.temp_values[idx][j + 1])
            trusted[prop] = t_star
            if t_star is not None:
                print(f"  {prop}: trusted for T >= {t_star:.4g} "
                      f"(bare order-to-order AND Euler-vs-Wynn agree to "
                      f"{rel_tol:.0%} there)")
            else:
                print(f"  {prop}: NO trusted temperature window at "
                      f"{rel_tol:.0%} tolerance -- do not use this curve "
                      f"quantitatively; add orders or loosen the criterion "
                      f"knowingly.")
        results['trusted_T'] = trusted

        # Check for negative specific heat (indicates convergence issues)
        if 'specific_heat' in results:
            negative_cv_temps = self.temp_values[results['specific_heat'] < 0]
            if len(negative_cv_temps) > 0:
                print(f"WARNING: Negative specific heat detected at {len(negative_cv_temps)} temperature points.")
                print(f"         Temperature range with Cv < 0: [{negative_cv_temps.min():.4e}, {negative_cv_temps.max():.4e}]")
                print(f"         This indicates NLCE convergence issues. Consider:")
                print(f"         1. Including more cluster orders (increase --order_cutoff)")
                print(f"         2. Using a different resummation method")
                print(f"         3. Using --fix_negative_cv flag to apply derivative-based fix")
                
                if fix_negative_cv and 'energy' in results:
                    # Calculate specific heat as derivative of energy: Cv = -dE/dT / T^2 = d(βE)/dβ
                    # More precisely: Cv = (1/T^2) * d<E>/dβ where β = 1/T
                    energy_derivative = -np.gradient(results['energy'], self.temp_values) / (self.temp_values**2)
                    
                    # Replace negative Cv with derivative-based estimate
                    mask = results['specific_heat'] < 0
                    results['specific_heat'][mask] = np.maximum(energy_derivative[mask], 0.0)
                    print(f"         Applied derivative-based fix to {np.sum(mask)} points.")
        
        return results
    
    def run(self, resummation_method='auto', order_cutoff=None, fix_negative_cv=False):
        """Run the full NLC calculation."""
        print("Reading cluster information...")
        self.read_clusters()
        
        print("Reading eigenvalues...")
        self.read_eigenvalues()
        
        print("Calculating weights...")
        self.calculate_weights()
        
        print("Performing NLC summation...")
        results = self.sum_nlc(resummation_method, order_cutoff, fix_negative_cv)
        
        # Add Lanczos-boost validity warning to results
        if self.lanczos_boost_active and self.lanczos_boost_temp_max is not None:
            results['lanczos_boost_temp_max'] = self.lanczos_boost_temp_max
            results['truncated_clusters'] = list(self.truncated_clusters.keys())
            
            # Warn about temperatures beyond valid range
            invalid_temps = self.temp_values[self.temp_values > self.lanczos_boost_temp_max]
            if len(invalid_temps) > 0:
                print(f"\n{'!'*70}")
                print(f"WARNING: Lanczos-boost results may be inaccurate for T > {self.lanczos_boost_temp_max:.4f}")
                print(f"         {len(invalid_temps)} of {len(self.temp_values)} temperature points are beyond valid range")
                print(f"         Consider using full ED for accurate high-T results")
                print(f"{'!'*70}\n")

        return results
    
    def compare_resummation_methods(self, order_cutoff=None, save_comparison=False):
        """
        Compare different resummation methods and provide diagnostic information.
        
        Args:
            order_cutoff: Maximum order to include in summation
            save_comparison: If True, save comparison plots and data
            
        Returns:
            Dictionary with results from different methods and diagnostics
        """
        methods = ['direct', 'euler', 'wynn', 'shanks', 'aitken']
        comparison_results = {}
        
        print("Comparing resummation methods...")
        
        # Get partial sums for analysis
        prop = 'energy'  # Use energy as representative property
        if prop not in self.weights:
            print("No energy weights available for comparison")
            return {}
            
        sum_by_order = defaultdict(lambda: np.zeros_like(self.temp_values))
        
        for cluster_id, weight in self.weights[prop].items():
            order = self.clusters[cluster_id]['order']
            if order_cutoff is not None and order > order_cutoff:
                continue
            sum_by_order[order] += weight * self.clusters[cluster_id]['multiplicity']
        
        if not sum_by_order:
            print("No data available for comparison")
            return {}
            
        # Create partial sums array
        max_order = max(sum_by_order.keys())
        partial_sums = []
        cumulative_sum = np.zeros_like(self.temp_values)
        
        for order in range(max_order + 1):
            if order in sum_by_order:
                cumulative_sum += sum_by_order[order]
            partial_sums.append(cumulative_sum.copy())
        
        partial_sums = np.array(partial_sums)
        
        # Analyze convergence
        convergence_info = self.analyze_convergence(partial_sums)
        comparison_results['convergence_analysis'] = convergence_info
        
        # Test each method
        for method in methods:
            try:
                result = self.apply_resummation(partial_sums, method=method)
                comparison_results[method] = {
                    'result': result,
                    'final_value_avg': np.mean(result),
                    'final_value_std': np.std(result)
                }
                
                # Calculate convergence metrics
                if len(partial_sums) > 1:
                    direct_result = partial_sums[-1]
                    improvement = np.mean(np.abs(result - direct_result))
                    comparison_results[method]['improvement_from_direct'] = improvement
                    
            except Exception as e:
                print(f"Error with method {method}: {e}")
                comparison_results[method] = {'error': str(e)}
        
        # Automatic selection
        auto_method = self.select_resummation_method(partial_sums)
        comparison_results['auto_selected'] = auto_method
        
        # Print summary
        print("\nResummation Method Comparison:")
        print("=" * 50)
        print(f"Convergence Analysis:")
        print(f"  Converged: {convergence_info['converged']}")
        print(f"  Oscillatory: {convergence_info['oscillatory']}")
        print(f"  Ratio test: {convergence_info['ratio_test']:.4f}" if convergence_info['ratio_test'] else "  Ratio test: N/A")
        print(f"  Auto-selected method: {auto_method}")
        print("\nMethod Results (average final values):")
        
        for method in methods:
            if method in comparison_results and 'result' in comparison_results[method]:
                avg_val = comparison_results[method]['final_value_avg']
                print(f"  {method:10s}: {avg_val:.6e}")
        
        if save_comparison:
            self._save_comparison_plots(comparison_results, partial_sums)
            
        return comparison_results
    
    def _save_comparison_plots(self, comparison_results, partial_sums):
        """Save comparison plots for different resummation methods."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Convergence of partial sums
            orders = np.arange(len(partial_sums))
            temp_mid = len(self.temp_values) // 2  # Use middle temperature for visualization
            
            ax1.plot(orders, partial_sums[:, temp_mid], 'bo-', label='Partial sums')
            ax1.set_xlabel('Order')
            ax1.set_ylabel('Energy (middle T)')
            ax1.set_title('Convergence of Partial Sums')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Method comparison at middle temperature
            methods = ['direct', 'euler', 'wynn', 'shanks', 'aitken']
            values = []
            labels = []
            
            for method in methods:
                if method in comparison_results and 'result' in comparison_results[method]:
                    values.append(comparison_results[method]['result'][temp_mid])
                    labels.append(method)
            
            if values:
                ax2.bar(labels, values)
                ax2.set_ylabel('Energy (middle T)')
                ax2.set_title('Method Comparison')
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Temperature dependence comparison
            temp_range = self.temp_values
            for method in methods:
                if method in comparison_results and 'result' in comparison_results[method]:
                    result = comparison_results[method]['result']
                    ax3.plot(temp_range, result, label=method, alpha=0.7)
            
            ax3.set_xlabel('Temperature')
            ax3.set_ylabel('Energy')
            ax3.set_title('Temperature Dependence')
            ax3.set_xscale('log')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Convergence analysis
            if len(partial_sums) > 1:
                diffs = np.abs(np.diff(partial_sums[:, temp_mid]))
                ax4.semilogy(orders[1:], diffs, 'ro-')
                ax4.set_xlabel('Order')
                ax4.set_ylabel('|Difference|')
                ax4.set_title('Convergence Rate')
                ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('resummation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Comparison plots saved to 'resummation_comparison.png'")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def plot_results(self, results, save_path=None):
        """Plot energy and specific heat vs temperature."""
        temperatures = 1.0 / self.temp_values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot energy
        ax1.plot(temperatures, results['energy'], 'b-')
        ax1.set_xlabel('Temperature (T)')
        ax1.set_ylabel('Energy per site')
        ax1.set_title('Energy vs Temperature')
        ax1.set_xscale('log')
        
        # Plot specific heat
        ax2.plot(temperatures, results['specific_heat'], 'r-')
        ax2.set_xlabel('Temperature (T)')
        ax2.set_ylabel('Specific Heat per site')
        ax2.set_title('Specific Heat vs Temperature')
        ax2.set_xscale('log')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NLC calculation for lattice models')
    parser.add_argument('--cluster_dir', required=True, help='Directory containing cluster information files')
    parser.add_argument('--eigenvalue_dir', required=True, help='Directory containing eigenvalue files from ED calculations')
    parser.add_argument('--output_dir', default='.', help='Directory to save output files')
    parser.add_argument('--resummation_method', default='auto', 
                       choices=['auto', 'direct', 'none', 'euler', 'wynn',
                                'shanks', 'aitken', 'pade', 'theta',
                                'robust', 'wynn_multi', 'brezinski',
                                'entropy_derived'],
                       help='Resummation method for series acceleration')
    parser.add_argument('--euler_resum', action='store_true', help='Use Euler resummation (deprecated, use --resummation_method euler)')
    parser.add_argument('--order_cutoff', type=int, help='Maximum order to include in summation')
    parser.add_argument('--plot', action='store_true', help='Generate plot of results')
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')
    parser.add_argument('--energy_unit', type=str, default='dimensionless',
                        choices=['dimensionless', 'K', 'meV'],
                        help='Unit of the ED energies relative to the temperature '
                             'grid. Default (dimensionless / K): kB=1, T and E on '
                             'the same scale -- no conversion (the historical '
                             'behaviour). meV: couplings were written in meV and '
                             'the temperature grid is in KELVIN -> all energies '
                             '(eigenvalues, stored P(T) curves and their grids) '
                             'are multiplied by 1/kB = 11.6045 K/meV.')
    parser.add_argument('--temp_min', type=float, default=1e-4, help='Minimum temperature for calculations')
    parser.add_argument('--temp_max', type=float, default=1.0, help='Maximum temperature for calculations')
    parser.add_argument('--temp_bins', type=int, default=200, help='Number of temperature points to calculate')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')
    parser.add_argument('--compare_methods', action='store_true', help='Compare different resummation methods')
    parser.add_argument('--save_comparison', action='store_true', help='Save comparison plots and data')
    parser.add_argument('--fix_negative_cv', action='store_true', 
                       help='Apply derivative-based fix for negative specific heat (convergence artifact workaround)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create NLC instance
    nlc = NLCExpansion(args.cluster_dir, args.eigenvalue_dir, args.temp_min, args.temp_max, args.temp_bins, args.measure_spin, args.SI_units,
                       energy_unit=args.energy_unit)
    
    # Handle backward compatibility for euler_resum flag
    resummation_method = args.resummation_method
    if args.euler_resum and resummation_method == 'auto':
        resummation_method = 'euler'
        print("Note: --euler_resum flag is deprecated, use --resummation_method euler instead")

    # Cross-pipeline alias normalisation: this kernel implements a
    # different subset than NLC_sum_ftlm / NLC_sum_triangular. Map any
    # method name not natively supported onto its closest equivalent.
    _RESUM_ALIAS = {
        'none': 'direct',
        'theta': 'wynn',          # no Brezinski theta in this kernel
        'robust': 'auto',
        'wynn_multi': 'wynn',
        'brezinski': 'wynn',
        'entropy_derived': 'wynn',
    }
    if resummation_method in _RESUM_ALIAS:
        new = _RESUM_ALIAS[resummation_method]
        print(f"Note: --resummation_method={resummation_method} mapped to '{new}' "
              f"in the NLC_sum (pyrochlore) kernel.")
        resummation_method = new
    
    # Run NLC calculation
    results = nlc.run(resummation_method=resummation_method, order_cutoff=args.order_cutoff, 
                      fix_negative_cv=args.fix_negative_cv)
    
    # Compare methods if requested
    if args.compare_methods:
        print("\n" + "="*60)
        print("COMPARING RESUMMATION METHODS")
        print("="*60)
        comparison = nlc.compare_resummation_methods(
            order_cutoff=args.order_cutoff,
            save_comparison=args.save_comparison
        )
    
    # Save results in separate files for each quantity
    energy_file = os.path.join(args.output_dir, "nlc_energy.txt")
    specific_heat_file = os.path.join(args.output_dir, "nlc_specific_heat.txt")
    entropy_file = os.path.join(args.output_dir, "nlc_entropy.txt")

    # Save the three thermodynamic curves; when stochastic (OFTLM) clusters
    # contributed, a third column carries the propagated std-error band.
    for prop, path, label in (
        ('energy', energy_file, 'Energy'),
        ('specific_heat', specific_heat_file, 'Specific_Heat'),
        ('entropy', entropy_file, 'Entropy'),
    ):
        err = results.get(f'{prop}_stderr')
        with open(path, 'w') as f:
            if err is not None:
                f.write(f"# Temperature\t{label}\tStd_Error\n")
                for i, temp in enumerate(nlc.temp_values):
                    f.write(f"{temp:.8e}\t{results[prop][i]:.8e}\t{err[i]:.8e}\n")
            else:
                f.write(f"# Temperature\t{label}\n")
                for i, temp in enumerate(nlc.temp_values):
                    f.write(f"{temp:.8e}\t{results[prop][i]:.8e}\n")

    # Convergence deliverables: both accelerations + bare orders per T
    # (the Euler-Wynn spread IS the resummation error bar), and the
    # trusted-T summary. These are what an order-8 paper figure quotes.
    if 'energy_euler' in results:
        comp_file = os.path.join(args.output_dir, "nlc_resummation_comparison.txt")
        with open(comp_file, 'w') as f:
            f.write("# Per-temperature comparison of resummation schemes.\n"
                    "# Trust a curve only where bare_last~=bare_prev AND "
                    "euler~=wynn (see nlc_convergence_summary.txt).\n")
            f.write("# T" + "".join(
                f"\t{p}_{s}" for p in ('energy', 'specific_heat', 'entropy')
                for s in ('euler', 'wynn', 'bare_last', 'bare_prev')) + "\n")
            for i, temp in enumerate(nlc.temp_values):
                row = [f"{temp:.8e}"]
                for p in ('energy', 'specific_heat', 'entropy'):
                    for s in ('euler', 'wynn', 'bare_last', 'bare_prev'):
                        row.append(f"{results[f'{p}_{s}'][i]:.8e}")
                f.write("\t".join(row) + "\n")

        summary_file = os.path.join(args.output_dir, "nlc_convergence_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("# Automated NLCE convergence report\n")
            f.write("# Criterion: last two bare orders agree AND Euler vs "
                    "Wynn agree, both to 1% (TKR arXiv:1207.3366; "
                    "Schaefer et al. PRB 102, 054408).\n")
            for prop, t_star in results.get('trusted_T', {}).items():
                if t_star is not None:
                    f.write(f"{prop}\ttrusted_T_min\t{t_star:.8e}\n")
                else:
                    f.write(f"{prop}\ttrusted_T_min\tNONE\n")
            err = results.get('specific_heat_stderr')
            if err is not None:
                f.write(f"specific_heat\tmax_stochastic_stderr\t"
                        f"{float(np.max(err)):.8e}\n")
        print(f"Convergence report: {summary_file}")
        print(f"Scheme comparison:  {comp_file}")

    # Save spin expectation values if requested
    if args.measure_spin:
        spin_file = os.path.join(args.output_dir, "nlc_spin_expectations.txt")
        with open(spin_file, 'w') as f:
            f.write("# Temperature\tsp\tsp_imag\tsm\tsm_imag\tsz\tsz_imag\n")
            for i, temp in enumerate(nlc.temp_values):
                sp = results['sp'][i]
                sm = results['sm'][i]
                sz = results['sz'][i]
                f.write(f"{temp:.8e}\t{sp.real:.8e}\t{sp.imag:.8e}\t{sm.real:.8e}\t{sm.imag:.8e}\t{sz.real:.8e}\t{sz.imag:.8e}\n")

    # Plot results if requested
    if args.plot:
        temperatures = nlc.temp_values
        
        # Plot energy
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['energy'], 'b-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Energy per site')
        plt.title('Energy vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        energy_plot_path = os.path.join(args.output_dir, "nlc_energy.png")
        plt.savefig(energy_plot_path)
        plt.close()
        
        # Plot specific heat
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['specific_heat'], 'r-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Specific Heat per site')
        plt.title('Specific Heat vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        specific_heat_plot_path = os.path.join(args.output_dir, "nlc_specific_heat.png")
        plt.savefig(specific_heat_plot_path)
        plt.close()
        
        # Plot entropy
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['entropy'], 'g-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Entropy per site')
        plt.title('Entropy vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        entropy_plot_path = os.path.join(args.output_dir, "nlc_entropy.png")
        plt.savefig(entropy_plot_path)
        plt.close()

        # Plot spin expectation values if requested
        if args.measure_spin:
            plt.figure(figsize=(8, 6))
            plt.plot(temperatures, results['sp'], 'm-', label='sp')
            plt.plot(temperatures, results['sm'], 'c-', label='sm')
            plt.plot(temperatures, results['sz'], 'y-', label='sz')
            plt.xlabel('Temperature (T)')
            plt.ylabel('Spin Expectation Values')
            plt.title('Spin Expectation Values vs Temperature')
            plt.xscale('log')
            plt.legend()
            plt.tight_layout()
            spin_plot_path = os.path.join(args.output_dir, "nlc_spin_expectations.png")
            plt.savefig(spin_plot_path)
            plt.close()
    
    print(f"NLC calculation completed! Results saved to {args.output_dir}")
    