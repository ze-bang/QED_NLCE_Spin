# QED_NLCE

Numerical Linked Cluster Expansion (NLCE) workflows for frustrated quantum
spin models, built on top of the [QED](https://github.com/ze-bang/QED) exact
diagonalization toolkit.

This package was extracted from `QED/workflows/nlce/` to allow the NLCE
toolkit to evolve independently of the underlying ED solver. The Python
package was renamed `workflows.nlce` → `qed_nlce`.

## Install

```bash
# Install the QED Python package (required runtime dep). qed_nlce
# calls the canonical qed.solve(H, ...) / qed.thermal(H, ...)
# Python verbs directly -- the C++ ./ED binary is no longer invoked.
git clone https://github.com/ze-bang/QED.git
cd QED && pip install .

# Then install qed_nlce. Pip will pick up the qed dep automatically.
pip install git+https://github.com/ze-bang/QED_NLCE.git
```

## Quick start

**Recommended default — let the framework pick the backend per cluster:**

```bash
qed-nlce --geometry triangular_site --pipeline auto \
         --max_order 8 \
         --J1 1.0 --temp_min 0.1 --temp_max 10 --temp_bins 100 --thermo \
         --base_dir output/tri_auto_o8
```

The `auto` pipeline runs FULL dense ED below `--auto_full_hilbert`
(default `2**14 = 16384`) and switches to KPM-DOS thermodynamics
above. KPM-DOS Hutchinson variance scales as `1/sqrt(R*D)` so
per-cluster error *improves* as the cluster grows — at `N=20`,
`R=20`, `M=2048` it is `~4e-4`, comfortably below the per-cluster
target dictated by NLCE Möbius condition number `κ ~ 30-80`.

**Other pipelines:**

```bash
# Force FULL ED everywhere
qed-nlce --geometry pyrochlore --pipeline full_ed --max_order 5 \
         --base_dir output/pyro_full_o5

# Force KPM-DOS everywhere (low-variance trace, large clusters)
qed-nlce --geometry triangular_site --pipeline kpm_dos --max_order 8 \
         --kpm_moments 2048 --kpm_random_vectors 20 --thermo \
         --base_dir output/tri_kpm_o8

# Legacy FTLM (kept for back-compat; high noise floor at NLCE order ≥ 6)
qed-nlce --geometry pyrochlore --pipeline ftlm --max_order 5 \
         --ftlm_samples 30 --base_dir output/pyro_ftlm_o5

# Lanczos lowest-eigenvalues (for ground-state observables only)
qed-nlce --geometry pyrochlore --pipeline lanczos_boost --max_order 5 \
         --base_dir output/pyro_lz_o5
```

**Legacy fitter integration** (`qed_nlce/analysis/nlc_fit_triangular.py`)
accepts `ed_method="AUTO"`, `"KPM_DOS"`, `"FTLM"`, or `"FULL"` in
`fixed_params`; the `nlce_triangular.py` shim auto-promotes those to
the matching pipeline.

## Pipelines

| Pipeline | Per-cluster ED | Best for | Notes |
| --- | --- | --- | --- |
| `auto` | FULL ↔ KPM-DOS (crossover at `--auto_full_hilbert`) | **Production runs at orders ≥ 6** | Smart default; orthogonal `--auto_fixed_sz` and `--auto_streaming_symmetry` axes. |
| `full_ed` | LAPACK dense ED, all eigenvalues | Small clusters / orders ≤ 5 | Noise-free anchor; cost `O(D^3)`. |
| `kpm_dos` | KPM Chebyshev moments + Hutchinson trace + Cheb-Gauss quadrature | Large clusters (`N ≥ 14`) | C++ kernel `ed::kpm_dos`. Variance `1/√(R·D)`. |
| `ftlm` | Finite-Temperature Lanczos | Legacy; large `D` with low memory | `~5%` per-cluster noise floor; Möbius amplifies to `15–40%` on summed `C(T)` at orders ≥ 6. |
| `lanczos_boost` | Lanczos lowest-`k` eigenvalues | Ground-state observables | No thermo. |

### `auto` pipeline knobs

| Flag | Default | Meaning |
| --- | --- | --- |
| `--auto_backend {kpm_dos,ftlm}` | `kpm_dos` | Iterative backend used above the FULL-ED ceiling. |
| `--auto_full_hilbert N` | `16384` (`2**14`) | FULL-ED ceiling on full Hilbert-space dim. |
| `--auto_kpm_moments M` | `2048` | KPM Chebyshev moments. |
| `--auto_kpm_random_vectors R` | `20` | KPM Hutchinson random-vector count. |
| `--auto_kpm_kernel {jackson,lorentz}` | `jackson` | KPM smoothing kernel. |
| `--auto_kpm_seed S` | `0` | Seed for Hutchinson vectors (0 = nondeterministic). |
| `--auto_min_samples`, `--auto_max_samples` | `40, 200` | Adaptive sample range, only used when `--auto_backend=ftlm`. |
| `--auto_fixed_sz` | off | Append `_FIXED_SZ` to every cluster's method (currently a single-Sz-block trace; **only physically correct if you know your model conserves Sz_total *and* you want the partial trace** — full-physical thermo with sector summing is not yet wired). |
| `--auto_streaming_symmetry` | off | Cluster-automorphism orbit-basis decomposition (cached per cluster). |

### Symmetry & U(1) Sz blocking

Two orthogonal axes for shrinking the per-cluster Hilbert space:

* **`--auto_streaming_symmetry`** — exploit the geometric automorphism
  group of each cluster (lattice symmetries). Adds a one-time orbit-
  basis construction per cluster (cached under the cluster's ham
  dir as `basis_cache/`). **Always physically correct** — symmetry
  sectors are added back together by the framework.

* **`--auto_fixed_sz`** — assert the Hamiltonian conserves
  `S^z_total`. Routes every cluster through the dispatcher with
  `params.use_fixed_sz = True`, which currently selects the *single*
  `Sz = 0` block in the C++ backend. **This is a partial trace** —
  use only when:

  1. Your model has no transverse field (`h_x`, `h_y`) and no Sx/Sy
     single-site or anisotropic xy terms — i.e. it actually commutes
     with `S^z_total`. (Pure Heisenberg / XXZ / Ising fit; transverse
     field, DM interactions, Kitaev terms break this.)
  2. You explicitly want the `Sz = 0` partial trace (e.g. you are
     studying a magnetization plateau or restricting to a specific
     sector). For full unconstrained thermo, leave this off.

  Auto-detection of U(1)-Sz from `InterAll.dat` is on the roadmap;
  for now it is an explicit user assertion.

```bash
# Heisenberg model with full geometric symmetry, all Sz sectors:
qed-nlce --geometry triangular_site --pipeline auto --max_order 8 \
         --J1 1.0 --thermo --auto_streaming_symmetry \
         --base_dir output/tri_auto_sym_o8

# Same model, restricted to Sz = 0 sector (single-block trace):
qed-nlce --geometry triangular_site --pipeline auto --max_order 8 \
         --J1 1.0 --thermo --auto_streaming_symmetry --auto_fixed_sz \
         --base_dir output/tri_auto_sym_sz0_o8
```

## Backend

Every cluster is diagonalized in-process by importing `qed` and
dispatching through the canonical three-verb Python surface
(`qed.solve` for ground-state lanes / `qed.thermal` for FTLM,
LTLM, KPM-DOS, mTPQ, cTPQ). This eliminates the per-cluster fork +
OpenMP / CUDA initialization overhead that dominates wall-time at
high NLCE orders.

MPI-only methods (`SCALAPACK`, `SCALAPACK_MIXED`, `mTPQ_MPI`) are
**not** supported by the in-process backend — a Python interpreter
cannot call `MPI_Init`. If you need them, run the standalone
`ed_distributed_main` (or the legacy `./ED`) directly under
`mpiexec`. `qed-nlce`'s preflight check rejects MPI methods up front
with a clear error message.

The legacy CLI flags `--ed_executable`, `--no_in_process`,
`--in_process`, `--auto_in_process` are silently accepted for
back-compat but ignored.

## Package layout

| Module | Purpose |
| --- | --- |
| `qed_nlce.core` | `Geometry` / `Pipeline` abstractions, `NLCEWorkflow` orchestrator, in-process ED bridge (`EDOptions`, `run_ed_in_process`, `can_run_in_process`). |
| `qed_nlce.geometries` | Concrete lattices: `pyrochlore`, `triangular_site`, `triangular_triangle`. |
| `qed_nlce.pipelines` | ED strategies: `auto`, `full_ed`, `kpm_dos`, `lanczos_boost`, `ftlm`. |
| `qed_nlce.prep` | Cluster generators (graph enumeration). |
| `qed_nlce.run` | NLCE summation kernels + legacy per-lattice driver scripts. |
| `qed_nlce.analysis` | Convergence diagnostics, fitting drivers, plot helpers. |
| `qed_nlce.cli` / `qed_nlce.__main__` | Unified `qed-nlce` CLI. |

## Relationship to QED

`qed_nlce` depends on QED purely through the `qed` Python package (a
required runtime dependency, declared in `pyproject.toml`). Every
cluster's ED is dispatched in-process via QED's canonical three-verb
Python surface (`qed.solve` / `qed.thermal` / `qed.spectral`), which
internally routes through the Phase 7+ 5-axis dispatcher (orthogonal
`use_gpu` / `use_mpi` / `use_fixed_sz` / `use_symmetry` axes).

`qed_nlce` does **not** link against any QED C++ library, does **not**
have a build-time dependency on QED, and does **not** invoke the
`./ED` binary as a subprocess.

Build-introspection preflight uses `qed.has_cuda_build()` /
`qed.has_mpi_build()` to abort early when the requested method is
incompatible with the installed `qed` build (e.g. asking for a `*_GPU`
method against a CPU-only build).

## License

Same as the QED project — see `LICENSE`.
