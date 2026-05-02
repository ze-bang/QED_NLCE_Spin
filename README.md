# QED_NLCE

Numerical Linked Cluster Expansion (NLCE) workflows for frustrated quantum
spin models, built on top of the [QED](https://github.com/ze-bang/QED) exact
diagonalization toolkit.

This package was extracted from `QED/workflows/nlce/` to allow the NLCE
toolkit to evolve independently of the underlying ED solver. The Python
package was renamed `workflows.nlce` → `qed_nlce`.

## Install

```bash
# Install the QED toolkit. The qed Python wheel is a *required*
# dependency of qed_nlce -- the in-process backend imports it
# directly. The ./ED binary is also required as a fallback for
# MPI-only methods (SCALAPACK*, mTPQ_MPI).
git clone https://github.com/ze-bang/QED.git
cd QED && pip install . && cmake --preset release && cmake --build build -j

# Then install qed_nlce. Pip will pick up the qed dep automatically.
pip install git+https://github.com/ze-bang/QED_NLCE.git
```

The `ED` binary is auto-discovered in this order:

1. `$QED_ED_BINARY` (or legacy `$ED_BINARY`).
2. `shutil.which("ED")` (binary on `$PATH`, e.g. installed via
   `cmake --install`).
3. `./build/ED`, `../QED/build/ED`, `../../QED/build/ED`.
4. `$VIRTUAL_ENV/bin/ED`.
5. The install root of the `qed` Python package.

Override at any time with `--ed_executable <path>`.

## Quick start

```bash
# Unified CLI -- pick a geometry and a pipeline.
qed-nlce --help
qed-nlce --geometry pyrochlore --pipeline ftlm \
         --max_order 5 --ftlm_samples 30 \
         --base_dir output/pyro_ftlm_o5

# By default qed_nlce runs each cluster in-process via
# qed.exact_diagonalization_from_directory(...). Pass --no_in_process
# to force every cluster through the ./ED subprocess (e.g. for
# benchmarking the legacy path).
qed-nlce --geometry triangular_site --pipeline full_ed \
         --max_order 6 --no_in_process \
         --base_dir output/tri_full_o6
```

## Backends

| Mode (CLI) | Implementation | When to use |
| --- | --- | --- |
| (default) | Per-cluster in-process via `qed.exact_diagonalization_from_directory(...)`. MPI-only methods (`SCALAPACK*`, `mTPQ_MPI`) transparently fall back to `./ED`. | Recommended; eliminates the per-cluster fork + OpenMP / CUDA initialization overhead that dominates wall-time at high NLCE orders. |
| `--no_in_process` | Forks `./ED` per cluster (legacy subprocess path). | Benchmarking the subprocess path, or when the qed wheel and the `./ED` binary were built with different feature flags. |

## Package layout

| Module | Purpose |
| --- | --- |
| `qed_nlce.core` | `Geometry` / `Pipeline` abstractions, `NLCEWorkflow` orchestrator, ED-bridge helpers (`EDOptions`, `build_ed_command`, `run_ed_subprocess`, `discover_ed_binary`, `run_ed_in_process`). |
| `qed_nlce.geometries` | Concrete lattices: `pyrochlore`, `triangular_site`, `triangular_triangle`. |
| `qed_nlce.pipelines` | ED strategies: `full_ed`, `lanczos_boost`, `ftlm`. |
| `qed_nlce.prep` | Cluster generators (graph enumeration). |
| `qed_nlce.run` | NLCE summation kernels + legacy per-lattice driver scripts. |
| `qed_nlce.analysis` | Convergence diagnostics, fitting drivers, plot helpers. |
| `qed_nlce.cli` / `qed_nlce.__main__` | Unified `qed-nlce` CLI. |

## Relationship to QED

`qed_nlce` depends on QED in two ways:

1. **Python**: it `import`s the `qed` Python package (a required
   runtime dependency, declared in `pyproject.toml`). Every cluster's
   ED is dispatched in-process via QED's Phase 7+ canonical 5-axis
   dispatcher (orthogonal `use_gpu` / `use_mpi` / `use_fixed_sz` /
   `use_symmetry` axes). `qed_nlce` does NOT link against any QED C++
   library and does NOT have a build-time dependency on QED.
2. **Subprocess**: for MPI-only methods (`SCALAPACK*`, `mTPQ_MPI`) it
   shells out to the `./ED` binary, since a Python interpreter cannot
   host `MPI_Init`. The same fallback kicks in if the user passes
   `--no_in_process`.

Build-introspection preflight uses `qed.has_cuda_build()` /
`qed.has_mpi_build()` / `qed.has_scalapack_build()` to warn early when
the requested method is incompatible with the installed `qed` build.

## License

Same as the QED project — see `LICENSE`.
