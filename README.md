# QED_NLCE

Numerical Linked Cluster Expansion (NLCE) workflows for frustrated quantum
spin models, built on top of the [QED](https://github.com/ze-bang/QED) exact
diagonalization toolkit.

This package was extracted from `QED/workflows/nlce/` to allow the NLCE
toolkit to evolve independently of the underlying ED solver. The Python
package was renamed `workflows.nlce` → `qed_nlce`.

## Install

```bash
# 1. Install the QED toolkit first (provides the ./ED and
#    ed_distributed_main binaries + the optional qed Python wheel
#    that powers the in-process backend). See
#    https://github.com/ze-bang/QED for instructions; in short:
git clone https://github.com/ze-bang/QED.git
cd QED && pip install . && cmake --preset release && cmake --build build -j

# 2. Install qed_nlce (this repo). Use the [qed] extra to also pull
#    the qed Python bindings -- required only for --in_process /
#    --auto_in_process and the build-introspection preflight.
pip install "git+https://github.com/ze-bang/QED_NLCE.git#egg=qed_nlce[qed]"
```

The `ED` binary is auto-discovered in this order:

1. `$QED_ED_BINARY` (or legacy `$ED_BINARY`).
2. `shutil.which("ED")` (binary on `$PATH`, e.g. installed via
   `cmake --install`).
3. `./build/ED`, `../QED/build/ED`, `../../QED/build/ED`.
4. `$VIRTUAL_ENV/bin/ED`.
5. The install root of the optional `qed` Python package.

Override at any time with `--ed_executable <path>`.

## Quick start

```bash
# Unified CLI -- pick a geometry and a pipeline.
qed-nlce --help
qed-nlce --geometry pyrochlore --pipeline ftlm \
         --max_order 5 --ftlm_samples 30 \
         --base_dir output/pyro_ftlm_o5

# In-process backend (skips the per-cluster ./ED fork; needs `pip
# install qed_nlce[qed]`). Big win for high-order workflows where the
# fork + OpenMP/CUDA init cost dominates the small-cluster ED runs.
qed-nlce --geometry triangular_site --pipeline full_ed \
         --max_order 6 --auto_in_process \
         --base_dir output/tri_full_o6
```

## Backends

| Mode (CLI) | Implementation | When to use |
| --- | --- | --- |
| (default) | Forks `./ED` per cluster (subprocess). | No `qed` Python wheel installed; or you need MPI / SCALAPACK methods. |
| `--auto_in_process` | Per-cluster: in-process via `qed.exact_diagonalization_from_directory` when supported, transparent fallback to `./ED` for `SCALAPACK*` / `mTPQ_MPI`. | Recommended default once the `[qed]` extra is installed. |
| `--in_process` | Hard-require the in-process backend; abort if `qed` is missing or method is MPI-only. | Reproducible benchmarks where the subprocess path must be excluded. |

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

`qed_nlce` is a **runtime consumer** of the QED toolkit, not a build-time
dependency. It does not link against any QED C++ library. Two execution
paths are supported:

1. **Subprocess** (zero Python deps on QED) — every cluster spawns
   `./ED`. Works with any QED build (CPU, MPI, CUDA, ScaLAPACK).
2. **In-process** (optional `[qed]` extra) — every cluster calls
   `qed.exact_diagonalization_from_directory(...)` directly. Bypasses
   the fork + OpenMP/CUDA initialization for each cluster, and gives
   the workflow access to QED's Phase 7+ canonical 5-axis dispatcher
   (orthogonal `use_gpu` / `use_mpi` / `use_fixed_sz` / `use_symmetry`
   axes, build introspection via `qed.has_cuda_build()` /
   `qed.has_mpi_build()` / `qed.has_scalapack_build()`).

## License

Same as the QED project — see `LICENSE`.
