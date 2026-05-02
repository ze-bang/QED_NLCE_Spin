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
# calls qed.exact_diagonalization_from_directory(...) directly --
# the C++ ./ED binary is no longer invoked.
git clone https://github.com/ze-bang/QED.git
cd QED && pip install .

# Then install qed_nlce. Pip will pick up the qed dep automatically.
pip install git+https://github.com/ze-bang/QED_NLCE.git
```

## Quick start

```bash
# Unified CLI -- pick a geometry and a pipeline.
qed-nlce --help
qed-nlce --geometry pyrochlore --pipeline ftlm \
         --max_order 5 --ftlm_samples 30 \
         --base_dir output/pyro_ftlm_o5

# Each cluster's ED runs in-process via
# qed.exact_diagonalization_from_directory(...). There is no
# subprocess fork per cluster.
qed-nlce --geometry triangular_site --pipeline full_ed \
         --max_order 6 \
         --base_dir output/tri_full_o6
```

## Backend

Every cluster is diagonalized in-process by importing `qed` and
calling `qed.exact_diagonalization_from_directory(...)`. This
eliminates the per-cluster fork + OpenMP / CUDA initialization
overhead that dominates wall-time at high NLCE orders.

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
| `qed_nlce.pipelines` | ED strategies: `full_ed`, `lanczos_boost`, `ftlm`. |
| `qed_nlce.prep` | Cluster generators (graph enumeration). |
| `qed_nlce.run` | NLCE summation kernels + legacy per-lattice driver scripts. |
| `qed_nlce.analysis` | Convergence diagnostics, fitting drivers, plot helpers. |
| `qed_nlce.cli` / `qed_nlce.__main__` | Unified `qed-nlce` CLI. |

## Relationship to QED

`qed_nlce` depends on QED purely through the `qed` Python package (a
required runtime dependency, declared in `pyproject.toml`). Every
cluster's ED is dispatched in-process via QED's Phase 7+ canonical
5-axis dispatcher (orthogonal `use_gpu` / `use_mpi` / `use_fixed_sz` /
`use_symmetry` axes).

`qed_nlce` does **not** link against any QED C++ library, does **not**
have a build-time dependency on QED, and does **not** invoke the
`./ED` binary as a subprocess.

Build-introspection preflight uses `qed.has_cuda_build()` /
`qed.has_mpi_build()` / `qed.has_scalapack_build()` to abort early
when the requested method is incompatible with the installed `qed`
build.

## License

Same as the QED project — see `LICENSE`.
