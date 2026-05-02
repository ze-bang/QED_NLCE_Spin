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
#    ed_distributed_main binaries that the NLCE pipelines invoke
#    via subprocess). See https://github.com/ze-bang/QED for
#    instructions; in short:
git clone https://github.com/ze-bang/QED.git
cd QED && pip install . && cmake --preset release && cmake --build build -j

# 2. Install qed_nlce (this repo).
pip install git+https://github.com/ze-bang/QED_NLCE.git
```

## Quick start

```bash
# Unified CLI — pick a geometry and a pipeline.
qed-nlce --help
qed-nlce --geometry pyrochlore --pipeline ftlm \
         --max_order 5 --ftlm_samples 30 \
         --output_dir output/pyro_ftlm_o5

# Equivalent module form.
python -m qed_nlce --geometry triangular_site --pipeline full_ed \
                   --max_order 6 --output_dir output/tri_full_o6
```

The pipelines call the `ED` binary via `subprocess`; pass
`--ed_executable /path/to/ED` if it is not on `$PATH` or under
`./build/ED`.

## Package layout

| Module | Purpose |
| --- | --- |
| `qed_nlce.core` | `Geometry` / `Pipeline` abstractions, `NLCEWorkflow` orchestrator, ED-bridge helpers (`EDOptions`, `build_ed_command`, `run_ed_subprocess`). |
| `qed_nlce.geometries` | Concrete lattices: `pyrochlore`, `triangular_site`, `triangular_triangle`. |
| `qed_nlce.pipelines` | ED strategies: `full_ed`, `lanczos_boost`, `ftlm`. |
| `qed_nlce.prep` | Cluster generators (graph enumeration). |
| `qed_nlce.run` | NLCE summation kernels + legacy per-lattice driver scripts. |
| `qed_nlce.analysis` | Convergence diagnostics, fitting drivers, plot helpers. |
| `qed_nlce.cli` / `qed_nlce.__main__` | Unified `qed-nlce` CLI. |

## Relationship to QED

`qed_nlce` is a **runtime consumer** of the QED toolkit, not a build-time
dependency. It does not link against any QED C++ library and does not
import the `qed._core` pybind11 module — it shells out to the `ED`
binary for every cluster diagonalization. This means:

- You can update QED and `qed_nlce` independently.
- Any QED build (CPU, MPI, CUDA) is usable by NLCE; pick the right
  binary via `--ed_executable`.
- Power users who want in-process Python invocation can install the
  optional `qed` dependency: `pip install "qed_nlce[qed]"`.

## License

Same as the QED project — see `LICENSE`.
