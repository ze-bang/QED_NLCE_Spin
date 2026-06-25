# QED_NLCE

Numerical Linked Cluster Expansion (NLCE) workflows for frustrated quantum
spin-1/2 models, powered by a **self-contained, symmetry-adapted full dense
exact diagonalization** core.

The diagonalizer is built on `numpy` + `scipy.linalg.eigh` (LAPACK)
only: there is no link to any external ED binary or library — every
cluster is diagonalized in-process. Cluster *generation*
(`qed_nlce.prep`) additionally uses [`pynauty`](https://github.com/pdobsan/pynauty)
for canonical graph certificates and automorphism counts.

## Install

```bash
pip install git+https://github.com/ze-bang/QED_NLCE.git
# or, from a checkout:
pip install .
```

Runtime dependencies are `numpy`, `scipy`, `h5py`, `matplotlib`,
`networkx`, `pandas`, `tqdm`, and `pynauty` (the latter only for the
cluster-generation step).

## Quick start

```bash
qed-nlce --geometry triangular_site --pipeline full_ed \
         --max_order 8 \
         --J1 1.0 --temp_min 0.1 --temp_max 10 --temp_bins 100 --thermo \
         --base_dir output/tri_full_o8

qed-nlce --geometry pyrochlore --pipeline full_ed --max_order 5 \
         --Jxx 1.0 --Jyy 1.0 --Jzz 1.0 --thermo \
         --base_dir output/pyro_full_o5
```

Every cluster is solved with the **full eigenvalue spectrum** via
symmetry-adapted dense diagonalization. Thermodynamics
(`C(T)`, `E(T)`, `S(T)`, `F(T)`) are computed exactly from that
spectrum and the per-cluster weights are summed by the NLCE kernel.

## Pipeline

There is a single, noise-free pipeline: `full_ed`. It diagonalizes
each cluster Hamiltonian exactly using LAPACK (`scipy.linalg.eigh`),
after blocking the Hilbert space by every available symmetry. Cost is
`O(D_block^3)` summed over the (much smaller) symmetry blocks rather
than `O(D^3)` on the full space.

## Symmetry blocking

The solver exhausts every symmetry the Hamiltonian admits, in this
order (each reduction provably preserves the full eigenvalue
multiset):

1. **U(1) total-`S^z` sectors** — applied whenever the Hamiltonian
   conserves `S^z_total` (auto-detected from the operator terms).
2. **Spatial automorphisms** — the geometric symmetry group of the
   cluster graph (respecting complex bond phases) is reduced to a
   maximal abelian subgroup, and the Hilbert space is decomposed into
   momentum/character orbit blocks.
3. **Spin-flip Z2** — when the Hamiltonian is invariant under the
   global spin flip (auto-detected; broken by a longitudinal field),
   conjugate `S^z` sectors are merged and split by flip parity.
4. **Reality / time-reversal** — blocks with no residual imaginary
   part are diagonalized with the real symmetric LAPACK path.

All symmetry reductions are added back together by the framework, so
the summed spectrum is identical to brute-force dense ED. This is what
lets the workflow reach higher NLCE order: e.g. for the `N = 12`
Heisenberg ring the largest dense block shrinks from `4096` to `66`.

```python
from qed_nlce.ed import solve_spectrum, SpinHalfOperator, OP_SP, OP_SM, OP_SZ

op = SpinHalfOperator(12)
for i in range(12):
    j = (i + 1) % 12
    op.add_two(OP_SZ, i, OP_SZ, j, 1.0)
    op.add_two(OP_SP, i, OP_SM, j, 0.5)
    op.add_two(OP_SM, i, OP_SP, j, 0.5)

spectrum = solve_spectrum(op, use_symmetry=True)   # full spectrum, symmetry-adapted
```

## Backend

Every cluster is diagonalized in-process by the dense core in
`qed_nlce.core.dense_ed` → `qed_nlce.ed.solve_spectrum`. There is no
subprocess fork, no MPI, no GPU. Results are written to
`<ed_dir>/cluster_{id}_order_{o}/output/ed_results.h5` with the full
sorted spectrum under `/eigendata/eigenvalues` and (optionally)
thermodynamics under `/thermodynamics/`.

## Package layout

| Module | Purpose |
| --- | --- |
| `qed_nlce.ed` | Self-contained dense ED core: `SpinHalfOperator`, symmetry analysis, `solve_spectrum`, `thermodynamics`, HDF5/`Trans.dat` I/O. |
| `qed_nlce.hamiltonians` | Cluster file reader + pyrochlore / triangular operator builders. |
| `qed_nlce.core` | `Geometry` / `Pipeline` abstractions, `NLCEWorkflow` orchestrator, in-process dense bridge (`run_ed_in_process`, `can_run_in_process`). |
| `qed_nlce.geometries` | Concrete lattices: `pyrochlore`, `triangular_site`, `triangular_triangle`. |
| `qed_nlce.pipelines` | The `full_ed` dense pipeline. |
| `qed_nlce.prep` | Cluster generators (graph enumeration). |
| `qed_nlce.run` | NLCE summation kernels + per-lattice driver scripts. |
| `qed_nlce.analysis` | Convergence diagnostics, fitting drivers, plot helpers. |
| `qed_nlce.cli` / `qed_nlce.__main__` | Unified `qed-nlce` CLI. |

## Benchmark

`scripts/benchmark_symmetry.py` times symmetry-adapted dense ED
against plain dense ED across Heisenberg-ring sizes and verifies the
two spectra agree to machine precision:

```bash
python scripts/benchmark_symmetry.py --max-n 12
```

## License

See `LICENSE`.
