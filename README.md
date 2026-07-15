# QED_NLCE

Numerical Linked Cluster Expansion (NLCE) workflows for frustrated quantum
spin-1/2 models (pyrochlore, triangular), powered by the **QED C++ exact
diagonalization engine** ([ze-bang/QED](https://github.com/ze-bang/QED)).

QED_NLCE owns the NLCE mathematics — cluster generation (via
[`pynauty`](https://github.com/pdobsan/pynauty) canonical certificates),
embedding multiplicities, subcluster tables, weighted subtraction,
resummation — and delegates every cluster diagonalization to `qed`.

## Install

```bash
pip install git+https://github.com/ze-bang/QED_NLCE.git
# or, from a checkout:
pip install .
```

The `qed` C++ package is a **required runtime dependency** for all real
pipelines (both ED tiers route through it); install it from the sibling
QED repository. Other runtime dependencies are `numpy`, `scipy`, `h5py`,
`matplotlib`, `networkx`, `pandas`, `tqdm`, and `pynauty`.

## Quick start

```bash
qed-nlce --geometry triangular_triangle --pipeline full_ed \
         --max_order 8 \
         --J1 1.0 --temp_min 0.1 --temp_max 10 --temp_bins 100 --thermo \
         --base_dir output/tri_full_o8

qed-nlce --geometry pyrochlore --pipeline full_ed --max_order 5 \
         --Jxx 1.0 --Jyy 1.0 --Jzz 1.0 --thermo \
         --base_dir output/pyro_full_o5
```

## Two-tier ED dispatch

Each cluster is dispatched by its raw Hilbert dimension `2^N` against
`--oftlm_cutoff` (default `2^18`):

1. **Exact tier** (`2^N <= cutoff`): full eigenvalue spectrum via
   `qed.full_spectrum` — auto-detected spatial symmetry (**abelian and
   non-abelian**, the latter through QED's symmetry-adapted-basis engine
   with full `d_Γ >= 2` irrep reduction), U(1) total-`S^z`, spin-flip Z2,
   and time-reversal. The result is mathematically identical to
   brute-force dense ED (verified to 1e-8 by
   `tests/test_qed_bridge_parity.py` against the in-repo pure-Python
   oracle). Thermodynamics (`C(T)`, `E(T)`, `S(T)`, `F(T)`) follow
   exactly from the spectrum. Pass `--device gpu` for QED's batched GPU
   block eigensolve.
2. **OFTLM tier** (`2^N > cutoff`): matrix-free Orthogonalized
   Finite-Temperature Lanczos (Morita & Tohyama, PRR 2, 013205 (2020))
   with `--oftlm_num_exact` low-lying states resolved exactly and a
   stochastic trace over the rest (`--oftlm_num_samples`, error
   `~ 1/sqrt(R)`). Produces per-cluster `P(T)` curves directly; the
   summation kernels consume both tiers uniformly, and warn when a
   stochastic cluster's NLCE weight is cancellation-dominated (the
   regime where sample noise gets amplified).

Per-cluster logs (`qed-bridge` tag) report eigenvalue counts and wall
time — watch these when pushing `--oftlm_cutoff` up.

## Bond-colored cluster dedup (direction-dependent models)

For the triangular `kitaev` (JKΓΓ') and `anisotropic` (YbMgGaO4-type)
models, couplings depend on each bond's lattice direction, so
**topologically isomorphic embeddings are not isospectral** (a straight
3-site chain and a 120°-bent one differ at O(0.1|J|)). The generators
therefore support `--bond_colored`: clusters are deduplicated by
bond-direction-colored graph isomorphism, canonical modulo the S3 color
permutations induced by the lattice point group (exact covariances of
both models). The geometries enable this automatically when
`--model kitaev|anisotropic`; `xxz_j1j2` keeps the (smaller) purely
topological cluster sets. See `qed_nlce/prep/_bond_color.py` and
`tests/test_bond_colored_dedup.py`.

## Robustness for long (order-7/8) runs

* **Fail-fast**: `qed` importability is checked at workflow start, not
  hours in.
* **Complete-or-abort**: if any cluster's ED fails, the workflow aborts
  before summation (a partial high-order sum looks deceptively
  complete). Override with `--allow_incomplete_ed`.
* **Atomic writes**: `ed_results.h5` is written via temp-file +
  `os.replace`, so a crash never leaves a truncated file that the
  summation would silently treat as empty.
* **Resumable generation**: a `.generation_complete.json` marker lets a
  restarted run skip the (expensive at high order) cluster-generation
  step when parameters match.
* **Content-addressed eigenvalue cache**: keyed on cluster graph hash,
  Hamiltonian content, temperature grid, and all solver-tier knobs
  (`oftlm_*`, `device`) — retuning any of them correctly busts the cache.

## Package layout

| Module | Purpose |
| --- | --- |
| `qed_nlce.ed` | ED layer: `full_spectrum_qed` (exact tier), `oftlm_thermodynamics` (stochastic tier), `thermodynamics`; plus the pure-Python `solve_spectrum` correctness oracle. |
| `qed_nlce.hamiltonians` | Cluster file reader + pyrochlore / triangular operator builders. |
| `qed_nlce.core` | `Geometry` / `Pipeline` abstractions, `NLCEWorkflow` orchestrator, in-process ED dispatch, caches. |
| `qed_nlce.geometries` | Concrete lattices: `pyrochlore`, `triangular_triangle` (the triangular NLCE: order = number of triangles, normalized per site). `triangular_site` is retired from the runtime path and kept only as a correctness oracle. |
| `qed_nlce.pipelines` | The `full_ed` pipeline. |
| `qed_nlce.prep` | Cluster generators (graph enumeration, bond-colored certificates). |
| `qed_nlce.run` | NLCE summation kernels + per-lattice driver scripts. |
| `qed_nlce.analysis` | Convergence diagnostics, fitting drivers, plot helpers. |
| `qed_nlce.cli` / `qed_nlce.__main__` | Unified `qed-nlce` CLI. |

## Testing

```bash
pytest              # default suite (slow high-order validation excluded)
pytest -m slow      # order-6..8 generation validation -- run before any
                    # production order-7/8 campaign
```

`tests/test_qed_bridge_parity.py` is the load-bearing guard: it pins the
QED-backed exact tier against the pure-Python symmetry-adapted oracle
across Heisenberg/XXZ rings, field-broken and Sz-broken models, and real
pyrochlore clusters.

## License

See `LICENSE`.
