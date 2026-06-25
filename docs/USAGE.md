# QED_NLCE — Usage Guide

A practical guide to running the self-contained, symmetry-adapted full
dense exact-diagonalization NLCE workflow.

---

## 1. The pipeline

There is one pipeline: **`full_ed`**. It diagonalizes every cluster
Hamiltonian exactly with LAPACK (`scipy.linalg.eigh`), after blocking
the Hilbert space by every symmetry the Hamiltonian admits. The full
eigenvalue spectrum is noise-free, so per-cluster thermodynamics enter
the NLCE sum exactly.

```bash
qed-nlce --geometry triangular_site --pipeline full_ed --max_order 8 \
         --J1 1.0 --thermo --temp_min 0.1 --temp_max 10 --temp_bins 100 \
         --base_dir output/tri_full_o8
```

---

## 2. Geometries and their parameters

| Geometry | Key flags |
| --- | --- |
| `pyrochlore` | `--Jxx --Jyy --Jzz --h --field_dir` |
| `triangular_site` | `--J1 --J2 --Jz_ratio --h --field_dir --model {xxz_j1j2,kitaev,anisotropic}` plus `--Jzz --Jpm --Jpmpm --Jzpm --Gamma --Gamma_prime --g_ab --g_c` |
| `triangular_triangle` | same flags as `triangular_site` |

Example (anisotropic triangular model):

```bash
qed-nlce --geometry triangular_triangle --pipeline full_ed --max_order 7 \
         --model anisotropic --Jzz 1.0 --Jpm 0.1 --Jpmpm 0.05 --Jzpm 0.0 \
         --thermo --base_dir output/tri_aniso_o7
```

---

## 3. Symmetry blocking

The dense solver exhausts every symmetry, in this order; each
reduction provably preserves the full eigenvalue multiset, and the
framework sums the sectors back together so the result equals
brute-force dense ED.

1. **U(1) total-`S^z` sectors** — auto-detected whenever the operator
   conserves `S^z_total` (broken by transverse fields, Kitaev/DM
   terms, `S^±S^±` bonds).
2. **Spatial automorphisms** — the cluster graph's geometric symmetry
   group (respecting complex bond phases) reduced to a maximal abelian
   subgroup, giving momentum/character orbit blocks.
3. **Spin-flip Z2** — auto-detected global-flip invariance (broken by
   a longitudinal field); merges conjugate `S^z` sectors and splits by
   flip parity.
4. **Reality / time-reversal** — blocks with no residual imaginary
   part use the real symmetric LAPACK path.

The largest dense block — the cubic-cost driver — shrinks
dramatically. For the `N = 12` Heisenberg ring it drops from `4096`
to `66`. Benchmark it yourself:

```bash
python scripts/benchmark_symmetry.py --max-n 12
```

Symmetry blocking is always on by default. There is no physical
correctness caveat: every sector is summed back.

---

## 4. Legacy fitter integration

The COBYLA-based fitter
[`qed_nlce/analysis/nlc_fit_triangular.py`](../qed_nlce/analysis/nlc_fit_triangular.py)
builds CLI lines for the
[`qed_nlce/run/nlce_triangular.py`](../qed_nlce/run/nlce_triangular.py)
shim, which drives the `full_ed` pipeline. In your fit config:

```python
fit_config = {
    "fixed_params": {
        "max_order": 7,
        # ...
    },
    "fit_params": {"J1": (0.8, 1.2), "Delta": (0.0, 1.5)},
}
```

---

## 5. HDF5 output schema

Every cluster writes
`<base_dir>/ed_results_order_<O>/cluster_<id>_order_<o>/output/ed_results.h5`:

| Path | Contents |
| --- | --- |
| `/eigendata/eigenvalues` | `(D,)` real, full sorted spectrum |
| `/thermodynamics/temperatures` | `(nT,)` real (with `--thermo`) |
| `/thermodynamics/energy` | `(nT,)` real |
| `/thermodynamics/specific_heat` | `(nT,)` real |
| `/thermodynamics/entropy` | `(nT,)` real |
| `/thermodynamics/free_energy` | `(nT,)` real |

The NLCE summation kernels read `/thermodynamics/*` when present and
otherwise recompute thermo on the fly from
`/eigendata/eigenvalues`.

---

## 6. Resummation

`full_ed` accepts `--resummation=<METHOD>`. The two kernels implement
different native subsets:

* `NLC_sum_triangular.py` (triangular geometries):
  `{none, euler, wynn, wynn_multi, brezinski, aitken, pade, entropy_derived}`.
* `NLC_sum.py` (pyrochlore):
  `{auto, direct, euler, wynn, shanks, aitken, pade}`.

An alias map inside each kernel normalises cross-kernel names (e.g.
`theta` → `brezinski`, `none` → `direct`).

---

## 7. Parallelism — running many clusters at once

NLCE is embarrassingly parallel across clusters: each ED is
independent and the cluster sum happens only at the end.

### Knobs

| Flag | Effect |
| ---- | ------ |
| `--parallel` | Enable parallelism in cluster-prep / ED steps. |
| `--num_cores N` | Total core budget (default: all logical cores). |
| `--ed_parallel_workers W` | Workers for the ED step. Threads-per-worker auto-pinned to `max(1, num_cores // W)`. |

ED uses the **`spawn`** start method: each worker boots a fresh
interpreter and the pool initializer pins
`OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS =
max(1, num_cores // W)` before scientific libraries are imported, so
`W·T` threads never thrash `num_cores`.

### Rule of thumb

| Scenario | Setting |
| -------- | ------- |
| Many small clusters | `--ed_parallel_workers N` (1 thread/worker, max throughput) |
| Few large clusters dominating wall time | `--ed_parallel_workers small` (more BLAS threads each) |
| Mixed | `--ed_parallel_workers ≈ √N` |

### Cache safety under parallel workers

The on-disk eigenvalue cache is content-addressed (SHA-256 over the
Hamiltonian + ED options) and uses a write-then-`os.replace`
discipline: each worker stages into `<dst>.tmp.<pid>.<rand>` and
atomically renames into place; the meta JSON is written last so the
lookup gate never observes a half-finished entry. It is therefore
safe to point `--cache_dir` at a shared cache from many concurrent
jobs without external locking.

---

## 8. Putting it on a SLURM cluster

One job per field/parameter point, cluster-level parallelism inside
the node; all array tasks share one cache dir on the parallel
filesystem (the atomic-write discipline above makes this safe):

```bash
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

FIELDS=(0.0 0.5 1.0 2.0)
H=${FIELDS[$SLURM_ARRAY_TASK_ID]}

qed-nlce \
    --geometry triangular_site --pipeline full_ed --max_order 8 \
    --base_dir   $SCRATCH/fit/h_${H}/ \
    --cache_dir  $SCRATCH/qed_nlce_cache/triangular_o8/ \
    --J1 1.0 --h ${H} \
    --thermo --temp_min 0.1 --temp_max 30 --temp_bins 200 \
    --parallel --num_cores 32
```
