# QED_NLCE — Usage Guide

End-to-end reference for the unified `qed-nlce` CLI, all five pipelines
(`auto`, `full_ed`, `kpm_dos`, `ftlm`, `lanczos_boost`), the optional
fixed-Sz / streaming-symmetry axes, and integration with the legacy
fitter.

> **TL;DR** — for production runs at NLCE order ≥ 6, use
> `--pipeline auto`. It runs FULL dense ED on small clusters and
> switches to the C++ KPM-DOS thermodynamics solver above
> `2**N > 16384`, giving per-cluster `C(T)` error of `~4 × 10⁻⁴` at
> `N = 20` — well below the per-cluster target dictated by the NLCE
> Möbius condition number.

---

## 1. Pipeline cheat-sheet

| Pipeline | Per-cluster ED | Use when | Per-cluster `C(T)` noise |
| --- | --- | --- | --- |
| **`auto`** | FULL ↔ KPM-DOS at `2**N == 16384` | **Always, unless you have a reason** | `0` (FULL branch) ‖ `~4e-4` (KPM, `N=20`, `R=20`) |
| `full_ed` | dense LAPACK, all eigenvalues | small clusters / orders ≤ 5 | `0` |
| `kpm_dos` | C++ Chebyshev KPM + Hutchinson + Cheb-Gauss quadrature | large clusters, force iterative everywhere | `1/√(R·D)` |
| `ftlm` | Finite-Temperature Lanczos | back-compat only | `~5%` (amplified by Möbius `κ ~ 30–80` to `15–40%`) |
| `lanczos_boost` | Lanczos lowest-`k` | ground-state observables | n/a (no thermo) |

---

## 2. The `auto` pipeline (recommended default)

```bash
qed-nlce --geometry triangular_site --pipeline auto --max_order 8 \
         --J1 1.0 --temp_min 0.1 --temp_max 10 --temp_bins 100 --thermo \
         --base_dir output/tri_auto_o8
```

### What it does

For each cluster of size `N`:

| Condition | Backend |
| --- | --- |
| `2**N <= --auto_full_hilbert` (default `16384`) | `FULL` dense ED, *all eigenvalues* |
| `2**N >  --auto_full_hilbert` | `--auto_backend` (default `kpm_dos`) |

The `--thermo` switch makes both branches write data that the NLCE
summation kernel (`NLC_sum_ftlm.py`) consumes:

* `kpm_dos` writes `/thermodynamics/{temperatures, energy, specific_heat, entropy, free_energy}`.
* `full_ed` writes `/eigendata/eigenvalues`; the summation kernel
  has an eigenvalue fallback that reconstructs `Z, E, C(T), F, S`
  on the fly.

### Knobs

| Flag | Default | Effect |
| --- | --- | --- |
| `--auto_backend kpm_dos|ftlm` | `kpm_dos` | Iterative backend above the FULL ceiling. |
| `--auto_full_hilbert N` | `16384` | FULL-ED ceiling (`2**14`). |
| `--auto_kpm_moments M` | `2048` | KPM Chebyshev moments. |
| `--auto_kpm_random_vectors R` | `20` | KPM Hutchinson trace samples. |
| `--auto_kpm_kernel jackson|lorentz` | `jackson` | KPM smoothing kernel. |
| `--auto_kpm_seed S` | `0` | RNG seed (0 = nondeterministic). |
| `--auto_min_samples`, `--auto_max_samples` | `40, 200` | Adaptive FTLM range (only when `--auto_backend=ftlm`). |
| `--auto_fixed_sz` | off | See §4 below — partial trace, advanced. |
| `--auto_streaming_symmetry` | off | See §4 below — geometric orbit-basis decomposition. |

### When the default crossover is wrong

* **Pyrochlore at orders ≥ 6** routinely produces clusters with
  `N = 14–20`. Default `--auto_full_hilbert=16384` keeps `N ≤ 14` on
  FULL ED and pushes `N ∈ {15, 16, …, 20}` to KPM-DOS.
* If your clusters are unusually small or you have lots of RAM, raise
  the ceiling: `--auto_full_hilbert 65536` (`N ≤ 16`).
* For very large per-cluster size (e.g. `N ≥ 22`) bump KPM moments:
  `--auto_kpm_moments 4096`.

---

## 3. Forcing a single backend

### `full_ed`

```bash
qed-nlce --geometry pyrochlore --pipeline full_ed --max_order 5 \
         --J1 1.0 --thermo --base_dir output/pyro_full_o5
```

* `--method FULL | FULL_GPU` — Pick GPU LAPACK if your `qed` build
  was compiled with CUDA.
* `--symmetrized` — Use the strongest symmetry guarantee the
  geometry exposes.
* `--measure_spin` — Forward `<S^z_i S^z_j>`-style observables.

### `kpm_dos`

```bash
qed-nlce --geometry triangular_site --pipeline kpm_dos --max_order 8 \
         --J1 1.0 --thermo \
         --kpm_moments 2048 --kpm_random_vectors 20 \
         --kpm_kernel jackson \
         --base_dir output/tri_kpm_o8
```

* `--kpm_moments M` — Chebyshev moments (default 2048). Cost scales
  `O(M·R·D)` matrix-vector products. Doubling `M` halves the kernel
  bandwidth.
* `--kpm_random_vectors R` — Hutchinson stochastic-trace samples
  (default 20). Variance scales `1/√(R·D)`, so very large `R` is
  unnecessary at `D ≥ 2**16`.
* `--kpm_quadrature_nodes Q` — Cheb-Gauss quadrature nodes
  (default `0` → auto-`2M`).
* `--kpm_kernel jackson|lorentz` — `jackson` (default) is positive
  and minimises Gibbs ringing; `lorentz` (with `--kpm_lorentz_lambda`,
  default 4) is provably positive at all `M` but slightly broader.

### `ftlm` (legacy)

```bash
qed-nlce --geometry pyrochlore --pipeline ftlm --max_order 5 \
         --J1 1.0 --thermo \
         --ftlm_samples 30 --krylov_dim 200 \
         --base_dir output/pyro_ftlm_o5
```

Kept for back-compat. The `~5%` per-cluster `C(T)` noise floor is
amplified by the NLCE Möbius condition number to `15–40%` on the
summed curve at orders ≥ 6 — prefer `auto` or `kpm_dos`.

### `lanczos_boost`

```bash
qed-nlce --geometry pyrochlore --pipeline lanczos_boost --max_order 5 \
         --J1 1.0 --measure_spin \
         --base_dir output/pyro_lz_o5
```

Lowest-`k` eigenvalues only (no thermodynamics). Useful for
ground-state correlators, spin-spin structure factors, BEC-style order
parameters.

---

## 4. Symmetry & U(1) Sz blocking

Two **orthogonal** axes that shrink the per-cluster Hilbert space:

### `--auto_streaming_symmetry` (always physically correct)

Exploits the geometric automorphism group of each cluster. The
framework constructs an orbit basis once per cluster, caches it under
`<ham_dir>/basis_cache/`, and the C++ dispatcher diagonalises one
sector at a time. **All sectors are summed back together** — physical
results are unchanged, only the per-cluster cost goes down.

```bash
qed-nlce --geometry triangular_site --pipeline auto --max_order 8 \
         --J1 1.0 --thermo --auto_streaming_symmetry \
         --base_dir output/tri_auto_sym_o8
```

### `--auto_fixed_sz` (partial trace — advanced)

Asserts the Hamiltonian commutes with `S^z_total`. Routes every
cluster through the dispatcher with `params.use_fixed_sz = True`,
which currently selects the *single* `Sz = 0` block in the C++
backend. **This is a partial trace, not a full thermo trace.**

Use only when:

1. Your model has *no* transverse field, *no* `Sx`/`Sy` single-site
   anisotropy, *no* DM, *no* Kitaev — i.e. it actually conserves
   `S^z_total`. (Pure Heisenberg / XXZ / Ising-with-longitudinal-field
   are fine. Anything with `h_x, h_y` is not.)
2. You explicitly want the `Sz = 0` partial trace (e.g. studying a
   plateau or restricting to a specific magnetisation).

For full unconstrained thermodynamics, leave this off. Sector-summing
support (loop over all `Sz` blocks) is on the roadmap.

```bash
# Sz = 0 partial trace, with geometric symmetry on top:
qed-nlce --geometry triangular_site --pipeline auto --max_order 8 \
         --J1 1.0 --thermo --auto_streaming_symmetry --auto_fixed_sz \
         --base_dir output/tri_auto_sym_sz0_o8
```

---

## 5. Legacy fitter integration

The COBYLA-based fitter
[`qed_nlce/analysis/nlc_fit_triangular.py`](qed_nlce/analysis/nlc_fit_triangular.py)
builds CLI lines for the [`qed_nlce/run/nlce_triangular.py`](qed_nlce/run/nlce_triangular.py)
shim, which auto-promotes `--method` onto the matching pipeline:

| `fixed_params["ed_method"]` | shim picks |
| --- | --- |
| `"AUTO"` | `--pipeline=auto` |
| `"KPM_DOS"`, `"KPM"` | `--pipeline=kpm_dos` |
| `"FTLM"`, `"FTLM_GPU"`, `"LTLM"` | `--pipeline=ftlm` |
| `"FULL"`, `"FULL_GPU"`, anything else | `--pipeline=full_ed` (forwards `--method`) |

So in your fit config:

```python
fit_config = {
    "fixed_params": {
        "ed_method": "AUTO",   # <-- triggers pipeline=auto
        "max_order": 7,
        # ...
    },
    "fit_params": {"J1": (0.8, 1.2), "Delta": (0.0, 1.5)},
}
```

You can also pass any of the `--auto_*` knobs through the fitter's
`extra_args` field if exposed; otherwise the shim's defaults apply.

---

## 6. HDF5 output schema

Every cluster writes `<base_dir>/ed_results_order_<O>/cluster_<id>_order_<o>/output/ed_results.h5`:

| Path | Written by | Contents |
| --- | --- | --- |
| `/eigendata/eigenvalues` | `full_ed`, `lanczos_boost` | `(D,)` or `(k,)` real |
| `/thermodynamics/temperatures` | `kpm_dos`, `ftlm` (with `--thermo`) | `(nT,)` real |
| `/thermodynamics/energy` | same | `(nT,)` real |
| `/thermodynamics/specific_heat` | same | `(nT,)` real |
| `/thermodynamics/entropy` | same | `(nT,)` real |
| `/thermodynamics/free_energy` | same | `(nT,)` real |
| `/ftlm/averaged/*` | `ftlm` only | per-temp arrays + `*_error` companions |

The summation kernel `NLC_sum_ftlm.py` accepts any of:

1. `/ftlm/averaged/*` (preferred for FTLM clusters).
2. `/thermodynamics/*` (preferred for KPM-DOS / FULL-with-thermo clusters).
3. `/eigendata/eigenvalues` fallback (computes thermo on the fly for
   FULL-ED clusters that didn't materialise `/thermodynamics/`).

This three-way fallback is what lets the `auto` pipeline mix FULL-ED
small clusters with KPM-DOS large clusters in a single NLCE summation.

---

## 7. Validation snippets

### Bit-exact match: auto (FULL branch only) vs full_ed

```bash
qed-nlce --geometry triangular_site --pipeline full_ed --max_order 6 \
         --J1 1.0 --thermo --base_dir /tmp/ref_full_o6
qed-nlce --geometry triangular_site --pipeline auto    --max_order 6 \
         --J1 1.0 --thermo --auto_full_hilbert 4096 --base_dir /tmp/ref_auto_o6
python - <<'PY'
import numpy as np
a = np.loadtxt('/tmp/ref_full_o6/nlc_results_order_6/nlc_specific_heat.txt')
b = np.loadtxt('/tmp/ref_auto_o6/nlc_results_order_6/nlc_specific_heat.txt')
print('median |dC|/|C|:', np.median(np.abs(a[:,1]-b[:,1])/np.maximum(np.abs(a[:,1]),1e-3)))
# expect ~1e-13
PY
```

### Auto crossover at large `N`

```bash
qed-nlce --geometry pyrochlore --pipeline auto --max_order 7 \
         --J1 1.0 --thermo \
         --auto_kpm_moments 2048 --auto_kpm_random_vectors 30 \
         --base_dir /tmp/pyro_auto_o7
```

Cluster log will show FULL ED for `N ≤ 14` and `KPM_DOS` for `N ≥ 15`.

---

## 8. Performance / sizing rules of thumb

| Cluster `N` | Hilbert `D` | Pipeline (default `auto`) | Per-cluster wall (single 16-core node) |
| --- | --- | --- | --- |
| ≤ 12 | ≤ 4096 | FULL ED | `< 1 s` |
| 13–14 | ≤ 16384 | FULL ED | `~10–60 s` |
| 15–16 | 32768–65536 | KPM-DOS, `M=2048`, `R=20` | `~30 s – 3 min` |
| 17–18 | 131072–262144 | KPM-DOS | `~2–10 min` |
| 19–20 | 524288–1048576 | KPM-DOS, consider `M=4096` | `~10 min – 1 h` |
| ≥ 21 | ≥ 2097152 | KPM-DOS, **strongly** consider `--auto_streaming_symmetry` | `> 1 h` |

KPM-DOS variance scales as `1/√(R·D)`, so per-cluster relative error
on `C(T)` *improves* as the cluster grows — `R = 20` is sufficient at
`N ≥ 16` and would only need bumping to `R = 40` for a `2× ` margin
at `N ≤ 14`.

NLCE Möbius condition number `κ ~ 30–80` is the crucial sanity check
on any iterative backend: per-cluster relative error must be
`<< 1 / κ ≈ 1–3%` to keep summed-`C(T)` error `< 30%`. KPM-DOS clears
this comfortably; FTLM does not.

---

## 9. Resummation

All four pipelines (`auto`, `full_ed`, `kpm_dos`, `ftlm`) accept
`--resummation=<METHOD>`. The full unified vocabulary is:

| Name | Family | Notes |
| --- | --- | --- |
| `auto` | meta | Pick best method by convergence diagnostics. |
| `none`, `direct` | none | No acceleration; use highest order directly. |
| `euler` | Euler | Best for alternating series. |
| `wynn`, `shanks` | Wynn-ε | Default series-acceleration choice. |
| `wynn_multi` | Wynn-ε | Multi-step Wynn (triangular kernel only). |
| `pade` | Padé | Triangular kernel only; needs ≥ 4 orders. |
| `aitken` | Aitken Δ² | Triangular kernel only. |
| `theta`, `brezinski` | Brezinski-θ | |
| `robust` | meta | Run several methods, cross-validate. |
| `entropy_derived` | physics | Triangular only; derives `C(T)` from `S(T)`. |

Cross-pipeline aliases are normalised inside each kernel — e.g.
`--resummation=none` is accepted by `NLC_sum_ftlm.py` (mapped to
`direct`), and `--resummation=theta` is accepted by
`NLC_sum_triangular.py` (mapped to `brezinski`). You can use the same
method-name vocabulary regardless of which pipeline runs underneath.

The pyrochlore kernel (`NLC_sum.py`) and triangular kernel
(`NLC_sum_triangular.py`) implement different native subsets; an alias
map inside each kernel covers the rest. For best fidelity stick to
each kernel's native vocabulary:

* `NLC_sum_ftlm.py` (used by `auto`, `kpm_dos`, `ftlm`):
  native = {auto, direct, euler, wynn, theta, robust}.
* `NLC_sum_triangular.py` (used by `full_ed` on triangular):
  native = {none, euler, wynn, wynn_multi, brezinski, aitken, pade, entropy_derived}.
* `NLC_sum.py` (used by `full_ed` on pyrochlore):
  native = {auto, direct, euler, wynn, shanks, aitken, pade}.


## 10. Parallelism — running many clusters at once

NLCE is *embarrassingly parallel* across clusters: each ED is
independent, and the cluster sum is performed only at the very end.
`qed-nlce` exposes within-node multiprocessing parallelism for the
heavy steps; for cross-node scaling, wrap the CLI in a SLURM array.

### Knobs

| Flag | Effect |
| ---- | ------ |
| `--parallel`            | Enable parallelism in cluster-prep / basis-precompute / ED steps. |
| `--num_cores N`         | Total core budget (default: all logical cores). |
| `--ed_parallel_workers W` | Override # workers for the ED step (default: `--num_cores`). Threads-per-worker auto-pinned to `max(1, num_cores // W)`. |

### What runs in parallel

| Step | Parallelised? | Backend |
| ---- | ------------- | ------- |
| 1. cluster generation | depends on geometry (some are serial) | `multiprocessing.Pool` (fork) |
| 2. Hamiltonian prep   | sequential (cheap) | — |
| 2.5. basis precompute | yes (when `--streaming-symmetry`) | `multiprocessing.Pool` (fork) |
| 3. ED                 | yes (NEW) | `multiprocessing.Pool` (`spawn`) |
| 4. NLCE summation     | sequential (cheap) | — |

ED uses the **`spawn` start method**, not `fork`. Forking after the
parent has loaded `qed`, MKL, OpenMP, or CUDA is unsafe — MKL spawns
worker threads that don't survive `fork(2)`, and CUDA driver state is
per-process. Spawning means each worker boots a fresh interpreter and
imports `qed` lazily on first call, by which point the pool initializer
has already pinned the BLAS/OMP thread counts. The cost is one
~0.3 s startup per worker, paid once.

### Why threads-per-worker matters

Naively running `W` workers each with `T = num_cores` BLAS threads
gives you `W·T` total threads chasing `num_cores` cores — pure
thrashing. The pool initializer therefore sets

```
OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS
                = max(1, num_cores // W)
```

inside every worker before `qed` is imported. To override, set
`OMP_NUM_THREADS` in your shell before launching `qed-nlce` (the
initializer only runs after env capture).

### Rule of thumb

| Scenario | Recommended setting |
| -------- | ------------------- |
| Many small clusters (order ≤ 8, FULL branch) | `--num_cores N` (default), `--ed_parallel_workers N` (1 thread/worker, max throughput) |
| Few large clusters dominating wall time | `--ed_parallel_workers small`, e.g. `2` on 16 cores → 8 BLAS threads each, MKL stays effective |
| Mixed workload | `--ed_parallel_workers ≈ √N` is a reasonable starting point |

### Cache safety under parallel workers

The on-disk eigenvalue cache is **content-addressed** (SHA-256 over
the Hamiltonian + ED options). Two parallel workers computing the
same digest produce byte-identical content, but a naive `shutil.copy2`
into `<cache>/<digest>.h5` would still tear under concurrent writes.

`EigenvalueCache` therefore uses a write-then-`os.replace` discipline:
each worker stages its result into a sibling
`<dst>.tmp.<pid>.<rand>` (file) or `<dst>.tmp.<pid>.<rand>` (dir),
then atomically renames into place. `os.replace` is atomic on POSIX
(`rename(2)`). The meta JSON is written **last**, so the lookup gate
(which keys cache-readiness on the meta file's existence) never
observes a half-finished entry. The `SubclusterCache` uses the same
pattern.

In short: it is safe to point `--cache_dir` at a shared cache from
many concurrent `qed-nlce` jobs — even on the same digest — without
any external locking. Worst case is the loser of a race overwrites
the winner with the same bytes.

### GPU caveat

The KPM-DOS kernel currently runs on CPU only; the FULL branch can
optionally use cuSOLVER (`*_GPU`). Combining `--parallel` with a
GPU-backed method is **not supported** out of the box: every worker
would target GPU 0 and contend on a single device. If you need
multi-GPU scaling, set `CUDA_VISIBLE_DEVICES` per worker manually
(e.g. via a SLURM array, one job per GPU).

### How does this compare to SOTA NLCE codes?

Three patterns are common in the literature:

1. **SLURM job arrays — one ED per job.**  Used by Singh / Oitmaa &
   collaborators. Trivially scales to thousands of nodes; coordination
   is the filesystem itself; no per-job parallelism needed beyond
   what one ED solver provides. `qed-nlce` supports this naturally:
   loop over cluster files in a shell wrapper, point each invocation
   at a different `--base_dir`.
2. **MPI master/worker farm.**  Used in Tang / Khatami / Rigol style
   pyrochlore NLCE codes — a single launcher hands clusters out via
   `MPI_Send` so that load balancing is dynamic across heterogeneous
   cluster sizes. Great for tightly-coupled HPC, but requires every
   node to have the same software stack.
3. **Within-node multiprocessing.Pool.**  Used by `qed-nlce`'s
   `--parallel` mode. Lower coordination cost (no MPI), zero
   filesystem chatter, and the cache shares state between workers via
   the same content-addressed directory. Best for single-node runs
   and the inner loop of a SLURM-array job (combine with #1 for
   multi-node throughput).

`qed-nlce` is therefore aligned with current best practice: pattern
#3 within a node, with pattern #1 as the recommended way to scale
beyond one node.


## 11. Worked example — high-order multi-field fit

This is the canonical "production" workload: a COBYLA fit of a few
exchange parameters against several measured `C(T,h)` curves at
**order 8 or 9**, where each NLCE evaluation is dominated by ED on
the largest two cluster orders. It exercises every piece of §10
(parallel ED + atomic cache + spawn safety) plus the fitter's own
outer-loop parallelism.

### Layout of the parallelism

The legacy fitter
[`qed_nlce/analysis/nlc_fit_triangular.py`](qed_nlce/analysis/nlc_fit_triangular.py)
exposes **two independent levels** of parallelism:

```
COBYLA iteration k
 └─ parameters {J1, Delta, ...}_k
     ├─ field h_1 ──► qed-nlce subprocess  ┐
     ├─ field h_2 ──► qed-nlce subprocess  │  --parallel_fields
     ├─        ⋮                           │  (process per field)
     └─ field h_M ──► qed-nlce subprocess  ┘
                       │
                       └─ inside each subprocess:
                          cluster_1 ─┐
                          cluster_2 ─┤  --parallel_ed
                              ⋮       ├  (process per cluster, spawn)
                          cluster_K ─┘
```

The nested layout makes core-budgeting non-trivial; the fitter
auto-divides on your behalf:

```
ed_cores_per_field = total_cores // n_fields_in_flight   (when both flags set)
threads_per_ed_worker = ed_cores_per_field // ed_workers
```

with both factors floored at 1 — so on a 64-core node fitting 4
fields at order 8 you get 16 ED workers per field, 1 BLAS thread per
worker, 64 saturated threads total and zero oversubscription.

### Recommended config for order ≥ 8

```python
fit_config = {
    "fixed_params": {
        # --- physics ---
        "max_order": 8,
        "ed_method": "AUTO",          # see §2 / §3
        "auto_full_hilbert": 16384,   # FULL up to N=14, KPM-DOS above
        "auto_streaming_symmetry": True,   # space-group projection
        "auto_fixed_sz": True,             # +U(1) Sz=0 partial trace

        # --- caching: KEY for fits ---
        # Each COBYLA step changes the Hamiltonian, so the eigenvalue
        # cache MISSES on the new parameter point. BUT: the
        # subcluster-cache HITS every iteration (topology is fixed),
        # and the eigenvalue-cache hits if COBYLA revisits a point
        # (it does, especially near the optimum). Pin the cache dir
        # to a project-scoped path so it survives `rm -rf base_dir`:
        "cache_dir": "/scratch/$USER/qed_nlce_cache/triangular_o8",

        # --- skip cluster-gen across fits (topology is invariant) ---
        "skip_cluster_gen": True,

        # --- two-level parallelism ---
        "parallel_fields": True,   # CLI: --parallel_fields
        "parallel_ed":     True,   # CLI: --parallel_ed
        # ed_num_cores omitted ⇒ auto = total_cores // n_fields
    },
    "fit_params": {
        "J1":    (0.5, 1.5),
        "Delta": (0.0, 1.5),
        "Jpm":   (-0.3, 0.3),
    },
}
```

Equivalent CLI invocation (skipping the Python harness):

```bash
nlce-fit-triangular \
    --max_order 8 \
    --parallel_fields --max_parallel_fields 4 \
    --parallel_ed                                  \
    --ed_method AUTO --auto_streaming_symmetry --auto_fixed_sz \
    --cache_dir /scratch/$USER/qed_nlce_cache/triangular_o8 \
    --skip_cluster_gen \
    --datasets exp_C_h0p0.txt exp_C_h0p5.txt exp_C_h1p0.txt exp_C_h2p0.txt
```

### What the cache buys you

For a fit of order 8 with 4 fields and ~50 COBYLA iterations:

| What | Cache behaviour | Speedup |
| --- | --- | --- |
| Subcluster topology (§6 of cache.py) | **always hit** (topology fixed) | skip step 2 entirely after iter 1 |
| Eigenvalues at a new parameter point | **miss** (digest changes with `J1`, `Delta`, `Jpm`) | full ED required |
| Eigenvalues at a *revisited* parameter point | **hit** (COBYLA's polytope reflection often probes same vertex twice) | typically 5–15 % of total ED calls |
| Re-running same fit with `--restart` | **all hit** | fit converges in seconds |

For a *parameter sweep* (not a fit) the cache hit rate is much
higher: only the parameter being swept changes the digest, and the
basis-precompute output is shared across the entire sweep.

### Putting it on a SLURM cluster

Combine §10 patterns #1 and #3 — one SLURM job per field, ED
parallelism inside the node:

```bash
#SBATCH --array=0-3            # one task per field
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

FIELDS=(0.0 0.5 1.0 2.0)
H=${FIELDS[$SLURM_ARRAY_TASK_ID]}

# All array tasks share the SAME cache dir on the parallel filesystem
# — the atomic-write discipline (§10) makes this safe.
qed-nlce \
    --geometry triangular_site --pipeline auto --max_order 8 \
    --base_dir   $SCRATCH/fit/h_${H}/ \
    --cache_dir  $SCRATCH/qed_nlce_cache/triangular_o8/ \
    --J1 ${J1} --Delta ${Delta} --h ${H} \
    --auto_streaming_symmetry --auto_fixed_sz \
    --thermo --temp_min 0.1 --temp_max 30 --temp_bins 200 \
    --parallel --num_cores 32
```

Wrap this in your fitter's outer loop (write parameters, sbatch, wait
on `squeue`, harvest `nlc_specific_heat.txt`) and you have a setup
that scales to thousands of nodes with zero coordination beyond the
shared cache directory.

### Pitfalls

* **`--parallel_ed` + `*_GPU` method**: see §10 GPU caveat. The fitter
  passes `ed_method` straight through, so picking `FULL_GPU` here
  will deadlock `n_fields × ed_workers` Python processes against a
  single GPU. Use CPU `AUTO` or pin GPUs via `CUDA_VISIBLE_DEVICES`.
* **`--cache_dir` on slow NFS**: HDF5 reads of ~MB cluster files
  dominate at order ≥ 8. Use local SSD (`/tmp`, `/scratch`) and let
  one SLURM-array job per node populate it; subsequent jobs benefit.
* **Stale `skip_cluster_gen=True` after upgrading the cluster
  generator**: the subcluster-cache key includes a generator-version
  field — after a `qed_nlce` version bump, the topology cache safely
  invalidates itself, but you'll see one slow iteration as it
  rebuilds.
