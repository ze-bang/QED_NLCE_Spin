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
