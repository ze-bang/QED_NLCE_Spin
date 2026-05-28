"""In-process QED backend: dispatch each cluster's ED step through the
canonical ``qed.solve`` / ``qed.thermal`` Python verbs.

When the ``qed`` Python package is importable (it is a required runtime
dependency declared in ``pyproject.toml``), every NLCE pipeline runs
each cluster's diagonalization *in the same Python process* instead of
forking ``./ED`` per cluster. This eliminates the per-cluster fork +
OpenMP / CUDA initialization overhead that dominates wall-time when
the workflow has hundreds of small clusters at high orders.

Backend surface
---------------

This module targets the post-May-2026 QED "three verbs" Python API:

* :func:`qed.solve(H, ...)`   -- ground-state / spectrum (LANCZOS,
  BLOCK_LANCZOS, KRYLOV_SCHUR, FULL).
* :func:`qed.thermal(H, ...)` -- finite-T thermodynamics (FTLM, LTLM,
  KPM_DOS, mTPQ, cTPQ). Accepts an in-memory ``Operator`` *or* a
  directory path (auto-loads ``Trans.dat`` / ``InterAll.dat`` + the
  optional ``automorphism_results/`` symmetry generators).

The legacy ``qed.exact_diagonalization_from_directory`` /
``qed.HamiltonianFileFormat`` symbols were removed in the QED surface
unification; this backend uses the new verbs directly.

Capability matrix (Phase 7+ canonical 5-axis dispatcher)::

    ==========================  ==============  =====================
    NLCE EDOptions.method       in-proc verb    flag plumbing
    ==========================  ==============  =====================
    FULL                        solve(FULL)
    FULL_GPU                    solve(FULL)     device='gpu'
    FULL_FIXED_SZ               solve(FULL)     sz=N//2
    LANCZOS                     solve(LANCZOS)
    LANCZOS_GPU                 solve(LANCZOS)  device='gpu'
    BLOCK_LANCZOS[_GPU]         solve(BLOCK_LANCZOS) [device='gpu']
    KRYLOV_SCHUR[_GPU]          solve(KRYLOV_SCHUR)  [device='gpu']
    FTLM                        thermal(FTLM)
    FTLM_GPU                    thermal(FTLM)   device='gpu'
    LTLM                        thermal(LTLM)
    KPM_DOS                     thermal(KPM_DOS)
    mTPQ                        thermal(mTPQ)
    mTPQ_CUDA / mTPQ_GPU        thermal(mTPQ)   device='gpu'
    cTPQ[_GPU]                  thermal(cTPQ)   [device='gpu']
    *_FIXED_SZ                  + sz_min=sz_max=N//2 (Sz=0 only)
    *_SZ_DECOMPOSED             + use_sz_if_conserved (all sectors)
    streaming_symmetry          + use_symmetry_if_available=True
    ==========================  ==============  =====================

MPI methods (``SCALAPACK*``, ``mTPQ_MPI``) cannot run in-process from
Python (a Python interpreter cannot host ``MPI_Init``); they are
rejected by :func:`can_run_in_process` so the workflow surfaces a
clear error rather than crashing inside the binding.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import qed

from .ed_runner import EDOptions


def qed_available() -> bool:
    """True iff the ``qed`` Python package is importable.

    Always True under normal installations (``qed`` is a required
    dependency declared in ``pyproject.toml``). Kept as a function so
    downstream code can still defensively gate on it.
    """
    return True


# Method names that have a clean in-process Python equivalent. Anything
# in this set is routed through ``qed.solve`` (ground-state methods) or
# ``qed.thermal`` (FTLM / LTLM / KPM_DOS / mTPQ / cTPQ). The suffix
# stripper in ``_resolve_method`` handles the ``_GPU`` / ``_FIXED_SZ`` /
# ``_SZ_DECOMPOSED`` decorators below.
_INPROC_METHODS = {
    "FULL", "FULL_GPU", "FULL_FIXED_SZ", "FULL_GPU_FIXED_SZ",
    "FULL_SZ_DECOMPOSED", "FULL_GPU_SZ_DECOMPOSED",
    "LANCZOS", "LANCZOS_GPU",
    "LANCZOS_FIXED_SZ", "LANCZOS_GPU_FIXED_SZ",
    "LANCZOS_SZ_DECOMPOSED", "LANCZOS_GPU_SZ_DECOMPOSED",
    "BLOCK_LANCZOS", "BLOCK_LANCZOS_GPU",
    "BLOCK_LANCZOS_FIXED_SZ", "BLOCK_LANCZOS_GPU_FIXED_SZ",
    "BLOCK_LANCZOS_SZ_DECOMPOSED", "BLOCK_LANCZOS_GPU_SZ_DECOMPOSED",
    "KRYLOV_SCHUR", "KRYLOV_SCHUR_GPU",
    "KRYLOV_SCHUR_FIXED_SZ", "KRYLOV_SCHUR_GPU_FIXED_SZ",
    "KRYLOV_SCHUR_SZ_DECOMPOSED", "KRYLOV_SCHUR_GPU_SZ_DECOMPOSED",
    "FTLM", "FTLM_GPU", "FTLM_FIXED_SZ", "FTLM_GPU_FIXED_SZ",
    "FTLM_SZ_DECOMPOSED", "FTLM_GPU_SZ_DECOMPOSED",
    "LTLM", "LTLM_FIXED_SZ", "LTLM_SZ_DECOMPOSED",
    "KPM_DOS", "KPM_DOS_FIXED_SZ", "KPM_DOS_SZ_DECOMPOSED",
    "mTPQ", "mTPQ_CUDA", "mTPQ_GPU", "cTPQ", "cTPQ_GPU",
}

# Methods that MUST go through the standalone ``mpiexec
# ed_distributed_main`` binary (require MPI bootstrap).
_MPI_ONLY_METHODS = {
    "SCALAPACK", "SCALAPACK_MIXED", "mTPQ_MPI",
}

# Methods that route through ``qed.thermal``. Everything else in
# ``_INPROC_METHODS`` routes through ``qed.solve``.
_THERMAL_METHODS = {
    "FTLM", "LTLM", "KPM_DOS", "mTPQ", "cTPQ",
}


def can_run_in_process(method: str) -> bool:
    """True iff ``method`` is supported by the in-process backend.

    Returns False for MPI-only methods and unknown method names so the
    workflow can surface a clear error rather than silently fall
    through. The legacy ``./ED`` subprocess path no longer exists.
    """
    m = method.upper().replace("MTPQ", "mTPQ").replace("CTPQ", "cTPQ")
    if m in _MPI_ONLY_METHODS:
        return False
    return m in _INPROC_METHODS


def _resolve_method(method_name: str):
    """Map an NLCE method string to ``(DiagonalizationMethod, use_gpu,
    use_fixed_sz, use_sz_decomposed)``.

    Strips the optional ``_GPU`` / ``_CUDA``, ``_FIXED_SZ``, and
    ``_SZ_DECOMPOSED`` suffixes (orthogonal axes of the canonical
    5-axis dispatcher) and looks the base name up on
    :class:`qed.DiagonalizationMethod`.

    ``_FIXED_SZ``: restrict to the single Sz=0 sector (n_up=N//2).
    ``_SZ_DECOMPOSED``: iterate ALL Sz sectors individually and
    recombine — gives the correct full partition function while still
    benefiting from block-diagonal structure (each block has dimension
    C(N,k) instead of 2^N).

    Raises :class:`ValueError` with a helpful message when the base
    name is not bound (e.g. legacy methods that did not survive the
    May-2026 surface collapse).
    """
    m = method_name.upper()
    use_gpu = False
    use_fixed_sz = False
    use_sz_decomposed = False

    # Strip Sz suffixes first (they come last in the canonical name).
    if m.endswith("_SZ_DECOMPOSED"):
        use_sz_decomposed = True
        m = m[: -len("_SZ_DECOMPOSED")]
    elif m.endswith("_FIXED_SZ"):
        use_fixed_sz = True
        m = m[: -len("_FIXED_SZ")]

    # Strip _GPU / _CUDA suffix and remember it.
    if m.endswith("_GPU") or m == "MTPQ_CUDA":
        use_gpu = True
        if m == "MTPQ_CUDA":
            base = "mTPQ"
        elif m == "CTPQ_GPU":
            base = "cTPQ"
        else:
            base = m[: -len("_GPU")]
    elif m == "MTPQ":
        base = "mTPQ"
    elif m == "CTPQ":
        base = "cTPQ"
    else:
        base = m

    try:
        method_enum = getattr(qed.DiagonalizationMethod, base)
    except AttributeError as exc:
        raise ValueError(
            f"qed.DiagonalizationMethod has no value '{base}' "
            f"(derived from NLCE method '{method_name}'). "
            f"Available methods: "
            f"{sorted(qed.DiagonalizationMethod.__members__.keys())}"
        ) from exc
    return method_enum, use_gpu, use_fixed_sz, use_sz_decomposed


def _parse_kpm_extra_flags(
    extra_flags: list[str],
) -> tuple[dict[str, object], Optional[int]]:
    """Parse the ``--kpm_*`` flags emitted by the kpm_dos pipeline.

    Returns a tuple ``(extra_params_dict, random_seed)``. The dict
    fields map onto :class:`qed.EDParameters` attributes that
    ``qed.thermal(..., extra_params=...)`` forwards verbatim. The
    seed is returned separately because :func:`qed.thermal` exposes
    it as a top-level kwarg.
    """
    extra_params: dict[str, object] = {}
    random_seed: Optional[int] = None
    for flag in extra_flags or []:
        if not flag.startswith("--kpm_") or "=" not in flag:
            continue
        key, _, val = flag[2:].partition("=")
        if key == "kpm_kernel":
            extra_params["kpm_use_jackson_kernel"] = (
                val.lower() == "jackson"
            )
        elif key == "kpm_lorentz_lambda":
            try:
                extra_params["kpm_lorentz_lambda"] = float(val)
            except ValueError:
                pass
        elif key == "kpm_quadrature_nodes":
            try:
                extra_params["kpm_num_quadrature_nodes"] = int(val)
            except ValueError:
                pass
        elif key == "kpm_seed":
            try:
                random_seed = int(val)
            except ValueError:
                pass
    return extra_params, random_seed


def _resolve_num_eigenvalues(
    eigenvalues_spec: Optional[str],
    full_dim: int,
) -> tuple[int, bool]:
    """Translate the legacy ``eigenvalues`` knob (``"FULL"`` / ``"LOWEST"``
    / int-string / ``None``) into a ``(num_eigenvalues, is_full)`` pair
    consumed by :func:`qed.solve`.

    ``full_dim`` is the dimension of the Hilbert space *the solver will
    actually walk* (the full ``2**N`` for an unprojected operator, or
    ``C(N, n_up)`` when restricted to a fixed-Sz sector).
    """
    if eigenvalues_spec is None:
        return 1, False
    spec = str(eigenvalues_spec).upper()
    if spec == "FULL":
        # qed.solve(num_eigenvalues=dim, solver=FULL) -> entire spectrum.
        return max(int(full_dim), 1), True
    if spec == "LOWEST":
        return 1, False
    try:
        return max(int(eigenvalues_spec), 1), False
    except (TypeError, ValueError):
        return 1, False


def _wants_compute_eigenvectors(extra_flags: list[str]) -> bool:
    """True iff the legacy ``--compute_eigenvectors`` extra-flag is
    set. The ``lanczos_boost`` pipeline plumbs it through this channel
    so the persisted ``ed_results.h5`` carries the LANCZOS eigenvalues
    + eigenvectors for the per-eigenstate observable kernels."""
    for flag in extra_flags or []:
        if flag == "--compute_eigenvectors" or flag.startswith(
            "--compute_eigenvectors="
        ):
            return True
    return False


def _dispatch_thermal(
    ham_subdir: str,
    out_subdir: str,
    num_sites: int,
    options: EDOptions,
    method_enum: "qed.DiagonalizationMethod",
    device: str,
    n_up_fixed: Optional[int],
    *,
    sz_decomposed: bool = False,
) -> None:
    """Route a thermal lane through :func:`qed.thermal` (directory
    form). The C++ orchestrator reads ``Trans.dat`` /
    ``InterAll.dat`` (and ``automorphism_results/`` when
    ``use_symmetry_if_available=True``) and writes the aggregated
    thermodynamic curves into ``<out_subdir>/ed_results.h5``."""
    kwargs: dict[str, object] = dict(
        method=method_enum,
        T_min=float(options.temp_min),
        T_max=float(options.temp_max),
        num_T=int(options.temp_bins),
        num_sites=int(num_sites),
        spin=float(options.spin_length),
        output_dir=str(out_subdir),
        device=device,
        verbose=False,
        auto_tune=False,
        use_symmetry_if_available=bool(options.streaming_symmetry),
    )

    if sz_decomposed:
        # Full Sz decomposition: iterate ALL sectors [0, N] and let
        # qed.thermal recombine via _combine_sector_thermodynamics.
        # Correct for finite-T NLCE (full partition function).
        # Benefit: each block is C(N,k) instead of 2^N.
        kwargs["use_sz_if_conserved"] = True
        # Do NOT set sz_min/sz_max → defaults to full [0, N] window.
    elif n_up_fixed is not None:
        # Legacy single-sector (Sz=0 only). Approximate — valid at
        # low T where Z is dominated by the Sz=0 block.
        kwargs["sz_min"] = int(n_up_fixed)
        kwargs["sz_max"] = int(n_up_fixed)
        kwargs["use_sz_if_conserved"] = True
    else:
        # Default semantics: full-Hilbert single dispatch (legacy NLCE
        # behaviour). Skip Sz iteration even when the operator commutes
        # with S^z_total so the C++ writes a single ed_results.h5.
        kwargs["use_sz_if_conserved"] = False

    if options.samples is not None:
        kwargs["num_samples"] = int(options.samples)

    if method_enum == qed.DiagonalizationMethod.FTLM:
        if options.krylov_dim is not None:
            kwargs["ftlm_krylov_dim"] = int(options.krylov_dim)
    elif method_enum == qed.DiagonalizationMethod.LTLM:
        if options.krylov_dim is not None:
            kwargs["ltlm_krylov_dim"] = int(options.krylov_dim)
    elif method_enum == qed.DiagonalizationMethod.KPM_DOS:
        if options.samples is not None:
            kwargs["kpm_num_random_vectors"] = int(options.samples)
        if options.krylov_dim is not None:
            kwargs["kpm_num_moments"] = int(options.krylov_dim)
        kpm_extra, kpm_seed = _parse_kpm_extra_flags(options.extra_flags)
        if kpm_seed is not None:
            kwargs["random_seed"] = int(kpm_seed)
        if kpm_extra:
            # ``qed.thermal`` forwards ``extra_params`` onto the
            # internal ``EDParameters`` via setattr (Jackson vs Lorentz
            # kernel, Lorentz lambda, quadrature nodes).
            kwargs["extra_params"] = kpm_extra

    result = qed.thermal(ham_subdir, **kwargs)

    # When Sz-decomposed, qed.thermal iterates sectors in Python and
    # recombines Z, but the C++ layer only writes per-sector data to the
    # HDF5. We must persist the recombined thermodynamics so the NLCE
    # summation step can read them from the standard location.
    if sz_decomposed and result is not None:
        import h5py
        import numpy as np
        h5_path = os.path.join(out_subdir, "ed_results.h5")
        with h5py.File(h5_path, "a") as f:
            grp = f.require_group("thermodynamics")
            for key in ("temperatures", "energy", "specific_heat",
                        "entropy", "free_energy"):
                arr = getattr(result, key, None)
                if arr is None:
                    continue
                data = np.asarray(arr, dtype=np.float64)
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=data)
            grp.attrs["sz_decomposed"] = True
            grp.attrs["used_sz_decomposition"] = bool(
                getattr(result, "used_sz_decomposition", True)
            )


def _dispatch_ground_state(
    ham_subdir: str,
    out_subdir: str,
    num_sites: int,
    options: EDOptions,
    method_enum: "qed.DiagonalizationMethod",
    device: str,
    n_up_fixed: Optional[int],
) -> None:
    """Route a ground-state lane through :func:`qed.solve`.

    Loads ``Trans.dat`` / ``InterAll.dat`` into an in-memory
    :class:`qed.Operator`, then calls ``qed.solve(op, ...)``. The C++
    orchestrator persists the spectrum (and eigenvectors when
    ``compute_eigenvectors=True``) into
    ``<out_subdir>/ed_results.h5:/eigendata/``.
    """
    op = qed.Operator(
        num_sites=int(num_sites), spin=float(options.spin_length)
    )
    trans = os.path.join(ham_subdir, "Trans.dat")
    inter = os.path.join(ham_subdir, "InterAll.dat")
    if os.path.exists(trans):
        op.load_trans(trans)
    if os.path.exists(inter):
        op.load_inter_all(inter)

    sector_dim = (
        math.comb(int(num_sites), int(n_up_fixed))
        if n_up_fixed is not None
        else (1 << int(num_sites))
    )
    num_eigs, is_full = _resolve_num_eigenvalues(
        options.eigenvalues, sector_dim
    )

    compute_vecs = _wants_compute_eigenvectors(options.extra_flags)

    kwargs: dict[str, object] = dict(
        num_eigenvalues=int(num_eigs),
        solver=method_enum,
        device=device,
        output_dir=str(out_subdir),
        compute_eigenvectors=bool(compute_vecs),
        verbose=False,
        plan=False,
        auto_tune=False,
        # Legacy NLCE semantics: do not auto-project onto N//2 unless
        # the caller explicitly opted in via ``_FIXED_SZ``.
        auto_sz=False,
    )
    if n_up_fixed is not None:
        kwargs["sz"] = int(n_up_fixed)

    # Streaming symmetry: ``qed.solve(symmetry=...)`` takes an in-
    # memory ``GeneratorSet``. The legacy NLCE workflow expects the
    # generators to live alongside the cluster as
    # ``automorphism_results/`` and lets the C++ side load them.
    # ``qed.find_symmetries(op)`` is the canonical way to populate a
    # ``GeneratorSet`` for the in-memory operator; we use it when
    # ``streaming_symmetry`` is set so the dispatcher routes through
    # the orchestrator's streaming-symmetry kernel uniformly.
    if options.streaming_symmetry:
        try:
            report = qed.find_symmetries(op, verbose=False)
        except Exception as exc:
            logging.warning(
                "[qed_backend] find_symmetries failed (%s); "
                "falling back to the no-symmetry lane.",
                exc,
            )
            report = None
        if report is not None and report.full_set is not None \
                and len(report.full_set.generators) > 0:
            kwargs["symmetry"] = report.full_set

    qed.solve(op, **kwargs)


def _dispatch_full_sz_decomposed(
    ham_subdir: str,
    out_subdir: str,
    num_sites: int,
    options: EDOptions,
    method_enum: "qed.DiagonalizationMethod",
    device: str,
) -> None:
    """Full diagonalization of every Sz sector, saving the combined spectrum.

    For NLCE thermodynamics with Sz conservation: diagonalizes each Sz
    block individually (dimension C(N,k) instead of 2^N), concatenates
    all eigenvalues, and saves them to ``ed_results.h5:/eigendata/``.
    The summation step then computes the correct full partition function
    from the complete spectrum.
    """
    import h5py

    N = int(num_sites)
    op = qed.Operator(num_sites=N, spin=float(options.spin_length))
    trans = os.path.join(ham_subdir, "Trans.dat")
    inter = os.path.join(ham_subdir, "InterAll.dat")
    if os.path.exists(trans):
        op.load_trans(trans)
    if os.path.exists(inter):
        op.load_inter_all(inter)

    all_eigenvalues: list = []
    for n_up in range(N + 1):
        sector_dim = math.comb(N, n_up)
        if sector_dim == 0:
            continue
        result = qed.solve(
            op,
            num_eigenvalues=sector_dim,
            solver=method_enum,
            device=device,
            sz=n_up,
            verbose=False,
            plan=False,
            auto_tune=False,
            auto_sz=False,
        )
        eigs = result.eigenvalues
        if eigs is not None and len(eigs) > 0:
            all_eigenvalues.extend(
                eigs.tolist() if hasattr(eigs, 'tolist') else list(eigs)
            )

    if not all_eigenvalues:
        raise RuntimeError(
            f"FULL_SZ_DECOMPOSED: no eigenvalues obtained from any "
            f"Sz sector (N={N})."
        )

    all_eigenvalues.sort()
    import numpy as np
    eig_array = np.array(all_eigenvalues, dtype=np.float64)

    h5_path = os.path.join(out_subdir, "ed_results.h5")
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("eigendata")
        grp.create_dataset("eigenvalues", data=eig_array)
        grp.attrs["num_eigenvalues"] = len(eig_array)
        grp.attrs["hilbert_dim"] = 1 << N
        grp.attrs["num_sites"] = N
        grp.attrs["sz_decomposed"] = True
        grp.attrs["num_sectors"] = N + 1

    logging.info(
        "FULL_SZ_DECOMPOSED: N=%d, %d eigenvalues from %d sectors, "
        "E_gs=%.6f",
        N, len(eig_array), N + 1, eig_array[0],
    )


def run_ed_in_process(
    ham_subdir: str,
    output_dir: str,
    num_sites: int,
    options: EDOptions,
    *,
    log_tag: str = "ED-inproc",
    cache: Optional["object"] = None,
    cache_key_extras: Optional[dict] = None,
) -> bool:
    """Run a single cluster's ED in-process via the canonical
    ``qed.solve`` / ``qed.thermal`` Python verbs.

    Returns True on success, False otherwise. Mirrors the contract of
    the deleted ``./ED`` subprocess runner.

    If ``cache`` is provided (a :class:`qed_nlce.core.cache.EigenvalueCache`)
    the eigenvalue cache is consulted before running ED, and the result
    is persisted on success. ``cache_key_extras`` must contain the
    canonicalisation inputs (``geometry``, ``cluster_file``).
    """
    # ----- cache lookup -----
    cache_key = None
    if cache is not None and cache_key_extras is not None:
        try:
            cache_key = cache.compute_key(
                geometry=cache_key_extras["geometry"],
                ham_subdir=ham_subdir,
                cluster_file=cache_key_extras.get("cluster_file"),
                options=options,
                num_sites=num_sites,
            )
            if cache.lookup(cache_key, output_dir):
                return True
        except Exception as exc:  # cache must never break correctness
            logging.warning(
                "[%s] eigenvalue cache lookup failed: %s", log_tag, exc
            )
            cache_key = None

    try:
        method_enum, use_gpu, use_fixed_sz, use_sz_decomposed = (
            _resolve_method(options.method)
        )
    except ValueError as exc:
        logging.error("[%s] method resolution failed: %s", log_tag, exc)
        return False

    # Build-introspection guard (cheap, helpful diagnostics).
    if use_gpu and not qed.has_cuda_build():
        logging.error(
            "[%s] requested GPU method '%s' but qed build has no "
            "CUDA support (qed.has_cuda_build() == False).",
            log_tag, options.method,
        )
        return False

    out_subdir = os.path.join(output_dir, "output")
    os.makedirs(out_subdir, exist_ok=True)

    device = "gpu" if use_gpu else "cpu"

    # Sz-sector routing:
    # - ``use_fixed_sz``: legacy single-sector (Sz=0 only, n_up=N//2).
    #   Approximation valid at low T where Sz=0 dominates Z.
    # - ``use_sz_decomposed``: iterate ALL Sz sectors, recombine Z.
    #   Physically exact for finite-T NLCE. Each sector has dimension
    #   C(N,k) instead of 2^N — gives memory/compute savings while
    #   keeping the full partition function correct.
    n_up_fixed: Optional[int] = (
        int(num_sites) // 2 if use_fixed_sz else None
    )

    method_base = method_enum.name  # e.g. "FTLM", "FULL", ...
    is_thermal = method_base in _THERMAL_METHODS

    # When Sz-decomposed + thermal method (KPM_DOS, FTLM, etc.), route
    # through qed.thermal which handles sector iteration + Z-recombination.
    # For FULL_SZ_DECOMPOSED, use the dedicated per-sector solve path
    # that collects all eigenvalues (NLCE summation computes thermo from them).
    if use_sz_decomposed and method_base in _THERMAL_METHODS:
        is_thermal = True

    try:
        if use_sz_decomposed and method_base not in _THERMAL_METHODS:
            _dispatch_full_sz_decomposed(
                ham_subdir, out_subdir, int(num_sites), options,
                method_enum, device,
            )
        elif is_thermal:
            _dispatch_thermal(
                ham_subdir, out_subdir, int(num_sites), options,
                method_enum, device, n_up_fixed,
                sz_decomposed=use_sz_decomposed,
            )
        else:
            _dispatch_ground_state(
                ham_subdir, out_subdir, int(num_sites), options,
                method_enum, device, n_up_fixed,
            )
    except Exception as exc:
        logging.error(
            "[%s] in-process ED failed: %s", log_tag, exc, exc_info=True
        )
        return False

    # ----- cache store -----
    if cache is not None and cache_key is not None:
        try:
            cache.store(cache_key, output_dir)
        except Exception as exc:  # store failures are non-fatal
            logging.warning(
                "[%s] eigenvalue cache store failed: %s", log_tag, exc
            )
    return True


__all__ = [
    "qed_available",
    "can_run_in_process",
    "run_ed_in_process",
]
