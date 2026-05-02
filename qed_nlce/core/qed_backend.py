"""In-process QED backend: route ED step through ``qed.exact_diagonalization_from_directory``.

When the optional ``qed`` Python package is importable, NLCE pipelines
can run each cluster's diagonalization *in the same Python process*
instead of forking ``./ED`` per cluster. This eliminates the per-cluster
fork + OpenMP/CUDA initialization overhead that dominates wall-time when
the workflow has hundreds of small clusters at high orders.

Capability matrix (Phase 7+ canonical 5-axis dispatcher):

==========================  ==============  =====================
NLCE EDOptions.method       in-proc method  EDParameters flags
==========================  ==============  =====================
FULL                        FULL            (none)
FULL_GPU                    FULL            use_gpu=True
SCALAPACK_MIXED             SCALAPACK       use_mpi=True (rejected: no in-proc MPI)
LANCZOS                     LANCZOS
LANCZOS_GPU                 LANCZOS         use_gpu=True
FTLM                        FTLM
FTLM_GPU                    FTLM            use_gpu=True
mTPQ                        mTPQ
mTPQ_CUDA / mTPQ_GPU        mTPQ            use_gpu=True
==========================  ==============  =====================

MPI methods (``SCALAPACK*``, ``mTPQ_MPI``) cannot run in-process from
Python (an interpreter cannot host ``MPI_Init``); we transparently fall
back to the ``./ED`` subprocess for those.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .ed_runner import EDOptions


def qed_available() -> bool:
    """True iff the ``qed`` Python package is importable."""
    try:
        import qed  # noqa: F401
        return True
    except Exception:
        return False


# Method names that have a clean in-process Python equivalent.
_INPROC_METHODS = {
    "FULL", "FULL_GPU",
    "LANCZOS", "LANCZOS_GPU", "LANCZOS_NO_ORTHO", "LANCZOS_SELECTIVE",
    "BLOCK_LANCZOS", "BLOCK_LANCZOS_GPU",
    "KRYLOV_SCHUR", "KRYLOV_SCHUR_GPU",
    "BLOCK_KRYLOV_SCHUR", "BLOCK_KRYLOV_SCHUR_GPU",
    "DAVIDSON", "DAVIDSON_GPU",
    "LOBPCG", "LOBPCG_GPU",
    "FTLM", "FTLM_GPU",
    "LTLM",
    "mTPQ", "mTPQ_CUDA", "mTPQ_GPU", "cTPQ", "cTPQ_GPU",
    "ARPACK_LM", "ARPACK_SM",
    "CHEBYSHEV_FILTERED",
}

# Methods that MUST go through ./ED (require MPI bootstrap).
_MPI_ONLY_METHODS = {
    "SCALAPACK", "SCALAPACK_MIXED", "mTPQ_MPI",
}


def can_run_in_process(method: str) -> bool:
    """True iff ``method`` is supported by the in-process backend.

    Returns False for MPI-only methods and unknown method names so the
    workflow can transparently fall through to the ``./ED`` subprocess.
    """
    m = method.upper().replace("MTPQ", "mTPQ").replace("CTPQ", "cTPQ")
    if m in _MPI_ONLY_METHODS:
        return False
    return m in _INPROC_METHODS


def _resolve_method(method_name: str):
    """Map an NLCE method string to (DiagonalizationMethod, use_gpu, use_fixed_sz)."""
    import qed  # local import: only needed when actually dispatching

    m = method_name.upper()
    use_gpu = False
    use_fixed_sz = False

    # Strip _GPU/_CUDA suffix and remember it.
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

    if base.endswith("_FIXED_SZ"):
        use_fixed_sz = True
        base = base[: -len("_FIXED_SZ")]

    try:
        method_enum = getattr(qed.DiagonalizationMethod, base)
    except AttributeError as exc:
        raise ValueError(
            f"qed.DiagonalizationMethod has no value '{base}' "
            f"(derived from NLCE method '{method_name}')"
        ) from exc
    return method_enum, use_gpu, use_fixed_sz


def run_ed_in_process(
    ham_subdir: str,
    output_dir: str,
    num_sites: int,
    options: EDOptions,
    *,
    log_tag: str = "ED-inproc",
) -> bool:
    """Run a single cluster's ED in-process via ``qed.exact_diagonalization_from_directory``.

    Returns True on success, False otherwise. Mirrors the contract of
    :func:`qed_nlce.core.run_ed_subprocess`.
    """
    try:
        import qed
    except ImportError:
        logging.error("[%s] qed package not importable -- cannot run in-process.", log_tag)
        return False

    method_enum, use_gpu, use_fixed_sz = _resolve_method(options.method)

    # Build-introspection guard (cheap, helpful diagnostics).
    if use_gpu and not qed.has_cuda_build():
        logging.error(
            "[%s] requested GPU method '%s' but qed build has no CUDA support",
            log_tag, options.method,
        )
        return False

    params = qed.EDParameters()
    params.num_sites = num_sites
    params.spin_length = options.spin_length
    params.use_gpu = use_gpu
    params.use_mpi = False
    params.use_fixed_sz = use_fixed_sz
    params.use_symmetry = options.streaming_symmetry

    # Eigenvalue count: NLCE FullED needs the entire spectrum; pass -1 to
    # request "all" (matches CLI ``--eigenvalues=FULL``). LANCZOS / FTLM
    # default to 1 / not-applicable.
    if options.eigenvalues is None:
        pass  # use library default
    elif str(options.eigenvalues).upper() == "FULL":
        params.num_eigenvalues = 2 ** num_sites  # request entire spectrum
        params.compute_eigenvectors = False
    elif str(options.eigenvalues).upper() == "LOWEST":
        params.num_eigenvalues = 1
    else:
        try:
            params.num_eigenvalues = int(options.eigenvalues)
        except (TypeError, ValueError):
            pass

    # Thermodynamics: the C++ dispatcher writes thermo files into output_dir
    # automatically when temp grid is set + method supports it.
    if options.thermo:
        params.temp_min = options.temp_min
        params.temp_max = options.temp_max
        params.num_temp_bins = options.temp_bins

    if options.samples is not None:
        params.num_samples = options.samples
    if options.krylov_dim is not None:
        params.ftlm_krylov_dim = options.krylov_dim
        params.ltlm_krylov_dim = options.krylov_dim

    out_subdir = os.path.join(output_dir, "output")
    os.makedirs(out_subdir, exist_ok=True)
    params.output_dir = out_subdir

    try:
        qed.exact_diagonalization_from_directory(
            ham_subdir,
            method=method_enum,
            params=params,
            format=qed.HamiltonianFileFormat.STANDARD,
        )
        return True
    except Exception as exc:
        logging.error("[%s] in-process ED failed: %s", log_tag, exc, exc_info=True)
        return False


__all__ = [
    "qed_available",
    "can_run_in_process",
    "run_ed_in_process",
]
