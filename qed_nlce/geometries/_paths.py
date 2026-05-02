"""Internal: file-system paths to bundled scripts.

Resolved once, lazily. Exposed only to the geometry/pipeline modules
inside this subpackage; downstream callers should not import from
here.
"""

from __future__ import annotations

import os

# This file lives in qed_nlce/geometries/_paths.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# qed_nlce/
NLCE_DIR = os.path.dirname(_THIS_DIR)
# workflows/
WORKFLOWS_DIR = os.path.dirname(NLCE_DIR)
# repo root (exact_diagonalization_cpp/)
REPO_ROOT = os.path.dirname(WORKFLOWS_DIR)


# ---- prep/ scripts (cluster generators) ----

PREP_DIR = os.path.join(NLCE_DIR, "prep")

PYROCHLORE_GENERATOR = os.path.join(PREP_DIR, "generate_pyrochlore_clusters.py")
TRIANGULAR_SITE_GENERATOR = os.path.join(PREP_DIR, "generate_triangular_clusters.py")
TRIANGULAR_TRIANGLE_GENERATOR = os.path.join(
    PREP_DIR, "generate_triangle_nlce_clusters.py"
)


# ---- python/edlib/ scripts (Hamiltonian helpers) ----
#
# These live inside the ``edlib`` Python package that ships alongside
# the ``qed`` wheel (installed by ``pip install qed``). We resolve the
# install location once at import time so qed_nlce works regardless of
# whether the QED source tree is colocated on disk.

def _edlib_dir() -> str:
    try:
        import edlib  # type: ignore
    except ImportError as e:  # pragma: no cover - qed is a required dep
        raise ImportError(
            "qed_nlce needs the `edlib` Python package (shipped with "
            "the `qed` wheel) to invoke per-cluster Hamiltonian helpers. "
            "Reinstall qed (e.g. `pip install --force-reinstall qed`)."
        ) from e
    return os.path.dirname(os.path.abspath(edlib.__file__))


EDLIB_DIR = _edlib_dir()

PYROCHLORE_HELPER = os.path.join(EDLIB_DIR, "helper_cluster.py")
TRIANGULAR_HELPER = os.path.join(EDLIB_DIR, "helper_cluster_triangular.py")
TRIANGULAR_TRIANGLE_HELPER = os.path.join(
    EDLIB_DIR, "helper_cluster_triangular_triangle_based.py"
)


# ---- qed_nlce/run/ scripts (NLCE summation kernels) ----

RUN_DIR = os.path.join(NLCE_DIR, "run")

NLC_SUM_FULL = os.path.join(RUN_DIR, "NLC_sum.py")
NLC_SUM_LANCZOS_BOOST = os.path.join(RUN_DIR, "NLC_sum_LB.py")
NLC_SUM_FTLM = os.path.join(RUN_DIR, "NLC_sum_ftlm.py")
NLC_SUM_TRIANGULAR = os.path.join(RUN_DIR, "NLC_sum_triangular.py")
