"""Core NLCE abstractions: geometries, pipelines, and the workflow orchestrator.

This subpackage defines the *only* extension points that downstream
research code is expected to subclass against:

* :class:`Geometry` -- describes a lattice (which generator to invoke,
  which Hamiltonian helper to invoke, which model parameters are
  meaningful). One concrete subclass per supported lattice lives in
  ``qed_nlce.geometries``.

* :class:`Pipeline` -- describes an ED strategy (which qed method
  to use, which extra CLI flags to thread through, which NLCE
  summation kernel to call afterwards). One concrete subclass per
  ED strategy lives in ``qed_nlce.pipelines``.

* :class:`NLCEWorkflow` -- composes one ``Geometry`` × one ``Pipeline``
  and runs the canonical four-step pipeline (clusters → Hamiltonians
  → ED → summation).

* :class:`EDOptions` -- the per-cluster knobs each ``Pipeline`` fills
  in. The in-process :func:`run_ed_in_process` translates them into a
  :class:`qed.EDParameters` and dispatches to ``qed`` directly; there
  is no ``./ED`` subprocess path.

* I/O helpers (:func:`get_cluster_files`, :func:`load_thermo_dataset`,
  ...) -- the canonical readers for the on-disk schema.
"""

from .ed_runner import EDOptions
from .qed_backend import (
    can_run_in_process,
    qed_available,
    run_ed_in_process,
)
from .cache import (
    EigenvalueCache,
    SubclusterCache,
    canonical_cluster_hash,
    default_cache_dir,
)
from .io import (
    HAS_H5PY,
    ClusterEntry,
    check_gpu_available,
    count_sites_in_info_file,
    get_cluster_files,
    get_num_sites,
    load_thermo_dataset,
    load_tpq_thermo_dataset,
    setup_logging,
)
from .geometry import Geometry, register_geometry, get_geometry, list_geometries
from .pipeline import Pipeline, register_pipeline, get_pipeline, list_pipelines
from .workflow import NLCEWorkflow

__all__ = [
    # ed_runner
    "EDOptions",
    # qed_backend (in-process)
    "can_run_in_process",
    "qed_available",
    "run_ed_in_process",
    # cache
    "EigenvalueCache",
    "SubclusterCache",
    "canonical_cluster_hash",
    "default_cache_dir",
    # io
    "HAS_H5PY",
    "ClusterEntry",
    "check_gpu_available",
    "count_sites_in_info_file",
    "get_cluster_files",
    "get_num_sites",
    "load_thermo_dataset",
    "load_tpq_thermo_dataset",
    "setup_logging",
    # geometry / pipeline / workflow
    "Geometry",
    "register_geometry",
    "get_geometry",
    "list_geometries",
    "Pipeline",
    "register_pipeline",
    "get_pipeline",
    "list_pipelines",
    "NLCEWorkflow",
]
