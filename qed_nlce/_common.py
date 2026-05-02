"""Backward-compatibility re-exports.

This module used to host the shared NLCE infrastructure (logging,
cluster discovery, ED-CLI builder, subprocess driver, HDF5/text
fallback readers). Those have been moved into
:mod:`qed_nlce.core` as part of the package modernization
(see :mod:`qed_nlce`'s docstring for the new layout).

The ``./ED`` subprocess bridge (``build_ed_command``,
``run_ed_subprocess``, ``DEFAULT_ED_PATH``) has since been removed
entirely: the ED step is now executed in-process via the ``qed``
Python package (see :func:`qed_nlce.core.run_ed_in_process`).

This shim is kept for the remaining read-side helpers so that
external scripts still doing ``from qed_nlce._common import
load_thermo_dataset, ...`` continue to work.
"""

from .core.ed_runner import EDOptions
from .core.io import (
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

__all__ = [
    "HAS_H5PY",
    "ClusterEntry",
    "EDOptions",
    "check_gpu_available",
    "setup_logging",
    "get_cluster_files",
    "get_num_sites",
    "count_sites_in_info_file",
    "load_thermo_dataset",
    "load_tpq_thermo_dataset",
]
