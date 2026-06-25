"""Concrete ED-pipeline implementations.

Importing this subpackage triggers registration of the supported
pipeline (``full_ed``). After import, :func:`qed_nlce.core.list_pipelines`
reports it, and :func:`qed_nlce.core.get_pipeline` can instantiate it by
name.

This package is dense-only: every cluster is solved by full,
symmetry-adapted dense diagonalization. The historical approximate
backends (FTLM, KPM-DOS, Lanczos-boost, auto-hybrid) have been removed.

To add a new pipeline:

  1. Drop a new module here.
  2. Subclass :class:`qed_nlce.core.Pipeline` and decorate with
     :func:`qed_nlce.core.register_pipeline`.
  3. Import the new module from this ``__init__`` so registration fires.
"""

from . import full_ed  # noqa: F401
