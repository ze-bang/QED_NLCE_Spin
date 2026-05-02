"""Concrete geometry implementations.

Importing this subpackage triggers registration of every supported
geometry (``pyrochlore``, ``triangular_site``, ``triangular_triangle``).
After import, :func:`qed_nlce.core.list_geometries` reports them
all, and :func:`qed_nlce.core.get_geometry` can instantiate any of
them by name.

To add a new lattice:

  1. Drop a new module in this directory (e.g. ``kagome.py``).
  2. Subclass :class:`qed_nlce.core.Geometry` and decorate with
     :func:`qed_nlce.core.register_geometry`.
  3. Import the new module from this ``__init__`` so the registration
     side-effect fires.
"""

from . import pyrochlore  # noqa: F401  -- registration side-effect
from . import triangular_site  # noqa: F401
from . import triangular_triangle  # noqa: F401
