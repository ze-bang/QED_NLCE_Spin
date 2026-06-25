"""In-memory spin-1/2 Hamiltonian builders for NLCE clusters.

Self-contained ports of the historical ``edlib.helper_cluster*`` physics
(pyrochlore non-Kramers QSI; triangular XXZ-J1J2 / Kitaev / anisotropic),
producing :class:`~qed_nlce.ed.operator.SpinHalfOperator` objects with no
external dependency.
"""

from .cluster import ClusterData, read_cluster_file
from .pyrochlore import build_pyrochlore_operator
from .triangular import build_triangular_operator

__all__ = [
    "ClusterData",
    "read_cluster_file",
    "build_pyrochlore_operator",
    "build_triangular_operator",
]
