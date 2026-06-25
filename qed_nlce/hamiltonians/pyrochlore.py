"""Build the pyrochlore spin-1/2 Hamiltonian for one cluster.

Ports the physics of the historical ``edlib.helper_cluster`` (XYZ /
non-Kramers quantum-spin-ice model in the local [111] frame + Zeeman)
directly into an in-memory :class:`~qed_nlce.ed.operator.SpinHalfOperator`,
so no ``Trans.dat`` / ``InterAll.dat`` round-trip or ``edlib`` import is
needed.

Model (local-frame, spin-1/2, op codes 0=S^+, 1=S^-, 2=S^z)::

    Jpm   = -(Jxx + Jyy) / 4
    Jpmpm =  (Jxx - Jyy) / 4
    gamma = exp(i 2 pi / 3)
    factor[a,b] = non-Kramers 4x4 bond phase (sublattice = id % 4)

    H = sum_{<ij>} [ Jzz S^z_i S^z_j
                     - Jpm (S^+_i S^-_j + S^-_i S^+_j)
                     + conj(Jpmpm*factor) S^+_i S^+_j
                     +      Jpmpm*factor  S^-_i S^-_j ]
        - sum_i h (field_dir . z_local[sub_i]) S^z_i
"""

from __future__ import annotations

import numpy as np

from ..ed.operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from .cluster import ClusterData

# Local z-axes of the four pyrochlore sublattices ([111]-type body diagonals).
_Z_LOCAL = np.array(
    [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ],
    dtype=float,
) / np.sqrt(3.0)

_GAMMA = np.exp(1j * 2 * np.pi / 3)
_NON_KRAMERS = np.array(
    [
        [0, 1, _GAMMA, _GAMMA**2],
        [1, 0, _GAMMA**2, _GAMMA],
        [_GAMMA, _GAMMA**2, 0, 1],
        [_GAMMA**2, _GAMMA, 1, 0],
    ],
    dtype=complex,
)


def build_pyrochlore_operator(
    cluster: ClusterData,
    Jxx: float,
    Jyy: float,
    Jzz: float,
    h: float = 0.0,
    field_dir=(1, 1, 1),
) -> SpinHalfOperator:
    """Construct the in-memory pyrochlore operator for ``cluster``."""
    op = SpinHalfOperator(cluster.n_sites)

    Jpm = -(Jxx + Jyy) / 4.0
    Jpmpm = (Jxx - Jyy) / 4.0

    field_dir = np.asarray(field_dir, dtype=float)
    nrm = np.linalg.norm(field_dir)
    if nrm > 0:
        field_dir = field_dir / nrm

    # Zeeman: -h (field_dir . z_local[sub_i]) S^z_i
    for i in range(cluster.n_sites):
        sub = cluster.sublattice[i]
        local_field = h * float(np.dot(field_dir, _Z_LOCAL[sub]))
        op.add_single(OP_SZ, i, -local_field)

    # Exchange on each undirected bond (dedupe).
    seen = set()
    for a, b in cluster.edges:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        i, j = key
        sa = cluster.sublattice[i]
        sb = cluster.sublattice[j]
        Jpmpm_ = Jpmpm * _NON_KRAMERS[sa, sb]

        op.add_two(OP_SZ, i, OP_SZ, j, Jzz)
        op.add_two(OP_SP, i, OP_SM, j, -Jpm)
        op.add_two(OP_SM, i, OP_SP, j, -Jpm)
        op.add_two(OP_SP, i, OP_SP, j, np.conj(Jpmpm_))
        op.add_two(OP_SM, i, OP_SM, j, Jpmpm_)

    return op
