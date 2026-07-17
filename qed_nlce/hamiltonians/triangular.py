"""Build the triangular-lattice spin-1/2 Hamiltonian for one cluster.

Faithful in-memory port of ``edlib.helper_cluster_triangular`` covering
its three models -- ``xxz_j1j2``, ``kitaev`` (JKGG') and ``anisotropic``
(YbMgGaO4-type bond-dependent exchange) -- plus the anisotropic-g Zeeman
term and isotropic J2. Op codes: 0=S^+, 1=S^-, 2=S^z.
"""

from __future__ import annotations

import numpy as np

from ..ed.operator import SpinHalfOperator, OP_SP, OP_SM, OP_SZ
from .cluster import ClusterData

_A1 = np.array([1.0, 0.0])
_A2 = np.array([0.5, np.sqrt(3) / 2])
_A3 = np.array([-0.5, np.sqrt(3) / 2])


def _bond_type(p1, p2) -> int:
    delta = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    dn = delta / (np.linalg.norm(delta) + 1e-10)
    dots = [
        abs(np.dot(dn, _A1 / np.linalg.norm(_A1))),
        abs(np.dot(dn, _A2 / np.linalg.norm(_A2))),
        abs(np.dot(dn, _A3 / np.linalg.norm(_A3))),
    ]
    return int(np.argmax(dots))


def _bond_phase(p1, p2):
    bt = _bond_type(p1, p2)
    phases = [0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0]
    phi = phases[bt]
    return np.exp(1j * phi), phi


def _adjacency(cluster: ClusterData) -> dict[int, set]:
    adj: dict[int, set] = {i: set() for i in range(cluster.n_sites)}
    for a, b in cluster.edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


# The six next-nearest-neighbour displacement vectors of the triangular
# lattice, built from the SAME primitives the bond classifier uses. With
# a1=(1,0), a2=(1/2,sqrt3/2): the NNN shell is +-(a1+a2), +-(2*a1-a2),
# +-(2*a2-a1), each of length sqrt(3) (vs 1 for NN and 2 for the 3rd shell).
_NNN_VECS = np.array([
    _A1 + _A2,
    2.0 * _A1 - _A2,
    2.0 * _A2 - _A1,
    -(_A1 + _A2),
    -(2.0 * _A1 - _A2),
    -(2.0 * _A2 - _A1),
])


def _nnn_pairs(cluster: ClusterData, tol: float = 1e-6) -> set:
    """Pairs of sites separated by a true next-nearest-neighbour vector.

    Matched against the KNOWN lattice geometry (``_NNN_VECS``), not graph
    hops. The old implementation walked two edges of the cluster's own NN
    graph and kept anything that was not a direct neighbour, which is wrong
    three ways on a triangular lattice:

      * two GEOMETRIC nearest neighbours whose bond is absent from the
        cluster (the triangle expansion only carries the bonds of its
        chosen triangles) are 2 hops apart, so they were counted as "NNN"
        at distance a;
      * two collinear hops land on the THIRD shell (distance 2a), which was
        also counted as "NNN";
      * a true NNN pair whose connecting site lies OUTSIDE the cluster has
        no 2-hop path inside it, so it was missed entirely.

    Measured on 40 order-6 clusters: 188 pairs at d=a, 209 at d=2a, and 111
    of 508 true NNN missed -- a pure-J2 model came out 1.97x the exact
    high-T limit C = 9R*J2^2/(16T^2). Distance/vector matching is exact here
    because the cluster positions are ideal lattice sites (a = |_A1| = 1),
    and it needs no per-cluster inference of the lattice constant.
    """
    pos = cluster.positions
    ids = sorted(pos)
    xy = {i: np.asarray(pos[i], dtype=float)[:2] for i in ids}
    out = set()
    for ii, i in enumerate(ids):
        for j in ids[ii + 1:]:
            d = xy[j] - xy[i]
            if np.min(np.linalg.norm(_NNN_VECS - d, axis=1)) < tol:
                out.add((min(i, j), max(i, j)))
    return out


def build_triangular_operator(
    cluster: ClusterData,
    *,
    J1: float = 1.0,
    J2: float = 0.0,
    Jz_ratio: float = 1.0,
    h: float = 0.0,
    field_dir=(0, 0, 1),
    model: str = "xxz_j1j2",
    Jzz=None,
    Jpm=None,
    Jpmpm=None,
    Jzpm=None,
    Gamma=None,
    Gamma_prime=None,
    g_ab: float = 2.0,
    g_c: float = 2.0,
) -> SpinHalfOperator:
    """Construct the in-memory triangular operator for ``cluster``."""
    op = SpinHalfOperator(cluster.n_sites)
    pos = cluster.positions

    field_dir = np.asarray(field_dir, dtype=float)
    nrm = np.linalg.norm(field_dir)
    if nrm > 0:
        field_dir = field_dir / nrm

    # Anisotropic-g Zeeman.
    hz = h * field_dir[2] * g_c
    hx = h * field_dir[0] * g_ab
    hy = h * field_dir[1] * g_ab
    for i in range(cluster.n_sites):
        if abs(hz) > 1e-12:
            op.add_single(OP_SZ, i, -hz)
        if abs(hx) > 1e-12:
            op.add_single(OP_SP, i, -0.5 * hx)
            op.add_single(OP_SM, i, -0.5 * hx)
        if abs(hy) > 1e-12:
            op.add_single(OP_SP, i, 0.5j * hy)
            op.add_single(OP_SM, i, -0.5j * hy)

    seen = set()
    for a, b in cluster.edges:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        i, j = key

        if model == "xxz_j1j2":
            Jxy = J1 * Jz_ratio
            op.add_two(OP_SZ, i, OP_SZ, j, J1)
            op.add_two(OP_SP, i, OP_SM, j, 0.5 * Jxy)
            op.add_two(OP_SM, i, OP_SP, j, 0.5 * Jxy)

        elif model == "kitaev":
            _build_kitaev_bond(op, i, j, pos, J1, J2, Gamma, Gamma_prime)

        elif model == "anisotropic":
            _build_anisotropic_bond(
                op, i, j, pos, J1, Jzz, Jpm, Jpmpm, Jzpm
            )
        else:
            raise ValueError(f"unknown triangular model {model!r}")

    # Isotropic J2 (NNN Heisenberg) for non-kitaev models.
    if abs(J2) > 1e-12 and model != "kitaev":
        for i, j in _nnn_pairs(cluster):
            op.add_two(OP_SZ, i, OP_SZ, j, J2)
            op.add_two(OP_SP, i, OP_SM, j, 0.5 * J2)
            op.add_two(OP_SM, i, OP_SP, j, 0.5 * J2)

    return op


def _build_kitaev_bond(op, i, j, pos, J1, J2, Gamma, Gamma_prime):
    J_H = J1
    J_K = J2
    _G = Gamma if Gamma is not None else 0.0
    _Gp = Gamma_prime if Gamma_prime is not None else 0.0
    bt = _bond_type(pos[i], pos[j])
    if bt == 0:
        op.add_two(OP_SP, i, OP_SP, j, complex(J_K / 4.0, -_Gp / 2.0))
        op.add_two(OP_SP, i, OP_SM, j, (2 * J_H + J_K) / 4.0)
        op.add_two(OP_SM, i, OP_SP, j, (2 * J_H + J_K) / 4.0)
        op.add_two(OP_SM, i, OP_SM, j, complex(J_K / 4.0, _Gp / 2.0))
        op.add_two(OP_SZ, i, OP_SZ, j, J_H)
        op.add_two(OP_SP, i, OP_SZ, j, complex(_Gp / 2.0, -_G / 2.0))
        op.add_two(OP_SM, i, OP_SZ, j, complex(_Gp / 2.0, _G / 2.0))
        op.add_two(OP_SZ, i, OP_SP, j, complex(_Gp / 2.0, -_G / 2.0))
        op.add_two(OP_SZ, i, OP_SM, j, complex(_Gp / 2.0, _G / 2.0))
    elif bt == 1:
        op.add_two(OP_SP, i, OP_SP, j, complex(-J_K / 4.0, -_Gp / 2.0))
        op.add_two(OP_SP, i, OP_SM, j, (2 * J_H + J_K) / 4.0)
        op.add_two(OP_SM, i, OP_SP, j, (2 * J_H + J_K) / 4.0)
        op.add_two(OP_SM, i, OP_SM, j, complex(-J_K / 4.0, _Gp / 2.0))
        op.add_two(OP_SZ, i, OP_SZ, j, J_H)
        op.add_two(OP_SP, i, OP_SZ, j, complex(_G / 2.0, -_Gp / 2.0))
        op.add_two(OP_SM, i, OP_SZ, j, complex(_G / 2.0, _Gp / 2.0))
        op.add_two(OP_SZ, i, OP_SP, j, complex(_G / 2.0, -_Gp / 2.0))
        op.add_two(OP_SZ, i, OP_SM, j, complex(_G / 2.0, _Gp / 2.0))
    else:
        op.add_two(OP_SP, i, OP_SP, j, complex(0.0, -_G / 2.0))
        op.add_two(OP_SP, i, OP_SM, j, J_H / 2.0)
        op.add_two(OP_SM, i, OP_SP, j, J_H / 2.0)
        op.add_two(OP_SM, i, OP_SM, j, complex(0.0, _G / 2.0))
        op.add_two(OP_SZ, i, OP_SZ, j, J_H + J_K)
        op.add_two(OP_SP, i, OP_SZ, j, complex(_Gp / 2.0, -_Gp / 2.0))
        op.add_two(OP_SM, i, OP_SZ, j, complex(_Gp / 2.0, _Gp / 2.0))
        op.add_two(OP_SZ, i, OP_SP, j, complex(_Gp / 2.0, -_Gp / 2.0))
        op.add_two(OP_SZ, i, OP_SM, j, complex(_Gp / 2.0, _Gp / 2.0))


def _build_anisotropic_bond(op, i, j, pos, J1, Jzz, Jpm, Jpmpm, Jzpm):
    _Jzz = Jzz if Jzz is not None else J1
    _Jpm = Jpm if Jpm is not None else 0.5 * J1
    _Jpmpm = Jpmpm if Jpmpm is not None else 0.0
    _Jzpm = Jzpm if Jzpm is not None else 0.0
    _gamma, phi = _bond_phase(pos[i], pos[j])
    cphi, sphi = np.cos(phi), np.sin(phi)

    op.add_two(OP_SZ, i, OP_SZ, j, _Jzz)
    op.add_two(OP_SP, i, OP_SM, j, _Jpm)
    op.add_two(OP_SM, i, OP_SP, j, _Jpm)
    op.add_two(OP_SP, i, OP_SP, j, complex(_Jpmpm * cphi, _Jpmpm * sphi))
    op.add_two(OP_SM, i, OP_SM, j, complex(_Jpmpm * cphi, -_Jpmpm * sphi))

    h2 = _Jzpm / 2.0
    op.add_two(OP_SP, i, OP_SZ, j, complex(-h2 * sphi, -h2 * cphi))
    op.add_two(OP_SM, i, OP_SZ, j, complex(-h2 * sphi, h2 * cphi))
    op.add_two(OP_SZ, i, OP_SP, j, complex(-h2 * sphi, -h2 * cphi))
    op.add_two(OP_SZ, i, OP_SM, j, complex(-h2 * sphi, h2 * cphi))
