"""Bond-direction coloring for triangular-lattice NLCE cluster dedup.

Why this exists
---------------
The triangular Hamiltonian builders (``qed_nlce.hamiltonians.triangular``)
assign per-bond couplings from the bond's real-space direction for the
``kitaev`` (JKGG') and ``anisotropic`` (YbMgGaO4-type) models. Two
embeddings whose *uncolored* site graphs are isomorphic can therefore
have genuinely different spectra: e.g. a straight 3-site chain (two
bonds of the same direction) vs a 120-degree bent chain (two different
directions) differ at the ~0.2|J| level for generic JKGG' couplings.
Deduplicating clusters by plain graph topology -- as the generators
historically did -- silently lumps such embeddings into one class and
weights a single representative by the combined embedding count,
corrupting the NLCE sum for direction-dependent models from order 3 up.

The fix is a *bond-colored* canonical certificate: each nearest-
neighbour bond carries one of three colors (its lattice direction
a1 / a2 / a3), and two clusters are equivalent iff their edge-colored
graphs are isomorphic **up to a global permutation of the three
colors**. The color permutations (the S3 induced by the C6v lattice
point group: 60-degree rotation cycles a1->a2->a3, reflections swap two
of them) are exact covariances of both the kitaev and anisotropic
Hamiltonian conventions used here -- the corresponding combined
spin-lattice rotation makes color-permuted embeddings isospectral --
so modding them out is safe and keeps the class count minimal.

Bond colors are classified from *integer lattice coordinates* with the
minimal-image convention, never from raw floating-point positions, so
PBC-wrapped embeddings are classified exactly.

Certificate encoding
--------------------
pynauty supports vertex colors only, so the edge-colored graph is
encoded by subdivision: every bond (u, v) with color c becomes an
auxiliary vertex placed in color class ``1 + c`` and connected to u and
v; all site vertices sit in class 0. The certificate of a cluster is
the lexicographic minimum over the 6 color permutations, making it a
canonical invariant of the colored-isomorphism-mod-S3 class.
"""
from __future__ import annotations

import itertools

import pynauty

__all__ = [
    "bond_color_from_dij",
    "minimal_image_dij",
    "colored_certificate",
    "unwrap_integer_coords",
    "S3_COLOR_PERMS",
]

# All 6 permutations of the three bond colors (S3, induced by C6v).
S3_COLOR_PERMS = tuple(itertools.permutations(range(3)))

# Integer NN displacements -> bond color, matching the convention in
# qed_nlce.hamiltonians.triangular (_A1 = a1, _A2 = a2, _A3 = a2 - a1):
#   (+-1,  0) -> 0  (a1)
#   ( 0, +-1) -> 1  (a2)
#   (+-(1,-1)) -> 2 (a3 = a2 - a1 direction)
_DIJ_TO_COLOR = {
    (1, 0): 0, (-1, 0): 0,
    (0, 1): 1, (0, -1): 1,
    (1, -1): 2, (-1, 1): 2,
}


def minimal_image_dij(ij_a, ij_b, L: int) -> tuple[int, int]:
    """Minimal-image integer displacement from site a to site b on an
    L x L periodic triangular lattice."""
    di = ((ij_b[0] - ij_a[0] + L // 2) % L) - L // 2
    dj = ((ij_b[1] - ij_a[1] + L // 2) % L) - L // 2
    return di, dj


def bond_color_from_dij(di: int, dj: int) -> int:
    """Bond color (0/1/2 = a1/a2/a3) of a nearest-neighbour displacement."""
    try:
        return _DIJ_TO_COLOR[(di, dj)]
    except KeyError:
        raise ValueError(
            f"({di}, {dj}) is not a nearest-neighbour displacement on the "
            "triangular lattice -- positions/coords are inconsistent"
        ) from None


def colored_certificate(nodes, colored_edges) -> bytes:
    """Canonical certificate of an edge-colored graph, modulo the S3
    action on the three bond colors.

    Parameters
    ----------
    nodes : iterable of hashable
        Cluster sites (any hashable ids).
    colored_edges : iterable of (u, v, color)
        Each bond with its color in {0, 1, 2}.

    Returns
    -------
    bytes -- equal for two clusters iff their colored graphs are
    isomorphic up to a global color permutation.
    """
    nodes_sorted = sorted(nodes)
    n = len(nodes_sorted)
    idx = {v: i for i, v in enumerate(nodes_sorted)}
    edges = [(idx[u], idx[v], int(c)) for u, v, c in colored_edges]
    m = len(edges)

    best = None
    for perm in S3_COLOR_PERMS:
        # Vertex classes: [sites, bond color 0, bond color 1, bond color 2].
        color_classes = [set(range(n)), set(), set(), set()]
        pg = pynauty.Graph(n + m)
        for k, (u, v, c) in enumerate(edges):
            aux = n + k
            color_classes[1 + perm[c]].add(aux)
            pg.connect_vertex(aux, [u, v])
        pg.set_vertex_coloring(color_classes)
        cert = pynauty.certificate(pg)
        if best is None or cert < best:
            best = cert
    return best


def unwrap_integer_coords(nodes, neighbors_of, ij_of, L: int) -> dict:
    """Assign PBC-unwrapped integer coordinates to a connected cluster.

    BFS from the smallest node; every edge step uses the minimal-image
    displacement (exact for NN bonds since the cluster is much smaller
    than the periodic cell). Returns {node: (i, j)} with the start node
    at (0, 0).

    ``neighbors_of(node)`` must yield the node's neighbours *within the
    cluster's induced subgraph*.
    """
    nodes_sorted = sorted(nodes)
    start = nodes_sorted[0]
    coord = {start: (0, 0)}
    queue = [start]
    while queue:
        cur = queue.pop(0)
        ci, cj = coord[cur]
        for nb in neighbors_of(cur):
            if nb in coord:
                continue
            di, dj = minimal_image_dij(ij_of[cur], ij_of[nb], L)
            coord[nb] = (ci + di, cj + dj)
            queue.append(nb)
    if len(coord) != len(nodes_sorted):
        raise ValueError("cluster is not connected under neighbors_of")
    return coord
