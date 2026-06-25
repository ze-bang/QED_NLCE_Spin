"""Parse NLCE cluster ``.dat`` files into an in-memory structure.

The cluster generators in :mod:`qed_nlce.prep` write a human-readable
``cluster_{id}_order_{order}.dat`` with header comments (multiplicity,
vertex/edge counts) followed by ``# Vertices``, ``# Edges``,
``# Tetrahedra`` / ``# Triangles``, ``# Adjacency Matrix`` and
``# Node Mapping`` sections. This reader returns a :class:`ClusterData`
with sites already remapped to contiguous matrix indices ``0..N-1`` and
keeps the original vertex ids (needed for the pyrochlore sublattice =
``id % 4`` and bond-phase bookkeeping).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ClusterData:
    cluster_id: int
    order: int
    multiplicity: float
    n_sites: int
    # matrix-index based:
    edges: list[tuple[int, int]] = field(default_factory=list)
    positions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    sublattice: dict[int, int] = field(default_factory=dict)
    tetrahedra: list[tuple[int, ...]] = field(default_factory=list)
    # bookkeeping: matrix index -> original vertex id
    orig_id: dict[int, int] = field(default_factory=dict)


def _parse_multiplicity(text: str) -> float:
    """Parse a multiplicity header value.

    Triangle-based NLCE writes fractional multiplicities such as
    ``1/3 = 0.333333``; site-based writes plain integers/floats. Accept
    either by preferring the explicit decimal after ``=`` and falling
    back to evaluating a ``p/q`` fraction.
    """
    text = text.strip()
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def read_cluster_file(filepath: str) -> ClusterData:
    """Read one cluster ``.dat`` file into a :class:`ClusterData`."""
    cluster_id = order = None
    multiplicity = None
    vertices: dict[int, tuple[float, float, float]] = {}
    edges_raw: list[tuple[int, int]] = []
    tets_raw: list[tuple[int, ...]] = []
    node_mapping: dict[int, int] = {}

    m = re.search(r"cluster_(\d+)_order_(\d+)", filepath)
    if m:
        cluster_id, order = int(m.group(1)), int(m.group(2))

    section = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Multiplicity" in line and ":" in line:
                    multiplicity = _parse_multiplicity(line.split(":", 1)[-1])
                elif "Cluster ID" in line and ":" in line:
                    cluster_id = int(line.split(":")[-1].strip())
                elif "Order" in line and ":" in line:
                    try:
                        order = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                if "Vertices" in line:
                    section = "vertices"
                elif "Edges" in line:
                    section = "edges"
                elif "Tetrahedra" in line:
                    section = "tetrahedra"
                elif "Triangles" in line:
                    section = "triangles"
                elif "Adjacency" in line:
                    section = "adjacency"
                elif "Node Mapping" in line:
                    section = "node_mapping"
                continue

            if section == "vertices":
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    vid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3]) if len(parts) >= 4 else 0.0
                    vertices[vid] = (x, y, z)
            elif section == "edges":
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    edges_raw.append((int(parts[0]), int(parts[1])))
            elif section in ("tetrahedra", "triangles"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    tets_raw.append(tuple(int(p) for p in parts))
            elif section == "node_mapping":
                mm = re.search(r"(\d+):\s*(\d+)", line)
                if mm:
                    node_mapping[int(mm.group(1))] = int(mm.group(2))

    if not node_mapping:
        node_mapping = {v: i for i, v in enumerate(sorted(vertices))}

    n_sites = len(vertices)
    cd = ClusterData(
        cluster_id=cluster_id if cluster_id is not None else -1,
        order=order if order is not None else n_sites,
        multiplicity=multiplicity if multiplicity is not None else 1.0,
        n_sites=n_sites,
    )
    for vid, idx in node_mapping.items():
        cd.orig_id[idx] = vid
        if vid in vertices:
            cd.positions[idx] = vertices[vid]
        cd.sublattice[idx] = vid % 4
    cd.edges = [(node_mapping[a], node_mapping[b]) for a, b in edges_raw]
    cd.tetrahedra = [
        tuple(node_mapping[v] for v in tet) for tet in tets_raw
    ]
    return cd
