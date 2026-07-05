"""On-disk content-addressed cache for NLCE artifacts.

Two caches live here:

1. **Eigenvalue cache** -- keyed by a content hash of the cluster's
   Hamiltonian directory + the ED options. A hit copies the cached
   ``ed_results.h5`` (and the ``thermo/`` block, if present) into the
   per-cluster output directory and skips the diagonalization entirely.
   This is what makes parameter scans (J, h, field-direction sweeps)
   tractable: only the *first* point in the scan pays the ED cost.

2. **Subcluster-table cache** -- keyed by the canonical hash of the
   set of clusters at one (geometry, order). Cluster generation is
   deterministic but slow at high orders, and the inclusion table is
   independent of any Hamiltonian parameters, so caching it once and
   reusing it across every parameter scan is a clean win.

The on-disk layout (``$QED_NLCE_CACHE`` or ``~/.cache/qed_nlce/``):

  eigenvalues/<hash[:2]>/<hash>.h5
  eigenvalues/<hash[:2]>/<hash>.meta.json
  subclusters/<geometry>/<hash>.txt

Hashes are SHA-256 truncated to 32 hex chars (collision probability is
2^-128, fine for a local cache). Cache entries are content-addressed:
deleting them is always safe; the next run rebuilds them.

Canonical-graph hashing
-----------------------

For each cluster's adjacency matrix we compute a Weisfeiler-Lehman
hash (via :mod:`networkx`) which is invariant under vertex relabelings.
This means two clusters that the generator produced with different
local IDs but the same graph topology share a hash and therefore share
a cache entry -- a useful win at high orders where many isomorphic
clusters appear in different embeddings.

If networkx is not available we fall back to a degree-sequence-based
canonical form, which is weaker (sometimes hashes non-isomorphic
graphs together; the eigenvalue cache then catches the divergence via
the Hamiltonian-content hash, so correctness is preserved).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from .ed_runner import EDOptions


__all__ = [
    "default_cache_dir",
    "canonical_cluster_hash",
    "EigenvalueCache",
    "SubclusterCache",
]


_HASH_LEN = 32  # truncated SHA-256 hex chars (= 128 bits)


# ---------------------------------------------------------------------------
# Atomic-write helpers (race-safe for parallel workers writing the same digest)
# ---------------------------------------------------------------------------


def _atomic_copy_file(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` atomically.

    Writes to a sibling ``<dst>.tmp.<pid>.<rand>`` then ``os.replace``s
    onto ``dst``. ``os.replace`` is atomic on POSIX *and* Windows when
    source and target are on the same filesystem, so two parallel
    workers writing the same digest will never observe a torn file.
    The loser of the race simply overwrites the winner with byte-
    identical content (caches are content-addressed by digest).
    """
    dst_dir = os.path.dirname(dst) or "."
    os.makedirs(dst_dir, exist_ok=True)
    tmp = os.path.join(
        dst_dir, f".{os.path.basename(dst)}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}",
    )
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass


def _atomic_copy_tree(src: str, dst: str) -> None:
    """Copy directory ``src`` to ``dst`` atomically.

    Materialises into a sibling tempdir, then ``os.replace``s the
    directory onto ``dst``. ``os.replace`` on a directory is atomic on
    POSIX (rename(2)) provided source and target are on the same
    filesystem and the target either does not exist or is an empty
    directory; we shutil.rmtree the existing target only on the loser
    of a race (same content, so safe).
    """
    dst_parent = os.path.dirname(dst) or "."
    os.makedirs(dst_parent, exist_ok=True)
    tmp = os.path.join(
        dst_parent, f".{os.path.basename(dst)}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}",
    )
    try:
        shutil.copytree(src, tmp)
        if os.path.isdir(dst):
            # Loser of the race -- atomically swap in our copy and
            # remove the existing one. Same digest means same content.
            old_swap = dst + f".old.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            try:
                os.rename(dst, old_swap)
                os.replace(tmp, dst)
            finally:
                if os.path.isdir(old_swap):
                    shutil.rmtree(old_swap, ignore_errors=True)
        else:
            os.replace(tmp, dst)
    finally:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)



def default_cache_dir() -> str:
    """Resolve the on-disk cache root.

    Order of precedence:
      1. ``$QED_NLCE_CACHE`` env var.
      2. ``$XDG_CACHE_HOME/qed_nlce`` (XDG basedir spec).
      3. ``~/.cache/qed_nlce``.
    """
    explicit = os.environ.get("QED_NLCE_CACHE")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return os.path.join(os.path.abspath(os.path.expanduser(xdg)), "qed_nlce")
    return os.path.join(os.path.expanduser("~"), ".cache", "qed_nlce")


# ---------------------------------------------------------------------------
# Canonical-graph hashing
# ---------------------------------------------------------------------------


def _parse_adjacency_from_dat(cluster_file: str) -> Optional[list[list[int]]]:
    """Parse the ``# Adjacency Matrix:`` block out of a cluster .dat file.

    Returns ``None`` on any parse failure -- callers fall back to the
    Hamiltonian-content hash, which is sufficient for correctness.
    """
    try:
        with open(cluster_file, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    m = re.search(
        r"#\s*Adjacency Matrix:\s*\n((?:[01\s]+\n)+)",
        text,
    )
    if not m:
        return None
    rows = []
    for line in m.group(1).strip().splitlines():
        toks = line.split()
        if not toks:
            continue
        try:
            rows.append([int(t) for t in toks])
        except ValueError:
            return None
    if not rows or any(len(r) != len(rows) for r in rows):
        return None
    return rows


def _wl_hash_or_degree_sequence(adjacency: list[list[int]]) -> str:
    """Canonical hash of a simple undirected graph.

    Uses NetworkX Weisfeiler-Lehman if available (graph-isomorphism
    invariant); otherwise falls back to a sorted degree-sequence hash.
    """
    try:
        import networkx as nx  # type: ignore

        n = len(adjacency)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i][j]:
                    g.add_edge(i, j)
        return nx.weisfeiler_lehman_graph_hash(g)
    except Exception:
        # Fallback: sorted degree sequence + edge count.
        degs = sorted(sum(row) for row in adjacency)
        edges = sum(sum(row) for row in adjacency) // 2
        s = f"degseq:{degs}:edges:{edges}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:_HASH_LEN]


def canonical_cluster_hash(cluster_file: str) -> str:
    """Hash of a cluster's *graph topology* (vertex labels factored out).

    Hashes the parsed adjacency matrix via Weisfeiler-Lehman (or a
    degree-sequence fallback). Falls back to hashing the raw file
    contents if the adjacency block can't be parsed.
    """
    adj = _parse_adjacency_from_dat(cluster_file)
    if adj is not None:
        return _wl_hash_or_degree_sequence(adj)[:_HASH_LEN]
    # Last-resort fallback: hash the file contents verbatim.
    h = hashlib.sha256()
    try:
        with open(cluster_file, "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(b"<missing>")
    return h.hexdigest()[:_HASH_LEN]


# ---------------------------------------------------------------------------
# Eigenvalue cache
# ---------------------------------------------------------------------------


@dataclass
class EigenvalueCacheKey:
    """Inputs that must match for an eigenvalue cache hit.

    The cache key is the SHA-256 of the JSON-serialized form. Anything
    that materially affects the eigenvalues / thermodynamics output
    must appear here.
    """

    geometry: str
    num_sites: int
    method: str
    eigenvalues: Optional[str]
    spin_length: float
    streaming_symmetry: bool
    use_symm: bool
    symmetrized: bool
    samples: Optional[int]
    krylov_dim: Optional[int]
    thermo: bool
    temp_min: float
    temp_max: float
    temp_bins: int
    # Dense/OFTLM dispatch threshold + OFTLM quality knobs: these decide
    # which solver tier actually ran (qed.full_spectrum vs stochastic
    # OFTLM) and, within OFTLM, its statistical accuracy. Omitting them
    # let a cache built with a stale --oftlm_cutoff/--oftlm_num_exact
    # silently serve a result computed by the OTHER tier, or by a less
    # accurate OFTLM configuration, after a user deliberately retuned
    # them (e.g. raising --oftlm_cutoff to force exact ED on a cluster
    # previously routed to OFTLM).
    oftlm_cutoff: int
    oftlm_num_exact: int
    oftlm_num_samples: int
    oftlm_krylov_dim: int
    oftlm_num_seeds: int
    exact_max_block: int
    exact_max_sector: int
    device: str
    # Content hashes of the inputs.
    cluster_graph_hash: str
    hamiltonian_content_hash: str

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:_HASH_LEN]


def _hash_dir_contents(d: str) -> str:
    """SHA-256 over the (sorted) contents of every regular file in ``d``.

    Symlinks, sockets, and subdirectories named ``output`` /
    ``basis_cache`` are skipped so that re-runs that incidentally
    write into the input directory don't bust the cache.
    """
    h = hashlib.sha256()
    if not os.path.isdir(d):
        h.update(b"<missing>")
        return h.hexdigest()[:_HASH_LEN]

    skip_dirs = {"output", "basis_cache", "__pycache__"}
    files: list[str] = []
    for root, dirs, fnames in os.walk(d):
        dirs[:] = [x for x in dirs if x not in skip_dirs]
        for fn in fnames:
            files.append(os.path.relpath(os.path.join(root, fn), d))
    for rel in sorted(files):
        full = os.path.join(d, rel)
        if not os.path.isfile(full):
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            with open(full, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()[:_HASH_LEN]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "stores": self.stores, "errors": self.errors}


class EigenvalueCache:
    """Content-addressed cache of per-cluster ED outputs.

    Usage pattern (see :func:`qed_nlce.core.dense_ed.run_ed_in_process`):

      1. ``key = cache.compute_key(geometry, ham_subdir, cluster_file, options, num_sites)``
      2. ``if cache.lookup(key, output_dir): return True``
      3. ... run ED ...
      4. ``cache.store(key, output_dir)``
    """

    def __init__(self, cache_dir: Optional[str] = None, *, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir or default_cache_dir()
        self.eig_dir = os.path.join(self.cache_dir, "eigenvalues")
        if self.enabled:
            os.makedirs(self.eig_dir, exist_ok=True)
        self.stats = CacheStats()

    # ----- key construction -----

    def compute_key(
        self,
        geometry: str,
        ham_subdir: str,
        cluster_file: Optional[str],
        options: EDOptions,
        num_sites: int,
    ) -> EigenvalueCacheKey:
        graph_hash = (
            canonical_cluster_hash(cluster_file) if cluster_file
            else "<no-cluster-file>"
        )
        ham_hash = _hash_dir_contents(ham_subdir)
        return EigenvalueCacheKey(
            geometry=geometry,
            num_sites=num_sites,
            method=options.method.upper(),
            eigenvalues=str(options.eigenvalues) if options.eigenvalues is not None else None,
            spin_length=float(options.spin_length),
            streaming_symmetry=bool(options.streaming_symmetry),
            use_symm=bool(options.use_symm),
            symmetrized=bool(options.symmetrized),
            samples=options.samples,
            krylov_dim=options.krylov_dim,
            thermo=bool(options.thermo),
            temp_min=float(options.temp_min),
            temp_max=float(options.temp_max),
            temp_bins=int(options.temp_bins),
            oftlm_cutoff=int(getattr(options, "oftlm_cutoff", 1 << 15)),
            oftlm_num_exact=int(getattr(options, "oftlm_num_exact", 16)),
            oftlm_num_samples=int(getattr(options, "oftlm_num_samples", 20)),
            oftlm_krylov_dim=int(getattr(options, "oftlm_krylov_dim", 100)),
            oftlm_num_seeds=int(getattr(options, "oftlm_num_seeds", 2)),
            exact_max_block=int(getattr(options, "exact_max_block", 120_000)),
            exact_max_sector=int(getattr(options, "exact_max_sector", 200_000)),
            device=str(getattr(options, "device", "cpu")),
            cluster_graph_hash=graph_hash,
            hamiltonian_content_hash=ham_hash,
        )

    # ----- on-disk layout -----

    def _entry_paths(self, digest: str) -> tuple[str, str, str]:
        shard = digest[:2]
        sd = os.path.join(self.eig_dir, shard)
        return (
            os.path.join(sd, f"{digest}.h5"),
            os.path.join(sd, f"{digest}.thermo"),  # dir
            os.path.join(sd, f"{digest}.meta.json"),
        )

    # ----- lookup / store -----

    def lookup(self, key: EigenvalueCacheKey, output_dir: str) -> bool:
        """If a cache entry exists for ``key``, materialize it under
        ``<output_dir>/output/`` and return True. Otherwise return False.
        """
        if not self.enabled:
            return False
        digest = key.digest()
        h5, thermo_dir, meta = self._entry_paths(digest)
        if not (os.path.isfile(h5) and os.path.isfile(meta)):
            self.stats.misses += 1
            return False
        try:
            out_subdir = os.path.join(output_dir, "output")
            os.makedirs(out_subdir, exist_ok=True)
            # Atomic materialization (mirror of the store path): a kill
            # mid-copy must never leave a torn ed_results.h5 that a
            # --skip_ed rerun would silently treat as valid-but-empty.
            _atomic_copy_file(h5, os.path.join(out_subdir, "ed_results.h5"))
            if os.path.isdir(thermo_dir):
                _atomic_copy_tree(thermo_dir, os.path.join(out_subdir, "thermo"))
            self.stats.hits += 1
            logging.info(
                "[cache] HIT  %s  -> %s", digest[:12], output_dir,
            )
            return True
        except OSError as e:
            logging.warning("[cache] HIT->copy failed for %s: %s", digest[:12], e)
            self.stats.errors += 1
            return False

    def store(self, key: EigenvalueCacheKey, output_dir: str) -> None:
        """Persist the per-cluster output directory to the cache.

        Race-safe under parallel workers: writes go through a
        per-process tempfile/tempdir then ``os.replace`` onto the final
        digest path. Two workers computing the same digest will
        produce byte-identical content; whichever finishes last simply
        overwrites with the same bytes.
        """
        if not self.enabled:
            return
        digest = key.digest()
        out_subdir = os.path.join(output_dir, "output")
        src_h5 = os.path.join(out_subdir, "ed_results.h5")
        if not os.path.isfile(src_h5):
            # Nothing to cache (e.g., text-only legacy output).
            return
        h5, thermo_dir, meta = self._entry_paths(digest)
        try:
            _atomic_copy_file(src_h5, h5)
            src_thermo = os.path.join(out_subdir, "thermo")
            if os.path.isdir(src_thermo):
                _atomic_copy_tree(src_thermo, thermo_dir)
            # Meta file last so a half-finished entry isn't observable
            # to the lookup() path (which keys readiness on the .meta).
            meta_dir = os.path.dirname(meta)
            os.makedirs(meta_dir, exist_ok=True)
            tmp_meta = os.path.join(
                meta_dir,
                f".{os.path.basename(meta)}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}",
            )
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(asdict(key), f, indent=2, sort_keys=True)
            os.replace(tmp_meta, meta)
            self.stats.stores += 1
            logging.info(
                "[cache] STORE %s  <- %s", digest[:12], output_dir,
            )
        except OSError as e:
            logging.warning("[cache] STORE failed for %s: %s", digest[:12], e)
            self.stats.errors += 1

    def log_summary(self, log_tag: str = "cache") -> None:
        s = self.stats
        if s.hits + s.misses + s.stores == 0:
            return
        total = s.hits + s.misses
        rate = (s.hits / total * 100.0) if total else 0.0
        logging.info(
            "[%s] %d hits / %d misses (%.1f%% hit rate), "
            "%d stored, %d errors",
            log_tag, s.hits, s.misses, rate, s.stores, s.errors,
        )


# ---------------------------------------------------------------------------
# Subcluster-table cache
# ---------------------------------------------------------------------------


class SubclusterCache:
    """Cache the per-(geometry, order) subcluster inclusion table.

    The subcluster table is a function of the *graph topology* of the
    cluster set only -- it does not depend on any Hamiltonian
    parameter. The cache is keyed by:

      - geometry name,
      - expansion order,
      - canonical hash of the sorted list of cluster graph hashes.

    Hits are extremely fast (a single file copy); misses are graceful
    no-ops (the workflow runs the cluster generator unchanged).
    """

    def __init__(self, cache_dir: Optional[str] = None, *, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir or default_cache_dir()
        self.sub_dir = os.path.join(self.cache_dir, "subclusters")
        if self.enabled:
            os.makedirs(self.sub_dir, exist_ok=True)
        self.stats = CacheStats()

    def _cluster_set_digest(self, cluster_info_dir: str, order: int) -> Optional[str]:
        """Hash the sorted list of canonical hashes of every cluster .dat."""
        if not os.path.isdir(cluster_info_dir):
            return None
        dats = [
            os.path.join(cluster_info_dir, fn)
            for fn in sorted(os.listdir(cluster_info_dir))
            if fn.endswith(".dat") and fn.startswith("cluster_")
        ]
        if not dats:
            return None
        h = hashlib.sha256()
        h.update(f"order={order}\n".encode("utf-8"))
        for p in dats:
            h.update(canonical_cluster_hash(p).encode("ascii"))
            h.update(b"\n")
        return h.hexdigest()[:_HASH_LEN]

    def _entry_path(self, geometry: str, digest: str) -> str:
        return os.path.join(self.sub_dir, geometry, f"{digest}.txt")

    def lookup(self, geometry: str, order: int, cluster_info_dir: str) -> bool:
        """If the subcluster table is cached, write it into
        ``<cluster_info_dir>/subclusters_info.txt`` and return True.
        """
        if not self.enabled:
            return False
        digest = self._cluster_set_digest(cluster_info_dir, order)
        if digest is None:
            self.stats.misses += 1
            return False
        src = self._entry_path(geometry, digest)
        if not os.path.isfile(src):
            self.stats.misses += 1
            return False
        try:
            dst = os.path.join(cluster_info_dir, "subclusters_info.txt")
            shutil.copy2(src, dst)
            self.stats.hits += 1
            logging.info(
                "[subcluster-cache] HIT  %s  -> %s", digest[:12], dst,
            )
            return True
        except OSError as e:
            logging.warning("[subcluster-cache] HIT->copy failed: %s", e)
            self.stats.errors += 1
            return False

    def store(self, geometry: str, order: int, cluster_info_dir: str) -> None:
        """Persist the freshly-built subcluster table to the cache."""
        if not self.enabled:
            return
        src = os.path.join(cluster_info_dir, "subclusters_info.txt")
        if not os.path.isfile(src):
            return
        digest = self._cluster_set_digest(cluster_info_dir, order)
        if digest is None:
            return
        dst = self._entry_path(geometry, digest)
        try:
            _atomic_copy_file(src, dst)
            self.stats.stores += 1
            logging.info(
                "[subcluster-cache] STORE %s  <- %s", digest[:12], src,
            )
        except OSError as e:
            logging.warning("[subcluster-cache] STORE failed: %s", e)
            self.stats.errors += 1

    def log_summary(self, log_tag: str = "subcluster-cache") -> None:
        s = self.stats
        if s.hits + s.misses + s.stores == 0:
            return
        total = s.hits + s.misses
        rate = (s.hits / total * 100.0) if total else 0.0
        logging.info(
            "[%s] %d hits / %d misses (%.1f%% hit rate), "
            "%d stored, %d errors",
            log_tag, s.hits, s.misses, rate, s.stores, s.errors,
        )
