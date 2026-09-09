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
    "cache_budget_from_env",
    "canonical_cluster_hash",
    "EigenvalueCache",
    "SubclusterCache",
    "CacheEntry",
    "scan_cache",
    "cache_stats",
    "prune_cache",
    "clear_cache",
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



def cache_budget_from_env() -> tuple:
    """Read the automatic-eviction budget from the environment.

    Returns ``(max_bytes, max_age_days)``, either of which may be None.

    On an HPC cluster the cache lands in ``$HOME/.cache`` unless
    ``$QED_NLCE_CACHE`` says otherwise, and ``$HOME`` is usually the
    quota-limited filesystem. A long fitting job stores an entry per
    (cluster, parameter point), so an unbounded cache is what fills the
    quota -- set ``QED_NLCE_CACHE_MAX_GB`` (and point
    ``QED_NLCE_CACHE`` at scratch) to keep a job self-limiting.
    """
    max_bytes = None
    raw = os.environ.get("QED_NLCE_CACHE_MAX_GB")
    if raw:
        try:
            max_bytes = int(float(raw) * 1024 ** 3)
        except ValueError:
            logging.warning("[cache] ignoring bad QED_NLCE_CACHE_MAX_GB=%r", raw)
    max_age = None
    raw = os.environ.get("QED_NLCE_CACHE_MAX_AGE_DAYS")
    if raw:
        try:
            max_age = float(raw)
        except ValueError:
            logging.warning(
                "[cache] ignoring bad QED_NLCE_CACHE_MAX_AGE_DAYS=%r", raw)
    return max_bytes, max_age


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
    # Engine-policy stamp: bumped whenever the symmetry/routing policy
    # changes what a "cached result" means (e.g. the qed-engine rewrite:
    # GeneratorSet-with-residue projection, plan_only routing, native
    # C++ operator ingestion, new OFTLM seed derivation). Adding these
    # fields changes every digest -> ONE loud full invalidation instead
    # of silently replaying results computed under the old policy.
    ed_engine: str = "qed-engine-v2"
    point_group: str = "auto"
    # Routing policy: False (default) = EXACT-ONLY (over-cap clusters warn
    # and solve exactly); True = stochastic OFTLM fallback for over-cap
    # clusters. Decides which tier produced a cached result.
    oftlm_fallback: bool = False

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
        # Automatic eviction. Scanning on every store would be O(entries)
        # per store, so only check periodically -- the budget is a ceiling
        # to stay under, not a byte-exact cap.
        self._max_bytes, self._max_age_days = cache_budget_from_env()
        self._stores_since_check = 0
        self._bytes_since_check = 0
        self._check_every = 32

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
            exact_max_sector=int(getattr(options, "exact_max_sector", 800_000)),
            device=str(getattr(options, "device", "cpu")),
            cluster_graph_hash=graph_hash,
            hamiltonian_content_hash=ham_hash,
            point_group=str(getattr(options, "point_group", "auto")),
            oftlm_fallback=bool(getattr(options, "oftlm_fallback", False)),
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
            _, written = _entry_recency_and_size(
                [p for p in self._entry_paths(digest)])
            self._maybe_enforce_budget(written)
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


    def _maybe_enforce_budget(self, written_bytes: int = 0) -> None:
        """Evict down to the configured budget, every `_check_every` stores.

        Entries are content-addressed and rebuilt on the next miss, so
        eviction is always safe. Failures here must never take down the
        run that produced the entry, hence the broad guard.
        """
        if self._max_bytes is None and self._max_age_days is None:
            return
        self._stores_since_check += 1
        self._bytes_since_check += max(0, int(written_bytes))
        # Check on whichever comes first: a store count, or enough new
        # bytes to matter against the budget. Counting alone overshoots
        # badly when entries are large (32 x 100 MB before the first
        # check); bytes alone never fires when they are tiny.
        slack = (self._max_bytes * 0.1) if self._max_bytes else float("inf")
        if (self._stores_since_check < self._check_every
                and self._bytes_since_check < slack):
            return
        self._stores_since_check = 0
        self._bytes_since_check = 0
        try:
            r = prune_cache(self.cache_dir,
                            max_age_days=self._max_age_days,
                            max_bytes=self._max_bytes)
            if r["entries_removed"]:
                logging.info(
                    "[cache] auto-evicted %d entries (%.1f MB); "
                    "%.1f MB now in %s",
                    r["entries_removed"], r["bytes_removed"] / 1024 ** 2,
                    r["bytes_after"] / 1024 ** 2, self.cache_dir,
                )
        except Exception as e:                      # never break the run
            logging.warning("[cache] auto-eviction failed: %s", e)


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


# ---------------------------------------------------------------------------
# Maintenance: stats and pruning
# ---------------------------------------------------------------------------
#
# Nothing above ever deletes a cache entry, so a machine that runs many
# scans accumulates them without bound. Entries are content-addressed and
# rebuildable (see the module docstring), so eviction is always safe: the
# worst case is paying the ED cost again on the next miss.
#
# "Recency" is the newest of mtime/atime over an entry's files. Access
# time is the honest signal for LRU, but many filesystems are mounted
# noatime (and WSL2's drvfs is inconsistent about it), so mtime is the
# floor -- an entry that was written recently is never treated as cold.


@dataclass
class CacheEntry:
    """One logical cache entry: a primary file plus its sidecars."""

    key: str
    kind: str                       # "eigenvalues" | "subclusters"
    paths: list                     # every file to unlink together
    size: int                       # summed bytes
    recency: float                  # newest mtime/atime across `paths`

    @property
    def age_days(self) -> float:
        import time
        return max(0.0, (time.time() - self.recency) / 86400.0)


def _stat_one(path) -> tuple:
    try:
        st = os.stat(path)
    except OSError:
        return 0.0, 0
    return max(st.st_mtime, st.st_atime), st.st_size


def _entry_recency_and_size(paths) -> tuple:
    """Newest timestamp and total bytes over `paths`.

    An eigenvalue entry carries a ``<digest>.thermo/`` directory beside
    its .h5, so directories are walked rather than stat'ed -- otherwise
    the thermo block counts as ~0 bytes and survives eviction as an
    orphan.
    """
    newest, total = 0.0, 0
    for p in paths:
        if os.path.isdir(p):
            for dirpath, _dirnames, filenames in os.walk(p):
                for fn in filenames:
                    t, sz = _stat_one(os.path.join(dirpath, fn))
                    newest = max(newest, t)
                    total += sz
        else:
            t, sz = _stat_one(p)
            newest = max(newest, t)
            total += sz
    return newest, total


def scan_cache(cache_dir: Optional[str] = None) -> list:
    """Enumerate every cache entry under ``cache_dir``.

    Eigenvalue entries group ``<hash>.h5`` with its ``<hash>.meta.json``
    sidecar so the pair is always evicted together -- a stranded
    ``.meta.json`` would otherwise advertise a payload that is gone.
    """
    root = cache_dir or default_cache_dir()
    entries = []

    eig_root = os.path.join(root, "eigenvalues")
    for dirpath, _dirnames, filenames in os.walk(eig_root):
        by_key = {}
        for fn in filenames:
            stem = fn[:-3] if fn.endswith(".h5") else (
                fn[:-10] if fn.endswith(".meta.json") else None)
            if stem is None:
                continue
            by_key.setdefault(stem, []).append(os.path.join(dirpath, fn))
        # The thermo block is a sibling *directory*, not a file, so it
        # never shows up in `filenames`.
        for d in _dirnames:
            if d.endswith(".thermo"):
                by_key.setdefault(d[:-7], []).append(os.path.join(dirpath, d))
        for key, paths in by_key.items():
            recency, size = _entry_recency_and_size(paths)
            entries.append(CacheEntry(key, "eigenvalues", sorted(paths),
                                      size, recency))

    sub_root = os.path.join(root, "subclusters")
    for dirpath, _dirnames, filenames in os.walk(sub_root):
        geom = os.path.basename(dirpath)
        for fn in filenames:
            if not fn.endswith(".txt"):
                continue
            p = os.path.join(dirpath, fn)
            recency, size = _entry_recency_and_size([p])
            entries.append(CacheEntry(f"{geom}/{fn[:-4]}", "subclusters",
                                      [p], size, recency))

    return entries


def cache_stats(cache_dir: Optional[str] = None) -> dict:
    """Summarise cache occupancy, by kind."""
    root = cache_dir or default_cache_dir()
    entries = scan_cache(root)
    out = {
        "cache_dir": root,
        "exists": os.path.isdir(root),
        "total_entries": len(entries),
        "total_bytes": sum(e.size for e in entries),
        "by_kind": {},
    }
    for kind in ("eigenvalues", "subclusters"):
        sel = [e for e in entries if e.kind == kind]
        out["by_kind"][kind] = {
            "entries": len(sel),
            "bytes": sum(e.size for e in sel),
            "oldest_age_days": max((e.age_days for e in sel), default=0.0),
            "newest_age_days": min((e.age_days for e in sel), default=0.0),
        }
    return out


def prune_cache(cache_dir: Optional[str] = None, *,
                max_age_days: Optional[float] = None,
                max_bytes: Optional[int] = None,
                dry_run: bool = False) -> dict:
    """Evict cache entries by age, then by total size (coldest first).

    ``max_age_days`` drops every entry untouched for longer than that.
    ``max_bytes`` then evicts coldest-first until the cache fits. Both
    are optional; with neither set this reports what it would do and
    changes nothing.

    Returns a summary dict. Safe by construction: every entry is
    content-addressed and rebuilt on the next miss.
    """
    root = cache_dir or default_cache_dir()
    entries = scan_cache(root)
    before_bytes = sum(e.size for e in entries)
    doomed, reasons = [], {}

    if max_age_days is not None:
        for e in entries:
            if e.age_days > max_age_days:
                doomed.append(e)
                reasons[id(e)] = f"age {e.age_days:.1f}d > {max_age_days}d"

    if max_bytes is not None:
        doomed_ids = {id(e) for e in doomed}
        kept = [e for e in entries if id(e) not in doomed_ids]
        running = sum(e.size for e in kept)
        # Coldest first: the least recently used entries go first.
        for e in sorted(kept, key=lambda x: x.recency):
            if running <= max_bytes:
                break
            doomed.append(e)
            reasons[id(e)] = "over size budget"
            running -= e.size

    removed_bytes, removed_files, errors = 0, 0, 0
    if not dry_run:
        for e in doomed:
            for p in e.paths:
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                    removed_files += 1
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logging.warning("[cache-prune] could not remove %s: %s",
                                    p, exc)
                    errors += 1
            removed_bytes += e.size
        _remove_empty_dirs(root)
    else:
        removed_bytes = sum(e.size for e in doomed)

    return {
        "cache_dir": root,
        "dry_run": dry_run,
        "entries_before": len(entries),
        "entries_removed": len(doomed),
        "bytes_before": before_bytes,
        "bytes_removed": removed_bytes,
        "bytes_after": before_bytes - removed_bytes,
        "files_removed": removed_files,
        "errors": errors,
        "reasons": {e.key: reasons[id(e)] for e in doomed},
    }


def _remove_empty_dirs(root: str) -> None:
    """Drop the shard directories an eviction just emptied."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root or dirnames or filenames:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            pass


def clear_cache(cache_dir: Optional[str] = None) -> dict:
    """Remove every entry. Equivalent to ``prune_cache(max_age_days=-1)``."""
    return prune_cache(cache_dir, max_age_days=-1.0)
