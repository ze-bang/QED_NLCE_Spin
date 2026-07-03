"""High-level NLCE workflow orchestrator.

:class:`NLCEWorkflow` composes one :class:`Geometry` with one
:class:`Pipeline` and runs the canonical four-step pipeline:

  1. **Cluster generation** -- delegated to ``Geometry.generate_clusters``.
  2. **Hamiltonian preparation** -- delegated to
     ``Geometry.prepare_hamiltonian`` (one call per cluster).
  3. **Exact diagonalization** -- per cluster, the workflow asks
     ``Pipeline.make_ed_options(args, num_sites)`` for the per-cluster
     options and runs ED in-process via
     :func:`qed_nlce.core.run_ed_in_process`, which calls the
     ``qed`` Python package directly. There is no longer any ``./ED``
     subprocess path.
  4. **NLCE summation** -- delegated to ``Pipeline.summation_command``
     (which still shells out to the pure-Python summation kernels in
     :mod:`qed_nlce.run`).

The orchestrator also handles:
  * the ``--streaming-symmetry`` orbit basis precompute step (delegated
    to ``Geometry.precompute_basis``),
  * directory layout (``clusters_*``, ``hamiltonians_*``, ``ed_results_*``,
    ``nlc_results_*``),
  * per-cluster thermo plotting (when ``--thermo`` is on).

Skipping any step is supported via ``--skip_cluster_gen``, ``--skip_ham_prep``,
``--skip_ed``, ``--skip_nlc``.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .dense_ed import assert_qed_available, can_run_in_process, run_ed_in_process
from .cache import EigenvalueCache, SubclusterCache, default_cache_dir
from .ed_runner import EDOptions
from .geometry import Geometry
from .io import (
    ClusterEntry,
    count_sites_in_info_file,
    get_cluster_files,
    setup_logging,
)
from .mpi_dispatch import mpi_available, scatter_ed_jobs, is_rank_zero, barrier
from .pipeline import Pipeline


__all__ = ["NLCEWorkflow"]


def _basis_worker(payload: tuple) -> bool:
    """Pure function form of one basis precompute call."""
    geometry, args, cluster_id, order, ham_subdir = payload
    return geometry.precompute_basis(args, cluster_id, order, ham_subdir)


# ---------------------------------------------------------------------------
# Parallel-ED worker (spawn-safe; reconstructs the cache from cache_dir).
# ---------------------------------------------------------------------------


@dataclass
class _EDPayload:
    """Picklable per-cluster ED job description for the spawn-Pool."""
    ham_subdir: str
    ed_subdir: str
    num_sites: int
    options: EDOptions
    log_tag: str
    cache_dir: Optional[str]
    cache_enabled: bool
    cache_key_extras: dict


def _ed_pool_initializer(omp_threads: int) -> None:
    """Pin BLAS/OMP threading inside each worker.

    Set BEFORE the worker imports ``qed`` / numpy / scipy so MKL,
    OpenBLAS, and OpenMP all initialise with the correct per-worker
    thread budget. Avoids the ``num_workers * total_cores`` thread
    explosion that otherwise destroys parallel scaling.
    """
    val = str(max(1, int(omp_threads)))
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[k] = val


def _ed_worker(payload: _EDPayload) -> tuple[str, bool, Optional[str]]:
    """Run ED for one cluster in a fresh worker process.

    Each worker imports ``qed`` lazily on first call, by which point
    the pool initializer has already pinned the BLAS/OMP thread
    counts. Cache instance is reconstructed locally because pickling
    the parent's :class:`EigenvalueCache` is unnecessary (it is just
    constructor args + a stats counter we don't aggregate cross-
    process).
    """
    try:
        cache = EigenvalueCache(
            payload.cache_dir or default_cache_dir(),
            enabled=payload.cache_enabled,
        )
        ok = run_ed_in_process(
            payload.ham_subdir, payload.ed_subdir,
            payload.num_sites, payload.options,
            log_tag=payload.log_tag,
            cache=cache, cache_key_extras=payload.cache_key_extras,
        )
        return (payload.log_tag, bool(ok), None)
    except Exception as exc:  # noqa: BLE001
        return (payload.log_tag, False, f"{type(exc).__name__}: {exc}")


class NLCEWorkflow:
    """One ``Geometry`` × one ``Pipeline``  =  one NLCE run."""

    def __init__(
        self,
        geometry: Geometry,
        pipeline: Pipeline,
        args: argparse.Namespace,
    ):
        # Fail fast: both ED tiers (exact qed.full_spectrum + OFTLM) need
        # the qed C++ package. Check before any cluster work starts, not
        # lazily on the first large cluster hours into an order-8 run.
        assert_qed_available()

        self.geometry = geometry
        self.pipeline = pipeline
        self.args = args

        # Directory layout (geometry can override the prefixes).
        self.base_dir = os.path.abspath(args.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        order = args.max_order
        self.cluster_dir = os.path.join(
            self.base_dir, f"{geometry.cluster_dir_prefix}{order}",
        )
        self.ham_dir = os.path.join(
            self.base_dir, f"{geometry.hamiltonian_dir_prefix}{order}",
        )
        self.ed_dir = os.path.join(
            self.base_dir, f"{geometry.ed_dir_prefix}{order}",
        )
        self.nlc_dir = os.path.join(
            self.base_dir, f"{geometry.nlc_dir_prefix}{order}",
        )
        for d in (self.cluster_dir, self.ham_dir, self.ed_dir, self.nlc_dir):
            if not (os.path.islink(d) or os.path.isdir(d)):
                os.makedirs(d, exist_ok=True)

        self.cluster_info_dir = geometry.cluster_info_path(self.base_dir, order)

        # On-disk caches (eigenvalue + subcluster). Disabled when the
        # user passes --no_cache; otherwise default to either the
        # explicit --cache_dir, the QED_NLCE_CACHE env var, or
        # ~/.cache/qed_nlce/.
        cache_enabled = not bool(getattr(args, "no_cache", False))
        cache_dir = getattr(args, "cache_dir", None) or default_cache_dir()
        self.eig_cache = EigenvalueCache(cache_dir, enabled=cache_enabled)
        self.subcluster_cache = SubclusterCache(cache_dir, enabled=cache_enabled)
        if cache_enabled:
            logging.getLogger().debug("Using NLCE cache dir: %s", cache_dir)

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        """Run all four steps of the NLCE pipeline (skipping per CLI flags)."""
        use_mpi = bool(getattr(self.args, "mpi", False)) and mpi_available()
        rank0 = is_rank_zero()

        if rank0:
            log_file = os.path.join(self.base_dir, "nlce_workflow.log")
            setup_logging(log_file)
            self._banner_top()

        # Steps 1, 2, 2.5 run on rank 0 only; other ranks wait at barrier
        if rank0:
            if not self.args.skip_cluster_gen:
                self._step1_clusters()
            else:
                logging.info("Skipping cluster generation step.")

        if use_mpi:
            barrier()

        clusters = self._discover_clusters()

        if rank0:
            if not self.args.skip_ham_prep:
                self._step2_hamiltonians(clusters)
            else:
                logging.info("Skipping Hamiltonian preparation step.")

            if self.pipeline.needs_basis_precompute(self.args) and not getattr(
                self.args, "skip_basis_precompute", False
            ):
                self._step2_5_basis_precompute(clusters)

        if use_mpi:
            barrier()

        if not self.args.skip_ed:
            self._step3_ed(clusters)
        else:
            if rank0:
                logging.info("Skipping ED step.")

        if use_mpi:
            barrier()

        if rank0:
            if not self.args.skip_nlc:
                self._step4_summation()
            else:
                logging.info("Skipping NLCE summation step.")

            self._banner_bottom()

    # ----------------------------------------------------------- helpers

    def _banner_top(self) -> None:
        logging.info("=" * 80)
        logging.info(
            "NLCE workflow:  geometry=%s  pipeline=%s",
            self.geometry.name, self.pipeline.name,
        )
        logging.info("  max_order=%d  base_dir=%s", self.args.max_order, self.base_dir)
        logging.info("=" * 80)

    def _banner_bottom(self) -> None:
        logging.info("=" * 80)
        logging.info("NLCE workflow completed.")
        logging.info("Results in: %s", self.base_dir)
        logging.info("=" * 80)

    def _discover_clusters(self) -> list[ClusterEntry]:
        if not os.path.exists(self.cluster_info_dir):
            logging.error("Cluster info directory not found: %s", self.cluster_info_dir)
            sys.exit(1)
        clusters = get_cluster_files(self.cluster_info_dir)
        if not clusters:
            logging.error("No cluster files discovered in %s", self.cluster_info_dir)
            sys.exit(1)
        logging.info("Discovered %d clusters in %s", len(clusters), self.cluster_info_dir)
        return clusters

    def _ham_subdir_for(self, cluster: ClusterEntry) -> str:
        return os.path.join(
            self.ham_dir, f"cluster_{cluster.cluster_id}_order_{cluster.order}"
        )

    def _ed_subdir_for(self, cluster: ClusterEntry) -> str:
        return os.path.join(
            self.ed_dir, f"cluster_{cluster.cluster_id}_order_{cluster.order}"
        )

    # ------------------------------------------------------------ step 1

    # Parameters that determine the cluster set (NOT coupling values --
    # those only affect the Hamiltonian/ED steps). A marker file with
    # this payload makes cluster generation resumable: rerunning after a
    # crash later in the pipeline skips the (expensive at order >= 7)
    # regeneration instead of silently redoing all of it.
    def _generation_marker_payload(self) -> dict:
        return {
            "geometry": self.geometry.name,
            "max_order": int(self.args.max_order),
            # model selects bond-colored dedup on triangular geometries
            "model": getattr(self.args, "model", None),
        }

    def _step1_clusters(self) -> None:
        logging.info("=" * 80)
        logging.info("Step 1: Cluster generation (%s)", self.geometry.name)
        logging.info("=" * 80)

        import glob
        import json

        marker = os.path.join(self.cluster_info_dir, ".generation_complete.json")
        payload = self._generation_marker_payload()
        sub_info = os.path.join(self.cluster_info_dir, "subclusters_info.txt")

        def _count_cluster_files() -> int:
            return len(glob.glob(
                os.path.join(self.cluster_info_dir, "cluster_*_order_*.dat")))

        if os.path.isfile(marker):
            try:
                with open(marker) as f:
                    existing = json.load(f)
            except (OSError, json.JSONDecodeError):
                existing = None
            existing_count = (existing or {}).pop("n_cluster_files", -1) \
                if isinstance(existing, dict) else -1
            if existing == payload and os.path.isfile(sub_info) \
                    and os.path.getsize(sub_info) > 0 \
                    and _count_cluster_files() == existing_count \
                    and existing_count > 0:
                logging.info(
                    "Cluster generation already complete for %s (%d cluster "
                    "files, marker matches); skipping. Delete %s to force "
                    "regeneration.", payload, existing_count, marker,
                )
                return
            logging.info(
                "Generation marker present but stale/incomplete (marker=%s "
                "want=%s, files=%d recorded=%d); regenerating.",
                existing, payload, _count_cluster_files(), existing_count,
            )

        ok = self.geometry.generate_clusters(self.args, self.args.max_order, self.cluster_dir)
        if not ok:
            logging.error("Cluster generation failed for geometry=%s", self.geometry.name)
            sys.exit(1)

        # Success marker (written AFTER all generator outputs, so a crash
        # mid-generation never leaves a marker claiming completeness).
        # Records the cluster-file count so partial deletions are detected
        # on the next run instead of silently truncating the NLCE sum.
        try:
            os.makedirs(self.cluster_info_dir, exist_ok=True)
            record = dict(payload)
            record["n_cluster_files"] = _count_cluster_files()
            tmp = f"{marker}.tmp-{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(record, f)
            os.replace(tmp, marker)
        except OSError as exc:
            logging.warning("Could not write generation marker: %s", exc)
        # Subcluster-table cache: if we just regenerated and the
        # generator wrote subclusters_info.txt, persist it. If the
        # generator skipped that step (or its output is empty), try
        # the cache.
        try:
            sub_info = os.path.join(self.cluster_info_dir, "subclusters_info.txt")
            if os.path.isfile(sub_info) and os.path.getsize(sub_info) > 0:
                self.subcluster_cache.store(
                    self.geometry.name, self.args.max_order, self.cluster_info_dir,
                )
            else:
                self.subcluster_cache.lookup(
                    self.geometry.name, self.args.max_order, self.cluster_info_dir,
                )
        except Exception as exc:
            logging.debug("subcluster cache hook failed: %s", exc)

    # ------------------------------------------------------------ step 2

    def _step2_hamiltonians(self, clusters: list[ClusterEntry]) -> None:
        logging.info("=" * 80)
        logging.info("Step 2: Hamiltonian preparation (%d clusters)", len(clusters))
        logging.info("=" * 80)
        try:
            from tqdm import tqdm  # noqa: F401
            iterable = tqdm(clusters, desc="Preparing Hamiltonians")
        except ImportError:
            iterable = clusters
        for cluster in iterable:
            ham_subdir = self._ham_subdir_for(cluster)
            os.makedirs(ham_subdir, exist_ok=True)
            ok = self.geometry.prepare_hamiltonian(
                self.args, cluster.cluster_id, cluster.order, cluster.path, ham_subdir,
            )
            if not ok:
                logging.warning(
                    "Hamiltonian prep failed for cluster %d (order %d)",
                    cluster.cluster_id, cluster.order,
                )

    # ------------------------------------------------------------ step 2.5

    def _step2_5_basis_precompute(self, clusters: list[ClusterEntry]) -> None:
        logging.info("=" * 80)
        logging.info(
            "Step 2.5: Orbit-basis precompute for streaming-symmetry (%d clusters)",
            len(clusters),
        )
        logging.info("=" * 80)
        tasks = [
            (self.geometry, self.args, c.cluster_id, c.order, self._ham_subdir_for(c))
            for c in clusters
        ]
        if getattr(self.args, "parallel", False):
            # Spawn + per-worker BLAS/OMP thread budgeting, mirroring the
            # ED stage. Default-fork Pool would inherit the parent's
            # OMP_NUM_THREADS in every worker, leading to
            # num_cores * num_cores threads thrashing the CPU during the
            # symmetry orbit precompute (which itself calls into qed
            # internals that link OpenMP/BLAS).
            total_cores = int(getattr(self.args, "num_cores", 1) or 1)
            workers = max(1, min(total_cores, len(tasks)))
            omp_threads = max(1, total_cores // workers)
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=workers,
                initializer=_ed_pool_initializer,
                initargs=(omp_threads,),
            ) as pool:
                results = pool.map(_basis_worker, tasks)
        else:
            results = [_basis_worker(t) for t in tasks]
        n_ok = sum(results)
        logging.info("Basis precompute: %d / %d clusters succeeded", n_ok, len(tasks))

    # ------------------------------------------------------------ step 3

    def _step3_ed(self, clusters: list[ClusterEntry]) -> None:
        logging.info("=" * 80)
        logging.info(
            "Step 3: Full dense exact diagonalization in-process "
            "(symmetry-adapted; pipeline=%s)", self.pipeline.name,
        )
        logging.info("=" * 80)

        jobs: list[tuple[ClusterEntry, _EDPayload]] = []
        for cluster in clusters:
            ham_subdir = self._ham_subdir_for(cluster)
            ed_subdir = self._ed_subdir_for(cluster)
            os.makedirs(os.path.join(ed_subdir, "output"), exist_ok=True)
            if not os.path.exists(ham_subdir):
                logging.warning("Hamiltonian dir missing: %s", ham_subdir)
                continue
            num_sites = count_sites_in_info_file(ham_subdir)
            if num_sites is None:
                logging.warning("Could not determine site count for %s", ham_subdir)
                continue
            options = self.pipeline.make_ed_options(self.args, num_sites)
            if not can_run_in_process(options.method):
                logging.error(
                    "Method '%s' is not supported by the in-process qed "
                    "backend (cluster %d, %d sites). MPI-only methods "
                    "(SCALAPACK*, mTPQ_MPI) require an external MPI "
                    "launcher; pick a non-MPI method.",
                    options.method, cluster.cluster_id, num_sites,
                )
                continue
            payload = _EDPayload(
                ham_subdir=ham_subdir,
                ed_subdir=ed_subdir,
                num_sites=num_sites,
                options=options,
                log_tag=f"{self.pipeline.name}:cluster_{cluster.cluster_id}",
                cache_dir=getattr(self.args, "cache_dir", None) or default_cache_dir(),
                cache_enabled=self.eig_cache.enabled,
                cache_key_extras={
                    "geometry": self.geometry.name,
                    "cluster_file": cluster.path,
                },
            )
            jobs.append((cluster, payload))

        if not jobs:
            logging.warning("No ED jobs to run.")
            return

        n_total = len(jobs)
        n_ok = 0

        # Decide serial vs parallel. The default is serial (back-compat
        # + single shared qed module + CUDA / OpenMP context). Going
        # parallel means a process pool with --num_cores workers, each
        # with reduced BLAS/OMP threads to avoid oversubscription.
        parallel_enabled = bool(getattr(self.args, "parallel", False))
        ed_workers = int(getattr(self.args, "ed_parallel_workers", 0) or 0)
        if parallel_enabled and ed_workers <= 0:
            ed_workers = int(getattr(self.args, "num_cores", 1) or 1)
        ed_workers = max(1, min(ed_workers, n_total))

        use_mpi = bool(getattr(self.args, "mpi", False)) and mpi_available()

        failed_tags: list[str] = []
        if use_mpi:
            # Cost-aware scatter_ed_jobs (mpi_dispatch.py) returns only an
            # aggregate count today -- per-cluster failure tags aren't
            # tracked across ranks. The require-complete gate below still
            # catches an incomplete sum, just without naming the culprits.
            n_ok = self._run_ed_mpi(jobs)
        elif not parallel_enabled or ed_workers == 1:
            n_ok, failed_tags = self._run_ed_serial(jobs)
        else:
            n_ok, failed_tags = self._run_ed_parallel(jobs, ed_workers)

        logging.info("ED: %d / %d clusters succeeded", n_ok, n_total)
        self.eig_cache.log_summary("eig-cache")
        self.subcluster_cache.log_summary("subcluster-cache")

        if n_ok < n_total and not bool(getattr(self.args, "allow_incomplete_ed", False)):
            if failed_tags:
                logging.error(
                    "ED step incomplete (%d / %d clusters failed): %s",
                    n_total - n_ok, n_total, ", ".join(failed_tags),
                )
            else:
                logging.error(
                    "ED step incomplete (%d / %d clusters failed); see ED "
                    "worker errors above for details.",
                    n_total - n_ok, n_total,
                )
            logging.error(
                "Aborting before NLCE summation -- a partial order-%d sum "
                "would silently look complete. Pass --allow_incomplete_ed "
                "to proceed anyway (e.g. for exploratory runs).",
                self.args.max_order,
            )
            sys.exit(1)

    # ---- serial / parallel ED dispatch helpers ----

    def _run_ed_serial(
        self, jobs: list[tuple[ClusterEntry, _EDPayload]]
    ) -> tuple[int, list[str]]:
        try:
            from tqdm import tqdm
            iterable = tqdm(jobs, desc="ED in-process")
        except ImportError:
            iterable = jobs
        n_ok = 0
        failed_tags: list[str] = []
        for _cluster, payload in iterable:
            ok = run_ed_in_process(
                payload.ham_subdir, payload.ed_subdir, payload.num_sites,
                payload.options, log_tag=payload.log_tag,
                cache=self.eig_cache,
                cache_key_extras=payload.cache_key_extras,
            )
            if ok:
                n_ok += 1
            else:
                failed_tags.append(payload.log_tag)
        return n_ok, failed_tags

    def _run_ed_parallel(
        self,
        jobs: list[tuple[ClusterEntry, _EDPayload]],
        ed_workers: int,
    ) -> tuple[int, list[str]]:
        # GPU oversubscription guard: N spawn workers x 1 GPU = N CUDA
        # contexts fighting for the same device memory -- OOM/context
        # thrash on exactly the biggest clusters. Serialize unless the
        # user explicitly claims multiple GPUs / MPS via env override.
        device = str(getattr(self.args, "device", "cpu")).lower()
        if device == "gpu" and ed_workers > 1 \
                and os.environ.get("QED_NLCE_GPU_PARALLEL") != "1":
            logging.warning(
                "--device gpu with %d parallel workers would create %d "
                "competing CUDA contexts on one GPU; forcing 1 worker. "
                "Set QED_NLCE_GPU_PARALLEL=1 to override (multi-GPU/MPS).",
                ed_workers, ed_workers,
            )
            ed_workers = 1

        # Pick BLAS/OMP threads-per-worker so total threads ~= num_cores.
        total_cores = int(getattr(self.args, "num_cores", 1) or 1)
        omp_threads = max(1, total_cores // ed_workers)
        logging.info(
            "Parallel ED: %d workers x %d BLAS/OMP threads each "
            "(%d clusters total)",
            ed_workers, omp_threads, len(jobs),
        )

        # Largest-first submission: with unordered completion, scheduling
        # the most expensive cluster last would extend the wall clock by
        # its full runtime; front-loading it overlaps it with the swarm
        # of small clusters.
        payloads = sorted(
            (p for (_c, p) in jobs), key=lambda p: -p.num_sites,
        )

        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(jobs), desc="ED parallel")
        except ImportError:
            pbar = None

        # ProcessPoolExecutor over a spawn context (fresh OMP/MKL/CUDA
        # state per worker, workers persist across tasks). Unlike
        # multiprocessing.Pool, a worker that dies abruptly (native
        # segfault in the C++ solver, OOM-kill) raises
        # BrokenProcessPool here instead of hanging imap_unordered
        # forever -- essential for unattended multi-day runs.
        import concurrent.futures as cf

        ctx = multiprocessing.get_context("spawn")
        n_ok = 0
        failed_tags: list[str] = []
        completed_tags: set[str] = set()
        try:
            with cf.ProcessPoolExecutor(
                max_workers=ed_workers,
                mp_context=ctx,
                initializer=_ed_pool_initializer,
                initargs=(omp_threads,),
            ) as pool:
                fut_to_tag = {
                    pool.submit(_ed_worker, p): p.log_tag for p in payloads
                }
                for fut in cf.as_completed(fut_to_tag):
                    try:
                        tag, ok, err = fut.result()
                    except cf.process.BrokenProcessPool:
                        raise
                    except Exception as exc:  # worker-side pickling etc.
                        tag, ok, err = fut_to_tag[fut], False, str(exc)
                    completed_tags.add(tag)
                    if ok:
                        n_ok += 1
                    else:
                        failed_tags.append(tag)
                        logging.error(
                            "ED worker %s failed%s",
                            tag, f": {err}" if err else "",
                        )
                    if pbar is not None:
                        pbar.update(1)
        except cf.process.BrokenProcessPool:
            logging.error(
                "A parallel ED worker process died abruptly (native crash "
                "or OOM kill). Completed clusters are preserved on disk; "
                "rerun to resume via the eigenvalue cache. Marking all "
                "unfinished clusters as failed."
            )
            for p in payloads:
                if p.log_tag not in completed_tags:
                    failed_tags.append(p.log_tag)
        finally:
            if pbar is not None:
                pbar.close()
        return n_ok, failed_tags

    def _run_ed_mpi(self, jobs: list[tuple[ClusterEntry, _EDPayload]]) -> int:
        """Distribute ED jobs across MPI ranks with cost-aware scheduling."""
        payloads = [p for (_c, p) in jobs]

        def _run_one(payload: _EDPayload) -> bool:
            cache = EigenvalueCache(
                payload.cache_dir or default_cache_dir(),
                enabled=payload.cache_enabled,
            )
            return bool(run_ed_in_process(
                payload.ham_subdir, payload.ed_subdir,
                payload.num_sites, payload.options,
                log_tag=payload.log_tag,
                cache=cache, cache_key_extras=payload.cache_key_extras,
            ))

        return scatter_ed_jobs(payloads, _run_one)

    # ------------------------------------------------------------ step 4

    def _step4_summation(self) -> None:
        logging.info("=" * 80)
        logging.info("Step 4: NLCE summation (pipeline=%s)", self.pipeline.name)
        logging.info("=" * 80)
        order_cutoff = getattr(self.args, "order_cutoff", None) or self.args.max_order
        cmd = self.pipeline.summation_command(
            self.args,
            cluster_info_dir=self.cluster_info_dir,
            ed_dir=self.ed_dir,
            nlc_dir=self.nlc_dir,
            order_cutoff=order_cutoff,
        )
        if cmd is None:
            logging.info("Pipeline %s declared no summation step.", self.pipeline.name)
            return
        logging.info("Running: %s", " ".join(cmd))
        # Tee the kernel's stdout/stderr into a persistent log: the
        # summation prints load-bearing diagnostics (OFTLM cancellation
        # warnings, temp-grid clamp warnings, SKIPPED clusters) that
        # would otherwise be lost in nohup/slurm runs. Also surface any
        # WARNING lines into the workflow log.
        sum_log = os.path.join(self.nlc_dir, "summation.log")
        try:
            proc = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
            )
            with open(sum_log, "w") as f:
                f.write(proc.stdout)
                if proc.stderr:
                    f.write("\n===== STDERR =====\n")
                    f.write(proc.stderr)
            n_warn = 0
            for line in proc.stdout.splitlines():
                if "WARNING" in line:
                    logging.warning("summation: %s", line.strip())
                    n_warn += 1
            logging.info(
                "NLCE summation completed (%d warnings; full output: %s).",
                n_warn, sum_log,
            )
        except subprocess.CalledProcessError as e:
            with open(sum_log, "w") as f:
                f.write(e.stdout or "")
                f.write("\n===== STDERR =====\n")
                f.write(e.stderr or "")
            logging.error(
                "NLCE summation FAILED (exit %s). Output: %s\nLast stderr:\n%s",
                e.returncode, sum_log,
                "\n".join((e.stderr or "").splitlines()[-15:]),
            )
            # A dead summation must not masquerade as a completed workflow
            # (unattended monitors key on the exit code).
            sys.exit(1)
