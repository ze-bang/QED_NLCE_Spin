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
from typing import Optional

from .qed_backend import can_run_in_process, run_ed_in_process
from .cache import EigenvalueCache, SubclusterCache, default_cache_dir
from .geometry import Geometry
from .io import (
    ClusterEntry,
    count_sites_in_info_file,
    get_cluster_files,
    setup_logging,
)
from .pipeline import Pipeline


__all__ = ["NLCEWorkflow"]


def _basis_worker(payload: tuple) -> bool:
    """Pure function form of one basis precompute call."""
    geometry, args, cluster_id, order, ham_subdir = payload
    return geometry.precompute_basis(args, cluster_id, order, ham_subdir)


class NLCEWorkflow:
    """One ``Geometry`` × one ``Pipeline``  =  one NLCE run."""

    def __init__(
        self,
        geometry: Geometry,
        pipeline: Pipeline,
        args: argparse.Namespace,
    ):
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
        log_file = os.path.join(self.base_dir, "nlce_workflow.log")
        setup_logging(log_file)

        self._banner_top()

        if not self.args.skip_cluster_gen:
            self._step1_clusters()
        else:
            logging.info("Skipping cluster generation step.")

        clusters = self._discover_clusters()

        if not self.args.skip_ham_prep:
            self._step2_hamiltonians(clusters)
        else:
            logging.info("Skipping Hamiltonian preparation step.")

        if self.pipeline.needs_basis_precompute(self.args) and not getattr(
            self.args, "skip_basis_precompute", False
        ):
            self._step2_5_basis_precompute(clusters)

        if not self.args.skip_ed:
            self._step3_ed(clusters)
        else:
            logging.info("Skipping ED step.")

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

    def _step1_clusters(self) -> None:
        logging.info("=" * 80)
        logging.info("Step 1: Cluster generation (%s)", self.geometry.name)
        logging.info("=" * 80)
        ok = self.geometry.generate_clusters(self.args, self.args.max_order, self.cluster_dir)
        if not ok:
            logging.error("Cluster generation failed for geometry=%s", self.geometry.name)
            sys.exit(1)
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
            with multiprocessing.Pool(processes=self.args.num_cores) as pool:
                results = pool.map(_basis_worker, tasks)
        else:
            results = [_basis_worker(t) for t in tasks]
        n_ok = sum(results)
        logging.info("Basis precompute: %d / %d clusters succeeded", n_ok, len(tasks))

    # ------------------------------------------------------------ step 3

    def _step3_ed(self, clusters: list[ClusterEntry]) -> None:
        logging.info("=" * 80)
        logging.info(
            "Step 3: Exact diagonalization in-process via `import qed` "
            "(pipeline=%s)", self.pipeline.name,
        )
        logging.info("=" * 80)

        n_total = 0
        n_ok = 0
        try:
            from tqdm import tqdm
            iterable = tqdm(clusters, desc="ED in-process")
        except ImportError:
            iterable = clusters

        for cluster in iterable:
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

            log_tag = f"{self.pipeline.name}:cluster_{cluster.cluster_id}"
            n_total += 1
            ok = run_ed_in_process(
                ham_subdir, ed_subdir, num_sites, options, log_tag=log_tag,
                cache=self.eig_cache,
                cache_key_extras={
                    "geometry": self.geometry.name,
                    "cluster_file": cluster.path,
                },
            )
            if ok:
                n_ok += 1

        if n_total == 0:
            logging.warning("No ED jobs to run.")
            return
        logging.info("ED: %d / %d clusters succeeded", n_ok, n_total)
        self.eig_cache.log_summary("eig-cache")
        self.subcluster_cache.log_summary("subcluster-cache")

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
        try:
            subprocess.run(cmd, check=True)
            logging.info("NLCE summation completed.")
        except subprocess.CalledProcessError as e:
            logging.error("NLCE summation failed: %s", e)
