"""High-level NLCE workflow orchestrator.

:class:`NLCEWorkflow` composes one :class:`Geometry` with one
:class:`Pipeline` and runs the canonical four-step pipeline:

  1. **Cluster generation** -- delegated to ``Geometry.generate_clusters``.
  2. **Hamiltonian preparation** -- delegated to
     ``Geometry.prepare_hamiltonian`` (one call per cluster).
  3. **Exact diagonalization** -- per cluster, the workflow asks
     ``Pipeline.make_ed_options(args, num_sites)`` for the per-cluster
     options, builds the argv via :func:`build_ed_command`, and runs
     it through :func:`run_ed_subprocess`.
  4. **NLCE summation** -- delegated to ``Pipeline.summation_command``.

The orchestrator also handles:
  * the ``--streaming-symmetry`` orbit basis precompute step (delegated
    to ``Geometry.precompute_basis``),
  * directory layout (``clusters_*``, ``hamiltonians_*``, ``ed_results_*``,
    ``nlc_results_*``),
  * per-cluster thermo plotting (when ``--thermo`` is on),
  * parallelism (``--parallel`` + ``--num_cores``).

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

from .ed_runner import build_ed_command, run_ed_subprocess
from .qed_backend import can_run_in_process, qed_available, run_ed_in_process
from .geometry import Geometry
from .io import (
    ClusterEntry,
    count_sites_in_info_file,
    get_cluster_files,
    setup_logging,
)
from .pipeline import Pipeline


__all__ = ["NLCEWorkflow"]


# Worker for multiprocessing.Pool (must be top-level / picklable).
def _ed_worker(payload: tuple) -> bool:
    """Pure function form of one ED launch. Picklable for multiprocessing."""
    (cmd, output_root, extra_env, log_tag) = payload
    return run_ed_subprocess(
        cmd,
        output_root=output_root,
        extra_env=extra_env,
        log_tag=log_tag,
    )


def _ed_inproc_worker(payload: tuple) -> bool:
    """Pure function form of one in-process ED launch."""
    (ham_subdir, output_dir, num_sites, options, log_tag) = payload
    return run_ed_in_process(
        ham_subdir, output_dir, num_sites, options, log_tag=log_tag,
    )


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
        logging.info("Step 3: Exact diagonalization (pipeline=%s)", self.pipeline.name)
        logging.info("=" * 80)

        # In-process backend selection. Three modes:
        #   --in_process       : require qed package, error if not usable.
        #   --auto_in_process  : prefer in-process, transparently fall back.
        #   (default)          : subprocess only (legacy behavior).
        in_process = bool(getattr(self.args, "in_process", False))
        auto_in_process = bool(getattr(self.args, "auto_in_process", False))
        use_inproc_default = in_process or auto_in_process

        if use_inproc_default and not qed_available():
            if in_process:
                logging.error(
                    "--in_process requires the 'qed' Python package "
                    "(install via `pip install qed_nlce[qed]` or `pip install qed`)."
                )
                sys.exit(1)
            logging.warning(
                "--auto_in_process: qed package not importable; "
                "falling back to ./ED subprocess for all clusters."
            )
            use_inproc_default = False

        ed_executable = self.args.ed_executable
        # Only require the ED binary if at least one cluster will use it.
        # We re-check below per-cluster.
        scalapack_threshold = getattr(self.args, "scalapack_threshold", 16)
        use_scalapack = not getattr(self.args, "no_scalapack", False)

        # Build all ED jobs up front so we can use a Pool cleanly.
        subprocess_jobs: list[tuple] = []
        inproc_jobs: list[tuple] = []
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

            # Decide per-cluster whether to use the in-process backend.
            # Auto-promotion to SCALAPACK_MIXED happens inside
            # build_ed_command, so we have to anticipate that here.
            promoted_method = options.method
            if (
                options.method.upper() == "FULL"
                and use_scalapack
                and num_sites >= scalapack_threshold
            ):
                promoted_method = "SCALAPACK_MIXED"

            cluster_can_inproc = (
                use_inproc_default and can_run_in_process(promoted_method)
            )
            log_tag = f"{self.pipeline.name}:cluster_{cluster.cluster_id}"

            if cluster_can_inproc:
                inproc_jobs.append(
                    (ham_subdir, ed_subdir, num_sites, options, log_tag)
                )
            else:
                if not os.path.exists(ed_executable):
                    logging.error(
                        "ED executable not found at %s. Build the project "
                        "first (see https://github.com/ze-bang/QED), or pass "
                        "--ed_executable=<path>, or set $QED_ED_BINARY.",
                        ed_executable,
                    )
                    sys.exit(1)
                cmd = build_ed_command(
                    ed_executable=ed_executable,
                    ham_subdir=ham_subdir,
                    output_dir=ed_subdir,
                    num_sites=num_sites,
                    options=options,
                    scalapack_threshold=scalapack_threshold,
                    use_scalapack=use_scalapack,
                )
                extra_env = self.pipeline.extra_env(self.args, num_sites)
                subprocess_jobs.append((cmd, ed_subdir, extra_env, log_tag))

        n_total = len(subprocess_jobs) + len(inproc_jobs)
        if n_total == 0:
            logging.warning("No ED jobs to run.")
            return

        logging.info(
            "ED job dispatch: %d in-process (qed), %d subprocess (./ED)",
            len(inproc_jobs), len(subprocess_jobs),
        )

        results: list[bool] = []
        parallel = getattr(self.args, "parallel", False)
        n_cores = self.args.num_cores

        # In-process jobs: run sequentially in the main interpreter to
        # share the qed module + its CUDA / OpenMP runtime context. (A
        # multiprocessing.Pool would re-import qed in every worker which
        # defeats the purpose.) For small clusters this is still much
        # faster than forking ./ED.
        if inproc_jobs:
            try:
                from tqdm import tqdm
                iterable = tqdm(inproc_jobs, desc="ED in-process")
            except ImportError:
                iterable = inproc_jobs
            results.extend(_ed_inproc_worker(j) for j in iterable)

        # Subprocess jobs: optionally parallel.
        if subprocess_jobs:
            if parallel:
                logging.info(
                    "Running %d subprocess ED jobs in parallel with %d cores",
                    len(subprocess_jobs), n_cores,
                )
                with multiprocessing.Pool(processes=n_cores) as pool:
                    results.extend(pool.map(_ed_worker, subprocess_jobs))
            else:
                try:
                    from tqdm import tqdm
                    iterable = tqdm(subprocess_jobs, desc="Running ED")
                except ImportError:
                    iterable = subprocess_jobs
                results.extend(_ed_worker(j) for j in iterable)

        n_ok = sum(results)
        logging.info("ED: %d / %d clusters succeeded", n_ok, n_total)

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
