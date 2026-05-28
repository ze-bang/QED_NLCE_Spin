"""MPI-based cluster distribution for NLCE ED step.

Provides embarrassingly-parallel distribution of per-cluster ED jobs
across MPI ranks. Each rank processes a subset of clusters using the
existing in-process ``qed.solve`` / ``qed.thermal`` backend.

Usage (HPC job script)::

    mpiexec -n $SLURM_NTASKS python -m qed_nlce --geometry pyrochlore \\
        --pipeline auto --max_order 8 --mpi --base_dir output/pyro_mpi_o8

The ``--mpi`` flag activates this module. Steps 1, 2, and 4 (cluster
generation, Hamiltonian prep, NLCE summation) run on rank 0 only.
Step 3 (ED) is scattered across all ranks with cost-aware scheduling.

Cost-aware scheduling sorts clusters by descending Hilbert-space
dimension and distributes them round-robin (largest-first). This
avoids the "long tail" problem where all ranks finish except one
stuck on the final large cluster.

Thread budget: each rank pins BLAS/OMP threads to
``total_cores_per_node / ranks_per_node`` to avoid oversubscription.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow import _EDPayload


def _pin_threads_for_mpi() -> int:
    """Set BLAS/OMP thread budget based on available cores and local rank count.

    Returns the number of threads per rank (for logging).
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        return 1

    cores = os.cpu_count() or 1

    # Determine ranks on this node via shared-memory communicator
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_size = local_comm.Get_size()
    local_comm.Free()

    threads = max(1, cores // local_size)
    val = str(threads)
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[k] = val
    return threads


def mpi_available() -> bool:
    """True if mpi4py is importable and MPI was initialized with >1 rank."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        return False


def scatter_ed_jobs(
    payloads: "list[_EDPayload]",
    run_one_fn,
    *,
    cache_enabled: bool = True,
    cache_dir: str | None = None,
) -> int:
    """Distribute ED payloads across MPI ranks and return total successes.

    ``run_one_fn(payload) -> bool`` is the per-cluster ED function
    (typically ``run_ed_in_process`` wrapped with cache logic).

    Cost-aware scheduling: sort by descending Hilbert dim (2^num_sites),
    then distribute round-robin so the biggest clusters start first.

    Returns the global count of successful clusters (reduced to rank 0).
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    threads = _pin_threads_for_mpi()
    if rank == 0:
        logging.info(
            "MPI ED dispatch: %d ranks, %d threads/rank, %d clusters",
            size, threads, len(payloads),
        )

    # Rank 0 does the scheduling; others receive their subset
    if rank == 0:
        # Sort by descending estimated cost (2^num_sites)
        indexed = sorted(
            enumerate(payloads),
            key=lambda x: -(2 ** x[1].num_sites),
        )
        # Round-robin assignment (largest first)
        assignments: list[list[int]] = [[] for _ in range(size)]
        for i, (orig_idx, _) in enumerate(indexed):
            assignments[i % size].append(orig_idx)
    else:
        assignments = None

    # Scatter the index lists
    local_indices = comm.scatter(assignments, root=0)

    # Broadcast the full payload list (they are small dataclasses)
    payloads = comm.bcast(payloads, root=0)

    # Each rank processes its assigned clusters
    local_ok = 0
    for idx in local_indices:
        payload = payloads[idx]
        try:
            ok = run_one_fn(payload)
            if ok:
                local_ok += 1
            else:
                logging.error(
                    "[rank %d] ED failed for %s", rank, payload.log_tag,
                )
        except Exception as exc:
            logging.error(
                "[rank %d] ED exception for %s: %s",
                rank, payload.log_tag, exc,
            )

    # Reduce success count to rank 0
    total_ok = comm.reduce(local_ok, op=MPI.SUM, root=0)
    return total_ok if rank == 0 else 0


def is_rank_zero() -> bool:
    """True if this is MPI rank 0 (or MPI is not active)."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def barrier() -> None:
    """MPI barrier (no-op if MPI unavailable)."""
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass
