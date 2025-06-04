"""
Common cluster management utilities for Dask SLURM and local clusters.
"""
import os
import logging
from typing import Optional, Union
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

logger = logging.getLogger("core.cluster")


def setup_dask_cluster(
    n_workers: int,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
    walltime: str = "1-00:00:00"
) -> Client:
    """Setup and return a Dask client with appropriate cluster configuration.
    
    Args:
        n_workers: Number of workers to use
        memory_per_worker: Memory per worker (e.g., "4GB")
        use_slurm: Whether to use SLURM cluster
        walltime: Walltime for SLURM jobs
        
    Returns:
        Configured Dask client
    """
    if use_slurm:
        # Ensure log directory exists for worker logs
        os.makedirs("logs", exist_ok=True)
        logger.info(
            f"Using SLURM cluster with {n_workers} workers, memory {memory_per_worker} – dask logs in logs/"
        )
        cluster = SLURMCluster(
            cores=1,
            processes=1,
            memory=memory_per_worker,
            walltime=walltime,
            job_extra=[
                "--cpus-per-task=1",
                "-o",
                "logs/dask-%j.out",
                "-e",
                "logs/dask-%j.err",
            ],
        )
        cluster.scale(n_workers)
        client = Client(cluster)
    else:
        logger.info(f"Starting local Dask cluster with {n_workers} workers")
        client = Client(n_workers=n_workers, threads_per_worker=1, memory_limit=memory_per_worker)
    
    logger.info(f"Dashboard: {client.dashboard_link}")
    return client


def cleanup_cluster(client: Client, use_slurm: bool = False):
    """Clean up Dask cluster and client.
    
    Args:
        client: Dask client to close
        use_slurm: Whether SLURM cluster was used
    """
    try:
        if use_slurm and hasattr(client, 'cluster'):
            client.cluster.close()
        client.close()
    except Exception as e:
        logger.warning(f"Error during cluster cleanup: {e}")


def optimize_workers_for_data_size(
    data_size_estimate: Optional[int] = None,
    chunk_size: int = 100000,
    max_workers: int = 128,
    memory_per_worker: str = "4GB"
) -> int:
    """Calculate optimal number of workers based on data size and memory constraints.
    
    Args:
        data_size_estimate: Estimated number of rows in dataset
        chunk_size: Target chunk size per partition
        max_workers: Maximum number of workers to use
        memory_per_worker: Memory available per worker
        
    Returns:
        Optimal number of workers
    """
    if data_size_estimate is None:
        return min(32, max_workers)  # Conservative default
    
    # Calculate workers based on chunk size
    optimal_workers = max(1, min(data_size_estimate // chunk_size, max_workers))
    
    # For very large datasets, cap workers to avoid overwhelming scheduler
    if data_size_estimate > 10_000_000:  # 10M rows
        optimal_workers = min(optimal_workers, 64)
    
    logger.info(f"Calculated optimal workers: {optimal_workers} for ~{data_size_estimate} rows")
    return optimal_workers