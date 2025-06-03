import argparse
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import dask.bag as db
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from tqdm import tqdm

from self_identification import SelfIdentificationDetector, detect_self_identification_in_entry
from helpers import get_all_jsonl_files, filter_entry, extract_columns  # Re-use shared helpers

logger = logging.getLogger("self_identify")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def process_file(
    file_path: str,
    detector: SelfIdentificationDetector,
    split: str,
    min_words: int,
    max_words: int,
) -> List[Dict[str, Any]]:
    """Stream a single JSONL Reddit file and return entries that contain self-identification."""
    results: List[Dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Apply the same filtering criteria as predefined for the project.
            if not filter_entry(entry, split=split, min_words=min_words, max_words=max_words):
                continue

            matches = detect_self_identification_in_entry(entry, detector)
            if not matches:
                continue  # no self-identification found

            # Skip entries with missing or deleted author – cannot collect posts later
            author_name = entry["author"]
            if (author_name is None) or (author_name == "[deleted]") or (author_name == ""):
                continue

            # Build structured output – ensure we always have a stable identifier.
            # Pushshift may not provide *author_id* in older dumps but does expose
            # a different user identifier called *author_fullname* with the same
            # semantic role. We attempt to use one or the other and log whenever
            # neither is available (indicating a potential data quality issue).
            #
            author_id = entry.get("author_id")
            author = entry.get("author")
                        
            result = {
                "author": author,
                "author_id": author_id,
                "self_identification": matches,
                "post": extract_columns(entry, None),
            }
            results.append(result)

    return results


def run_pipeline(
    input_dir: str,
    output_jsonl: str,
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
):
    """Main entry point: Detect self-identified users and write them to *output_jsonl*."""
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    # Gather all JSONL files
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_* files found inside {input_dir}")
    logger.info(f"Found {len(files)} JSONL files to scan for self-identification.")

    # Initialize detector once (can be broadcasted to workers implicitly)
    detector = SelfIdentificationDetector()

    # Configure Dask cluster
    if use_slurm:
        # Ensure log directory exists for worker logs as well
        os.makedirs("logs", exist_ok=True)
        logger.info(
            f"Using SLURM cluster with {n_workers} workers, memory {memory_per_worker} – dask logs in logs/"
        )
        cluster = SLURMCluster(
            cores=1,
            processes=1,
            memory=memory_per_worker,
            walltime="1-00:00:00",
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

    bag = db.from_sequence(files, npartitions=len(files))
    processed_bag = bag.map(
        lambda fp: process_file(fp, detector=detector, split=split, min_words=min_words, max_words=max_words)
    ).flatten()

    with ProgressBar():
        results: List[Dict[str, Any]] = processed_bag.compute()

    logger.info(f"Detected {len(results)} self-identification posts. Writing to {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for item in tqdm(results, desc="Writing output"):
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    client.close()
    if use_slurm:
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect self-identified users in Reddit JSONL dump")
    parser.add_argument("--input_dir", required=True, help="Directory containing RS_*.jsonl files")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file for self-identification matches")
    parser.add_argument("--split", choices=["text", "multimodal"], default="text")
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--memory_per_worker", type=str, default="4GB")
    parser.add_argument("--use_slurm", action="store_true")

    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_jsonl=args.output_jsonl,
        split=args.split,
        min_words=args.min_words,
        max_words=args.max_words,
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm,
    ) 