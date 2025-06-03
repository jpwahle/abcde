import argparse
import json
import logging
import os
import csv
from typing import Dict, Any, List, Optional, Tuple

import dask.bag as db
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from tqdm import tqdm

from self_identification import SelfIdentificationDetector, detect_self_identification_with_resolved_age
from helpers import get_all_jsonl_files, filter_entry, extract_columns

logger = logging.getLogger("self_identify")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def flatten_result_to_csv_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested result structure for CSV output with CapitalCase headers."""
    row = {}
    
    # Author information
    row["Author"] = result.get("author", "")
    
    # Self-identification with resolved age
    self_id = result.get("self_identification", {})
    
    # Use resolved age if available, otherwise fallback to first raw age
    if "resolved_age" in self_id:
        resolved = self_id["resolved_age"]
        row["SelfIdentificationAgeMajorityVote"] = resolved.get("age", "")
        row["SelfIdentificationRawAges"] = "|".join(map(str, resolved.get("raw_matches", [])))
    else:
        # Fallback to first age from raw matches
        age_matches = self_id.get("age", [])
        row["SelfIdentificationAgeMajorityVote"] = age_matches[0] if age_matches else ""
        row["SelfIdentificationRawAges"] = "|".join(map(str, age_matches))
    
    # Post information with Post prefix
    post = result.get("post", {})
    row["PostID"] = post.get("id", "")
    row["PostSubreddit"] = post.get("subreddit", "")
    row["PostTitle"] = post.get("title", "")
    row["PostSelftext"] = post.get("selftext", "")
    row["PostCreatedUtc"] = post.get("created_utc", "")
    row["PostScore"] = post.get("score", "")
    row["PostNumComments"] = post.get("num_comments", "")
    row["PostPermalink"] = post.get("permalink", "")
    row["PostUrl"] = post.get("url", "")
    row["PostMediaPath"] = post.get("media_path", "")
    
    return row


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

            # Use resolved age detection
            matches = detect_self_identification_with_resolved_age(entry, detector)
            if not matches:
                continue  # no self-identification found

            # Skip entries with missing or deleted author – cannot collect posts later
            author_name = entry["author"]
            if (author_name is None) or (author_name == "[deleted]") or (author_name == ""):
                continue

            # Skip automated accounts - AutoModerator and Bot entries
            if author_name in ("AutoModerator", "Bot"):
                continue

            author = entry.get("author")
                        
            result = {
                "author": author,
                "self_identification": matches,
                "post": extract_columns(entry, None),
            }
            results.append(result)

    return results


def run_pipeline(
    input_dir: str,
    output_csv: str,
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
    output_tsv: bool = False,
):
    """Main entry point: Detect self-identified users and write them to CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

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

    logger.info(f"Detected {len(results)} self-identification posts. Writing to {output_csv}")
    
    if results:
        # Flatten results for CSV/TSV
        csv_rows = [flatten_result_to_csv_row(result) for result in results]
        
        # Write to CSV or TSV
        separator = '\t' if output_tsv else ','
        file_extension = 'tsv' if output_tsv else 'csv'
        output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
        
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            fieldnames = [
                "Author", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
                "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
                "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()
            
            for row in tqdm(csv_rows, desc=f"Writing {file_extension.upper()}"):
                writer.writerow(row)
    else:
        logger.warning("No results found. Creating empty CSV file.")
        # Determine output file and separator for empty file case
        separator = '\t' if output_tsv else ','
        file_extension = 'tsv' if output_tsv else 'csv'
        output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
        
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            fieldnames = [
                "Author", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
                "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
                "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()

    client.close()
    if use_slurm:
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect self-identified users in Reddit JSONL dump")
    parser.add_argument("--input_dir", required=True, help="Directory containing RS_*.jsonl files")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for self-identification matches")
    parser.add_argument("--split", choices=["text", "multimodal"], default="text")
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--memory_per_worker", type=str, default="4GB")
    parser.add_argument("--use_slurm", action="store_true")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")

    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        split=args.split,
        min_words=args.min_words,
        max_words=args.max_words,
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm,
        output_tsv=args.output_tsv,
    ) 