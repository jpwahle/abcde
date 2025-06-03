import argparse
import json
import logging
import os
import csv
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import dask.bag as db
import dask.dataframe as dd
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


def flatten_result_to_csv_row(result: Dict[str, Any], data_source: str = "reddit") -> Dict[str, Any]:
    """Flatten nested result structure for CSV output with CapitalCase headers."""
    row = {}
    
    # Author information - adapt to data source
    if data_source == "tusc":
        # TUSC uses different user ID fields depending on split
        row["Author"] = result.get("UserID") or result.get("userID", "")
        row["AuthorName"] = result.get("UserName") or result.get("userName", "")
    else:  # reddit
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
    
    # Post information with Post prefix - adapt to data source
    if data_source == "tusc":
        row["PostID"] = result.get("TweetID", "")
        row["PostText"] = result.get("Tweet", "")
        row["PostCreatedAt"] = result.get("createdAt", "")
        row["PostYear"] = result.get("Year", "")
        row["PostMonth"] = result.get("Month", "")
        
        # Location info varies by TUSC split
        if "City" in result:  # city split
            row["PostCity"] = result.get("City", "")
        if "Country" in result:  # country split
            row["PostCountry"] = result.get("Country", "")
            row["PostMyCountry"] = result.get("MyCountry", "")
        row["PostPlace"] = result.get("Place", "")
        row["PostPlaceID"] = result.get("PlaceID", "")
        row["PostPlaceType"] = result.get("PlaceType", "")
    else:  # reddit
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


def detect_self_identification_in_tusc_entry(entry: Dict[str, Any], detector: "SelfIdentificationDetector") -> Dict[str, Any]:
    """Detect self identification inside a TUSC entry.
    
    TUSC entries have 'Tweet' field instead of 'title' and 'selftext'.
    """
    tweet_text = entry.get("Tweet", "") or ""
    
    # Create a minimal entry structure compatible with detector
    adapted_entry = {
        "title": "",  # TUSC doesn't have titles
        "selftext": tweet_text  # Use tweet text as body
    }
    
    return detect_self_identification_with_resolved_age(adapted_entry, detector)


def process_tusc_batch(
    df_batch: pd.DataFrame,
    detector: SelfIdentificationDetector,
    split: str,
    min_words: int,
    max_words: int,
) -> List[Dict[str, Any]]:
    """Process a batch of TUSC data and return entries that contain self-identification."""
    results: List[Dict[str, Any]] = []

    for _, row in df_batch.iterrows():
        # Convert row to dict
        entry = row.to_dict()
        
        # Basic filtering
        tweet_text = entry.get("Tweet", "")
        if not tweet_text or pd.isna(tweet_text):
            continue
            
        # Word count filtering
        word_count = len(tweet_text.split())
        if word_count < min_words or word_count > max_words:
            continue

        # Detect self-identification
        matches = detect_self_identification_in_tusc_entry(entry, detector)
        if not matches:
            continue  # no self-identification found

        # Skip entries with missing user information
        user_id = entry.get("UserID" if split == "country" else "userID", "")
        if not user_id or pd.isna(user_id):
            continue

        # Create result combining entry data with self-identification
        result = entry.copy()
        result["self_identification"] = matches
        results.append(result)

    return results


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


def run_reddit_pipeline(
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
    """Run pipeline for Reddit JSONL files."""
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
    
    _write_results_to_csv(results, output_csv, output_tsv, data_source="reddit")

    client.close()
    if use_slurm:
        cluster.close()


def run_tusc_pipeline(
    input_file: str,
    output_csv: str,
    split: str = None,  # Will be auto-determined from filename
    min_words: int = 5,
    max_words: int = 1000,
    chunk_size: int = 100000,
    test_mode: bool = False,
    test_samples: int = 10000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
    output_tsv: bool = False,
):
    """Run pipeline for TUSC parquet file."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Auto-determine split type from filename if not provided
    if split is None:
        filename = Path(input_file).name.lower()
        if 'city' in filename:
            split = "city"
        elif 'country' in filename:
            split = "country"
        else:
            # Default to country if unclear
            split = "country"
            logger.warning(f"Could not determine split type from filename '{filename}', defaulting to 'country'")
    
    logger.info(f"Using split type: {split}")

    # Initialize detector
    detector = SelfIdentificationDetector()

    # Configure Dask cluster
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

    try:
        # Read parquet file with Dask DataFrame
        logger.info(f"Reading {input_file} with Dask DataFrame")
        
        if test_mode:
            # In test mode, read only a subset
            df = pd.read_parquet(input_file, engine='pyarrow')
            df = df.head(test_samples)
            ddf = dd.from_pandas(df, npartitions=min(n_workers, len(df) // chunk_size + 1))
            logger.info(f"Test mode: Processing {len(df)} samples")
        else:
            # Read with Dask and partition appropriately
            ddf = dd.read_parquet(input_file, engine='pyarrow')
            # Repartition to have reasonable chunk sizes
            ddf = ddf.repartition(partition_size=f"{chunk_size} rows")
            logger.info(f"Processing {len(ddf)} rows with {ddf.npartitions} partitions")

        # Process partitions to find self-identified users
        def process_partition(df_partition):
            """Process a single partition to find self-identified users.

            Returning a pandas.Series (object dtype) instead of a raw list allows
            the Dask scheduler to concatenate the individual partition results
            without tripping over unknown types during dispatch. Each series
            contains a single element – the list of result dictionaries for
            that particular partition.
            """
            if df_partition.empty:
                # Return an empty Series; keeps metadata consistent
                return pd.Series([], dtype="object")

            results_list = process_tusc_batch(df_partition, detector, split, min_words, max_words)

            # Wrap the list in a single-row Series so that Dask receives a
            # pandas object (it knows how to concatenate these), while we can
            # still easily unpack the lists after .compute().
            return pd.Series([results_list], dtype="object")

        # Apply processing function to each partition
        logger.info("Starting parallel processing...")
        with ProgressBar():
            # Process all partitions and collect results
            partition_results = ddf.map_partitions(
                process_partition,
                meta=pd.Series([], dtype="object")
            ).compute()
            
            # Flatten the list of lists
            results = []
            for partition_result in partition_results:
                if isinstance(partition_result, list):
                    results.extend(partition_result)

        logger.info(f"Detected {len(results)} self-identification posts. Writing to {output_csv}")
        
        _write_results_to_csv(results, output_csv, output_tsv, data_source="tusc", split=split)

    finally:
        # Clean up cluster
        client.close()
        if use_slurm:
            cluster.close()


def _write_results_to_csv(results: List[Dict[str, Any]], output_csv: str, output_tsv: bool, data_source: str = "reddit", split: str = None):
    """Helper function to write results to CSV/TSV with appropriate headers."""
    if results:
        # Flatten results for CSV/TSV
        csv_rows = [flatten_result_to_csv_row(result, data_source) for result in results]
        
        # Write to CSV or TSV
        separator = '\t' if output_tsv else ','
        file_extension = 'tsv' if output_tsv else 'csv'
        output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
        
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            # Determine fieldnames based on data source
            if data_source == "tusc":
                base_fieldnames = [
                    "Author", "AuthorName", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
                    "PostID", "PostText", "PostCreatedAt", "PostYear", "PostMonth"
                ]
                
                if split == "city":
                    location_fieldnames = ["PostCity", "PostPlace", "PostPlaceID", "PostPlaceType"]
                else:  # country
                    location_fieldnames = ["PostCountry", "PostMyCountry", "PostPlace", "PostPlaceID", "PostPlaceType"]
                
                fieldnames = base_fieldnames + location_fieldnames
            else:  # reddit
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
            # Create empty file with appropriate headers
            if data_source == "tusc":
                base_fieldnames = [
                    "Author", "AuthorName", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
                    "PostID", "PostText", "PostCreatedAt", "PostYear", "PostMonth"
                ]
                if split == "city":
                    location_fieldnames = ["PostCity", "PostPlace", "PostPlaceID", "PostPlaceType"]
                else:
                    location_fieldnames = ["PostCountry", "PostMyCountry", "PostPlace", "PostPlaceID", "PostPlaceType"]
                fieldnames = base_fieldnames + location_fieldnames
            else:
                fieldnames = [
                    "Author", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
                    "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
                    "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath"
                ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect self-identified users in data (Reddit JSONL or TUSC parquet)")
    
    # Data source selection
    parser.add_argument("--data_source", choices=["reddit", "tusc"], default="reddit", help="Data source type")
    
    # Reddit-specific arguments
    parser.add_argument("--input_dir", help="Directory containing RS_*.jsonl files (for Reddit)")
    
    # TUSC-specific arguments
    parser.add_argument("--input_file", help="Path to input parquet file (for TUSC)")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for Dask DataFrame partitions (TUSC only)")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with limited samples (TUSC only)")
    parser.add_argument("--test_samples", type=int, default=10000, help="Number of samples in test mode (TUSC only)")
    
    # Common arguments
    parser.add_argument("--output_csv", required=True, help="Output CSV file for self-identification matches")
    parser.add_argument("--split", help="Split type: 'text'/'multimodal' for Reddit, 'city'/'country' for TUSC (auto-determined if not specified)")
    parser.add_argument("--min_words", type=int, default=5, help="Minimum word count filter")
    parser.add_argument("--max_words", type=int, default=1000, help="Maximum word count filter")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")

    args = parser.parse_args()
    
    if args.data_source == "reddit":
        if not args.input_dir:
            parser.error("--input_dir is required for Reddit data source")
        if not args.split:
            args.split = "text"  # Default for Reddit
            
        run_reddit_pipeline(
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
    elif args.data_source == "tusc":
        if not args.input_file:
            parser.error("--input_file is required for TUSC data source")
            
        run_tusc_pipeline(
            input_file=args.input_file,
            output_csv=args.output_csv,
            split=args.split,  # Will be auto-determined if None
            min_words=args.min_words,
            max_words=args.max_words,
            chunk_size=args.chunk_size,
            test_mode=args.test_mode,
            test_samples=args.test_samples,
            n_workers=args.n_workers,
            memory_per_worker=args.memory_per_worker,
            use_slurm=args.use_slurm,
            output_tsv=args.output_tsv,
        ) 