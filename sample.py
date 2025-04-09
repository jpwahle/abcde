#!/usr/bin/env python3
import os
import json
import random
import argparse
import requests
from typing import Dict, Any, Generator, List, Optional, Tuple
from tqdm import tqdm
import dask.bag as db
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar
import math
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("reddit_sampler")

# --------------- #
# STAGE 1: LOADING & SAMPLING
# --------------- #


def get_all_jsonl_files(directory: str) -> List[str]:
    """
    Retrieve all .jsonl files from a given directory (non-recursive).
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("RS_") and not os.path.isdir(os.path.join(directory, f))
    ]


def count_lines(file_path: str) -> int:
    """Count lines in a file efficiently."""
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


def stream_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Given a .jsonl file path, yield each line as a Python dict."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def stream_jsonl_with_probability(
    file_path: str, sample_probability: float, seed: int
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream items from a JSONL file, yielding each with the given probability.
    This implements i.i.d. sampling directly during the stream.
    """
    random_gen = random.Random(seed)
    for item in stream_jsonl(file_path):
        if random_gen.random() < sample_probability:
            yield item


def sample_file_with_percentage(
    file_path: str,
    sample_percentage: float,
    split: str,
    min_words: int,
    max_words: int,
    download_dir: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample a percentage of entries from a single file, then filter and process them.
    Returns a list of processed entries that pass all filters.
    """
    # Convert percentage to probability (0-1 range)
    sample_probability = sample_percentage / 100.0

    # Add some randomness to the seed based on the filename to prevent correlation across files
    file_seed = seed + hash(file_path) % 10000

    results = []
    # Sample entries with the given probability
    for entry in stream_jsonl_with_probability(
        file_path, sample_probability, file_seed
    ):
        processed = process_entry(entry, split, min_words, max_words, download_dir)
        if processed:
            results.append(processed)

    return results


# --------------- #
# STAGE 2: FILTERING
# --------------- #


def filter_entry(
    entry: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
) -> bool:
    """
    Return True if this entry passes all filters, else False.
    """
    # 1. Over 18 check
    if entry.get("over_18", False):
        return False

    # 2. Promoted check
    if entry.get("promoted") is True:
        return False
    if entry.get("whitelist_status") == "promo_specified":
        return False

    # 3. Title + selftext checks
    selftext = entry.get("selftext", "")
    # Remove if there's no body text
    if not selftext.strip():
        return False

    # 4. Word count check
    word_count = len(selftext.strip().split())
    if word_count < min_words or word_count > max_words:
        return False

    # 5. Media checks
    has_video = bool(entry.get("is_video", False))
    url = entry.get("url", "")
    has_image = any(
        url.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".gif"]
    )

    if split == "text":
        # Must NOT have images or videos
        if has_video or has_image:
            return False
    elif split == "multimodal":
        # Must HAVE images or videos
        if not (has_video or has_image):
            return False

    return True


def verify_media_availability(entry: Dict[str, Any]) -> bool:
    """Verify media accessibility with HEAD request."""
    url = entry.get("url", "")
    if not url:
        return True

    try:
        resp = requests.head(url, timeout=5)
        return resp.status_code in (200, 302)
    except requests.RequestException:
        return False


# --------------- #
# STAGE 3: DOWNLOADING (OPTIONAL for multimodal)
# --------------- #


def download_media(entry: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Download media to out_dir and return local file path."""
    url = entry.get("url", "")
    if not url:
        return None

    # Decide on file extension and name
    filename = f"{entry.get('id', 'unknown')}_{url.split('/')[-1]}"  # Add ID to avoid collisions

    # Example subdirectory logic:
    subdir = "videos" if entry.get("is_video", False) else "images"
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    local_path = os.path.join(out_dir, subdir, filename)

    # Skip if already downloaded
    if os.path.exists(local_path):
        return local_path

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except Exception:
        return None


# --------------- #
# STAGE 4: COLUMN EXTRACTION
# --------------- #


def extract_columns(
    entry: Dict[str, Any], local_media_path: Optional[str]
) -> Dict[str, Any]:
    """Extract required columns from the entry."""
    return {
        "id": entry.get("id"),
        "title": entry.get("title", "").strip(),
        "selftext": entry.get("selftext", "").strip(),
        "subreddit_name": entry.get("subreddit", ""),
        "subreddit_id": entry.get("subreddit_id", ""),
        "num_comments": entry.get("num_comments", 0),
        "score": entry.get("score", 0),
        "external_url": entry.get("url", ""),
        "author_name": entry.get("author", "[deleted]"),
        "author_id": entry.get("author_id", None),
        "local_media_path": local_media_path,
    }


def process_entry(
    entry: Dict[str, Any], split: str, min_words: int, max_words: int, download_dir: str
) -> Optional[Dict[str, Any]]:
    """Process a single entry: filter, verify media, download, and extract columns."""
    if not filter_entry(entry, split, min_words, max_words):
        return None

    if split == "multimodal":
        if not verify_media_availability(entry):
            return None
        local_path = download_media(entry, out_dir=download_dir)
        if local_path is None:
            return None
    else:
        local_path = None

    return extract_columns(entry, local_path)


# --------------- #
# MAIN PROCESSING FUNCTION
# --------------- #


def process_reddit_data_dask_percentage(
    input_dir: str,
    output_jsonl: str,
    split: str,
    sample_percentage: float,
    min_words: int,
    max_words: int,
    download_dir: str = "sampled_data",
    seed: int = 42,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
    max_samples: Optional[int] = None,
):
    """
    Process Reddit data using Dask, sampling a percentage of the dataset i.i.d.

    Args:
        input_dir: Directory containing RS_*.jsonl files
        output_jsonl: Path to output JSONL file
        split: Either 'text' or 'multimodal'
        sample_percentage: Percentage of data to sample (0-100)
        min_words, max_words: Word count filters
        download_dir: Directory for downloaded media (if multimodal)
        seed: Random seed for reproducibility
        n_workers: Number of Dask workers
        memory_per_worker: Memory per worker
        use_slurm: Whether to use a SLURM cluster
        max_samples: Optional cap on the total number of samples
    """
    os.makedirs(
        os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else ".",
        exist_ok=True,
    )
    os.makedirs(download_dir, exist_ok=True)

    # Step 1: Set up Dask client
    if use_slurm:
        # Set up SLURM cluster
        logger.info(f"Setting up SLURM cluster with {n_workers} workers")
        cluster = SLURMCluster(
            cores=1,  # 1 core per worker
            processes=1,  # 1 process per worker
            memory=memory_per_worker,
            walltime="1-00:00:00",  # 1 day
            job_extra=["--cpus-per-task=1"],
        )
        cluster.scale(n_workers)  # Scale to desired number of workers
        client = Client(cluster)
    else:
        # Set up local cluster
        logger.info(f"Setting up local Dask cluster with {n_workers} workers")
        client = Client(
            n_workers=n_workers, threads_per_worker=1, memory_limit=memory_per_worker
        )

    logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    # Step 2: Gather all JSONL files
    logger.info(f"Finding JSONL files in {input_dir}")
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_*.jsonl files found in directory: {input_dir}")
    logger.info(f"Found {len(files)} JSONL files")

    # Step 3: Create a Dask bag from the files and process each file
    # We'll map the sampling function to each file
    bag = db.from_sequence(files)

    # Map the sampling function to each file
    processed_bag = bag.map(
        lambda file_path: sample_file_with_percentage(
            file_path=file_path,
            sample_percentage=sample_percentage,
            split=split,
            min_words=min_words,
            max_words=max_words,
            download_dir=download_dir,
            seed=seed,
        )
    ).flatten()

    # If max_samples is specified, we need to limit the number of samples
    if max_samples is not None:
        # First get count of sampled entries
        with ProgressBar():
            total_sampled = processed_bag.count().compute()

        logger.info(f"Total sampled entries before cap: {total_sampled}")

        if total_sampled > max_samples:
            # Take a random subset of the sampled entries
            # We repartition to ensure randomness across all entries
            processed_bag = processed_bag.random_sample(
                max_samples / total_sampled, random_state=seed
            )

    # Compute the results
    logger.info("Processing data with Dask...")
    with ProgressBar():
        results = processed_bag.compute()

    # Flatten results if needed
    if isinstance(results, list) and results and isinstance(results[0], list):
        results = [item for sublist in results for item in sublist]

    # Write to JSONL
    logger.info(f"Writing {len(results)} entries to {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for entry in tqdm(results, desc="Writing to output"):
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Sampling complete! ✓")
    logger.info(f"Output file: {output_jsonl}")
    if split == "multimodal":
        logger.info(f"Media downloaded to: {download_dir}")

    # Clean up
    client.close()
    if use_slurm:
        cluster.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sample a percentage of Reddit data using Dask"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing RS_*.jsonl files",
    )
    parser.add_argument(
        "--output_jsonl", type=str, required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["text", "multimodal"],
        default="text",
        help="Whether to sample text-only or multimodal content",
    )
    parser.add_argument(
        "--sample_percentage",
        type=float,
        default=1.0,
        help="Percentage of data to sample (0-100)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on total number of samples",
    )
    parser.add_argument("--min_words", type=int, default=5, help="Minimum word count")
    parser.add_argument("--max_words", type=int, default=200, help="Maximum word count")
    parser.add_argument(
        "--download_dir",
        type=str,
        default="sampled_data",
        help="Directory for downloaded media",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_workers", type=int, default=16, help="Number of Dask workers"
    )
    parser.add_argument(
        "--memory_per_worker",
        type=str,
        default="4GB",
        help="Memory per worker (e.g., '4GB')",
    )
    parser.add_argument(
        "--use_slurm",
        action="store_true",
        help="Use SLURM cluster instead of local cluster",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger("reddit_sampler").setLevel(logging.DEBUG)

    process_reddit_data_dask_percentage(
        input_dir=args.input_dir,
        output_jsonl=args.output_jsonl,
        split=args.split,
        sample_percentage=args.sample_percentage,
        min_words=args.min_words,
        max_words=args.max_words,
        download_dir=args.download_dir,
        seed=args.seed,
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
