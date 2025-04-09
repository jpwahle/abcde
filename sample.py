#!/usr/bin/env python3
import os
import json
import random
import argparse
import requests
from typing import Dict, Any, Generator, List, Optional
from tqdm import tqdm
import dask.bag as db
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


# --------------- #
# STAGE 1: LOADING & SAMPLING
# --------------- #


def get_all_jsonl_files(directory: str) -> List[str]:
    """
    Retrieve all .jsonl files from a given directory (non-recursive).
    Adjust if you want recursion or a custom pattern.
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("RS_") and not os.path.isdir(os.path.join(directory, f))
    ]


def stream_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Given a .jsonl file path, yield each line as a Python dict.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def reservoir_sample(
    files: List[str], n_samples: int, seed: int = 42
) -> Generator[Dict[str, Any], None, None]:
    """
    Perform reservoir sampling across multiple .jsonl files.
    This yields exactly n_samples items, i.i.d. from all lines combined.

    If the total number of lines across all files is smaller than n_samples,
    you will simply yield all of them (but that’s unlikely in a huge dataset).
    """
    random.seed(seed)
    reservoir = []
    total_count = 0

    # Use tqdm to show progress across all files
    for fp in tqdm(files, desc="Reading files for reservoir sampling"):
        # Because we don't know how many lines are in each file,
        # we can wrap the line iteration in another tqdm, but we’ll only see
        # indefinite progress. Alternatively, you can omit it or do a first pass
        # to count lines. We'll show it just as an example:
        for entry in tqdm(
            stream_jsonl(fp), desc=f"Lines in {os.path.basename(fp)}", leave=False
        ):
            total_count += 1
            if len(reservoir) < n_samples:
                reservoir.append(entry)
            else:
                # Replace items with gradually decreasing probability
                r = random.randint(0, total_count - 1)
                if r < n_samples:
                    reservoir[r] = entry

    # Shuffle once more to ensure randomness within the reservoir
    random.shuffle(reservoir)

    # Convert the final reservoir list into a generator
    for item in reservoir:
        yield item


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

    - If `split` == 'text', ensure no images/videos.
    - If `split` == 'multimodal', ensure there's an image or a video link.
    - Remove adult content (over_18=True).
    - Remove entries marked as promoted or ads.
    - Remove entries with only a title but empty selftext.
    - Remove entries outside the word count range.
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
    title = entry.get("title", "")
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
    """
    Attempt to do a HEAD request for the image/video and see if it’s accessible.
    If not accessible, return False. Otherwise True.
    """
    url = entry.get("url", "")
    if not url:
        # If no URL, it might be okay for text-only, but for multimodal,
        # we presumably want a URL. We'll handle that logic in filter_entry.
        return True

    try:
        resp = requests.head(url, timeout=5)
        # Accept 200 or 302, etc.  Adjust as needed.
        return resp.status_code in (200, 302)
    except requests.RequestException:
        return False


# --------------- #
# STAGE 3: DOWNLOADING (OPTIONAL for multimodal)
# --------------- #


def download_media(entry: Dict[str, Any], out_dir: str) -> Optional[str]:
    """
    Download the image or video to `out_dir`, return local file path,
    or None if download fails or if there's no media.

    This is a minimal example. In practice, you might differentiate images vs. videos
    and store them in different subdirs. Also verify MIME type, handle chunked downloads, etc.
    """
    url = entry.get("url", "")
    if not url:
        return None

    # Decide on file extension
    filename = url.split("/")[-1]  # naive approach
    # Example subdirectory logic:
    subdir = "videos" if entry.get("is_video", False) else "images"
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    local_path = os.path.join(out_dir, subdir, filename)

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
    """
    From the raw Reddit entry, pick out the columns we want, plus the local media path if any.
    """
    return {
        "id": entry.get("id"),
        "title": entry.get("title", "").strip(),
        "selftext": entry.get("selftext", "").strip(),
        "subreddit_name": entry.get("subreddit", ""),
        "subreddit_id": entry.get("subreddit_id", ""),
        "num_comments": entry.get("num_comments", 0),
        "score": entry.get("score", 0),
        "external_url": entry.get("url", ""),  # or domain, etc.
        "author_name": entry.get("author", "[deleted]"),
        "author_id": entry.get("author_id", None),
        # Add local media path and direct link (if any):
        "local_media_path": local_media_path,
    }


def reservoir_sample_chunk(
    entries: List[Dict[str, Any]], n_samples: int, seed: int
) -> List[Dict[str, Any]]:
    """Perform reservoir sampling on a chunk of entries."""
    random.seed(seed)
    reservoir = []
    total_count = 0
    for entry in entries:
        total_count += 1
        if len(reservoir) < n_samples:
            reservoir.append(entry)
        else:
            r = random.randint(0, total_count - 1)
            if r < n_samples:
                reservoir[r] = entry
    return reservoir


def process_entry(
    entry: Dict[str, Any], split: str, min_words: int, max_words: int, download_dir: str
) -> Optional[Dict[str, Any]]:
    """Process a single entry: filter, verify media (if needed), download (if needed), and extract columns."""
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


def process_reddit_data_dask(
    input_dir: str,
    output_jsonl: str,
    split: str,
    n_samples: int,
    min_words: int,
    max_words: int,
    download_dir: str = "sampled_data",
    seed: int = 42,
    n_workers: int = 64,
    memory_per_worker: str = "16GB",
):
    # Set up SLURM cluster
    cluster = SLURMCluster(
        cores=1,  # 1 core per worker
        processes=1,  # 1 process per worker
        memory=memory_per_worker,
        walltime="1-06:00:00",
        job_extra=["--cpus-per-task=1"],
    )
    cluster.scale(n_workers)  # Scale to desired number of workers
    client = Client(cluster)  # Connect to the cluster
    print(f"Dask dashboard available at: {client.dashboard_link}")

    # Step 1: Gather all JSONL files
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_*.jsonl files found in directory: {input_dir}")

    # Step 2: Create a Dask Bag from the files
    bag = db.from_sequence(files).map(lambda fp: list(stream_jsonl(fp))).flatten()

    # Step 3: Filter and process entries in parallel
    processed_bag = bag.map(
        lambda entry: process_entry(entry, split, min_words, max_words, download_dir)
    ).filter(lambda x: x is not None)

    # Step 4: Perform reservoir sampling on the processed entries
    # Since reservoir sampling is inherently sequential, we’ll approximate it by sampling chunks
    chunk_size = n_samples // n_workers + 1
    sampled_bag = processed_bag.repartition(n_workers).map_partitions(
        lambda part: reservoir_sample_chunk(list(part), chunk_size, seed)
    )
    sampled_entries = sampled_bag.flatten().take(
        n_samples, npartitions=-1
    )  # Take final n_samples

    # Step 5: Write to JSONL
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for entry in tqdm(sampled_entries, desc="Writing to disk", total=n_samples):
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Finished writing {len(sampled_entries)} entries to {output_jsonl}")
    client.close()
    cluster.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--split", type=str, choices=["text", "multimodal"], default="text"
    )
    parser.add_argument("--n_samples", type=int, default=3_000_000)
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=200)
    parser.add_argument("--download_dir", type=str, default="sampled_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_workers", type=int, default=32, help="Number of Dask workers"
    )
    parser.add_argument(
        "--memory_per_worker", type=str, default="4GB", help="Memory per worker"
    )
    args = parser.parse_args()

    process_reddit_data_dask(
        input_dir=args.input_dir,
        output_jsonl=args.output_jsonl,
        split=args.split,
        n_samples=args.n_samples,
        min_words=args.min_words,
        max_words=args.max_words,
        download_dir=args.download_dir,
        seed=args.seed,
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
    )


if __name__ == "__main__":
    main()
