#!/usr/bin/env python3
import os
import json
import random
import argparse
import requests

from typing import Dict, Any, Generator, List, Optional
from tqdm import tqdm


# --------------- #
# STAGE 1: LOADING & SAMPLING
# --------------- #

def get_all_jsonl_files(directory: str) -> List[str]:
    """
    Retrieve all .jsonl files from a given directory (non-recursive).
    Adjust if you want recursion or a custom pattern.
    """
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
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
    files: List[str],
    n_samples: int,
    seed: int = 42
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
        for entry in tqdm(stream_jsonl(fp), desc=f"Lines in {os.path.basename(fp)}", leave=False):
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
    has_image = any(url.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".gif"])

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

def extract_columns(entry: Dict[str, Any], local_media_path: Optional[str]) -> Dict[str, Any]:
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


# --------------- #
# STAGE 5: MAIN PIPELINE
# --------------- #

def process_reddit_data(
    input_dir: str,
    output_jsonl: str,
    split: str,
    n_samples: int,
    min_words: int,
    max_words: int,
    download_dir: str = "sampled_data",
    seed: int = 42
):
    """
    Main pipeline:
      1) Gather all monthly JSONL files
      2) Reservoir sample N i.i.d. entries
      3) Filter each entry
      4) (Optional) verify & download media
      5) Extract columns
      6) Write to a JSONL output
    """
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_*.jsonl files found in directory: {input_dir}")

    # Stage 1: reservoir sampling
    sampled_entries = list(reservoir_sample(files, n_samples, seed=seed))

    # Prepare output directory
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    count_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        # Wrap the iteration of sampled_entries with tqdm so we can see progress
        for entry in tqdm(sampled_entries, desc="Filtering & writing to disk", total=n_samples):
            if not filter_entry(entry, split, min_words, max_words):
                continue

            # If we are in the multimodal split, verify the media is accessible
            if split == "multimodal":
                if not verify_media_availability(entry):
                    continue

            local_path = None
            if split == "multimodal":
                # Attempt download
                local_path = download_media(entry, out_dir=download_dir)
                if local_path is None:
                    # If we can't download the file, skip it
                    continue

            # Extract columns
            data_to_write = extract_columns(entry, local_path)
            # Write to JSONL
            out_f.write(json.dumps(data_to_write, ensure_ascii=False) + "\n")
            count_written += 1

    print(f"Finished writing {count_written} entries to {output_jsonl}")


# --------------- #
# STAGE 6: ARGPARSE
# --------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing monthly RS_YYYY-MM JSONL files.")
    parser.add_argument("--output_jsonl", type=str, required=True, default="outputs/sample.jsonl",
                        help="Where to write the filtered JSONL sample.")
    parser.add_argument("--split", type=str, choices=["text", "multimodal"], default="text",
                        help="Which dataset split we are creating: text-only or multimodal.")
    parser.add_argument("--n_samples", type=int, default=3_000_000,
                        help="Number of i.i.d. examples to reservoir sample.")
    parser.add_argument("--min_words", type=int, default=5,
                        help="Minimum word count in selftext.")
    parser.add_argument("--max_words", type=int, default=200,
                        help="Maximum word count in selftext.")
    parser.add_argument("--download_dir", type=str, default="sampled_data",
                        help="Where to download images/videos for the multimodal split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    process_reddit_data(
        input_dir=args.input_dir,
        output_jsonl=args.output_jsonl,
        split=args.split,
        n_samples=args.n_samples,
        min_words=args.min_words,
        max_words=args.max_words,
        download_dir=args.download_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
