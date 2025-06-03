"""Shared utility helpers for working with the raw Reddit Pushshift dump.

These helpers used to live in sample.py but are now extracted so they can be
imported by multiple scripts without circular dependencies.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Generator, List, Optional

import requests

__all__ = [
    "get_all_jsonl_files",
    "count_lines",
    "stream_jsonl",
    "stream_jsonl_with_probability",
    "sample_file_with_percentage",
    "filter_entry",
    "verify_media_availability",
    "download_media",
    "extract_columns",
    "process_entry",
]

# --------------- #
# FILE HANDLING & SAMPLING
# --------------- #

def get_all_jsonl_files(path: str) -> List[str]:
    """Return a list of Reddit submission files starting with ``RS_``.

    * If *path* is a **file**, we return a single-element list with that file – this
      makes it possible to point the scripts directly to a specific month such
      as ``/data/RS_2010-01`` (no extension) or ``RS_2010-01.json``.
    * If *path* is a **directory**, we recursively collect all files whose
      basename starts with ``RS_`` regardless of the file extension (``.jsonl``,
      ``.json``, or none).
    """
    if os.path.isfile(path):
        # User pointed to a single file – return it as-is
        return [path]

    jsonl_files: List[str] = []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            if fname.startswith("RS_"):
                jsonl_files.append(os.path.join(root, fname))
    return jsonl_files


def count_lines(file_path: str) -> int:
    """Return the number of lines in *file_path* (memory-efficient)."""
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


# Streaming helpers -----------------------------------------------------------

def stream_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Yield each JSON line as a Python dict."""
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
    file_path: str,
    sample_probability: float,
    seed: int,
) -> Generator[Dict[str, Any], None, None]:
    """I.i.d. sample from *file_path* while streaming.

    Each entry is yielded with probability *sample_probability*.
    """
    rng = random.Random(seed)
    for item in stream_jsonl(file_path):
        if rng.random() < sample_probability:
            yield item


# Sampling per file -----------------------------------------------------------

def sample_file_with_percentage(
    file_path: str,
    sample_percentage: float,
    split: str,
    min_words: int,
    max_words: int,
    download_dir: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample *sample_percentage*% from a single file and return processed rows."""
    sample_prob = sample_percentage / 100.0
    file_seed = seed + hash(file_path) % 10000  # avoid cross-file correlation

    results: List[Dict[str, Any]] = []
    for entry in stream_jsonl_with_probability(file_path, sample_prob, file_seed):
        processed = process_entry(entry, split, min_words, max_words, download_dir)
        if processed:
            results.append(processed)
    return results


# --------------- #
# FILTERING & MEDIA HANDLING
# --------------- #

def filter_entry(
    entry: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
) -> bool:
    """Apply the standard project filters to a single Pushshift Reddit record."""
    # Adult flag --------------------------------------------------------------
    if entry.get("over_18", False):
        return False

    # Promoted content --------------------------------------------------------
    if entry.get("promoted") is True:
        return False
    if entry.get("whitelist_status") == "promo_specified":
        return False

    # Title & body presence ---------------------------------------------------
    selftext = entry.get("selftext", "")
    if not selftext.strip():
        return False

    # Word-count filter -------------------------------------------------------
    n_words = len(selftext.strip().split())
    if n_words < min_words or n_words > max_words:
        return False

    # Media presence check ----------------------------------------------------
    has_video = bool(entry.get("is_video", False))
    url = entry.get("url", "")
    has_image = any(url.lower().endswith(ext) for ext in (".jpg", ".png", ".jpeg", ".gif"))

    if split == "text":
        # text split must NOT contain media
        if has_video or has_image:
            return False
    elif split == "multimodal":
        # multimodal split must HAVE media
        if not (has_video or has_image):
            return False

    return True


def verify_media_availability(entry: Dict[str, Any]) -> bool:
    """Check if *entry['url']* is reachable via a HEAD request."""
    url = entry.get("url", "")
    if not url:
        return True  # no media linked – treat as available

    try:
        resp = requests.head(url, timeout=5)
        return resp.status_code in {200, 302}
    except requests.RequestException:
        return False


# Download helper -------------------------------------------------------------

def download_media(entry: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Download media referenced in *entry* and return the local path (or None)."""
    url = entry.get("url", "")
    if not url:
        return None

    filename = f"{entry.get('id', 'unknown')}_{url.split('/')[-1]}"
    subdir = "videos" if entry.get("is_video", False) else "images"
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
    local_path = os.path.join(out_dir, subdir, filename)

    # Skip download if already present ---------------------------------------
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


# Column extraction -----------------------------------------------------------

def extract_columns(entry: Dict[str, Any], local_media_path: Optional[str]) -> Dict[str, Any]:
    """Return a subset of original Pushshift fields without renaming them.

    The function keeps the exact key names used in the original Pushshift
    dump so that downstream processing (and potential database imports)
    can rely on a stable, idiomatic schema. Additional, project-specific
    keys (such as *local_media_path*) are preserved with their custom
    names because they do not collide with Pushshift fields.
    """

    return {
        # Original Pushshift field names -------------------------------
        "id": entry.get("id"),
        "title": entry.get("title", "").strip(),
        "selftext": entry.get("selftext", "").strip(),
        "subreddit": entry.get("subreddit", ""),
        "subreddit_id": entry.get("subreddit_id", ""),
        "num_comments": entry.get("num_comments", 0),
        "score": entry.get("score", 0),
        "url": entry.get("url", ""),
        "created_utc": entry.get("created_utc"),
        "author": entry.get("author"),
        "author_id": entry.get("author_id"),
        # Project-specific augmentation --------------------------------
        "local_media_path": local_media_path,
    }


# Entry-level pipeline --------------------------------------------------------

def process_entry(
    entry: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
    download_dir: str,
) -> Optional[Dict[str, Any]]:
    """Apply filtering (+ optional media download) and extract columns."""
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