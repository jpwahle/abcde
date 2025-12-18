"""JSONL file handling and filtering utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from abcde.utils.text import clean_text_newlines


def get_all_jsonl_files(path: str) -> List[str]:
    """Get all JSONL files from a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for root, _, fnames in os.walk(path):
        for nm in fnames:
            if nm.startswith("RS_"):
                files.append(os.path.join(root, nm))
    # Sort by name to ensure consistent order
    files.sort()
    return files


def filter_entry(
    entry: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
) -> bool:
    """Filter a Reddit entry based on various criteria."""
    if entry.get("over_18", False):
        return False
    if (
        entry.get("promoted") is True
        or entry.get("whitelist_status") == "promo_specified"
    ):
        return False
    text = entry.get("selftext", "")
    if not text.strip():
        return False
    n = len(text.strip().split())
    if n < min_words or n > max_words:
        return False
    has_vid = bool(entry.get("is_video", False))
    url = entry.get("url", "") or ""
    has_img = any(
        url.lower().endswith(ext) for ext in (".jpg", ".png", ".jpeg", ".gif")
    )
    if split == "text" and (has_vid or has_img):
        return False
    if split == "multimodal" and not (has_vid or has_img):
        return False
    return True


def extract_columns(
    entry: Dict[str, Any], local_media_path: Optional[str]
) -> Dict[str, Any]:
    """Extract standard columns from a Reddit entry."""
    return {
        "id": entry.get("id"),
        "subreddit": entry.get("subreddit"),
        "title": entry.get("title", ""),
        "selftext": clean_text_newlines(entry.get("selftext", "")),
        "created_utc": entry.get("created_utc"),
        "score": entry.get("score"),
        "num_comments": entry.get("num_comments"),
        "author": entry.get("author"),
        "permalink": entry.get("permalink"),
        "url": entry.get("url"),
        "media_path": local_media_path,
    }
