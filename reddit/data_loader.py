"""
Reddit-specific data loading and processing functions.
Handles large-scale parallel processing of Reddit JSONL files.
"""
import json
import logging
from typing import Dict, Any, List
import dask.bag as db
from dask.diagnostics import ProgressBar

from helpers import get_all_jsonl_files, filter_entry, extract_columns
from self_identification import SelfIdentificationDetector, detect_self_identification_with_resolved_age
from core.data_processing import apply_linguistic_features

logger = logging.getLogger("reddit.data_loader")


def process_reddit_file_for_self_identification(
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

            # Apply the same filtering criteria as predefined for the project
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


def process_reddit_file_for_user_posts(
    file_path: str, 
    user_birthyears: Dict[str, int], 
    split: str, 
    min_words: int, 
    max_words: int,
    include_features: bool = True
) -> List[Dict[str, Any]]:
    """Process Reddit file to collect posts from specific users."""
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

            # Ensure we use the same identifier logic as in identify_self_users.py
            author_fullname = entry.get("author_fullname")
            author_name = entry.get("author")

            # Skip automated accounts
            if author_name in ("AutoModerator", "Bot"):
                continue

            # Retrieve inferred birth year if this post was written by a self-identified user
            birth_year = None

            # First try to match using author_fullname (preferred method)
            if author_fullname and author_fullname in user_birthyears:
                birth_year = user_birthyears[author_fullname]
            # Fallback to author name if author_fullname is not available or not found
            elif author_name and author_name in user_birthyears:
                birth_year = user_birthyears[author_name]

            if birth_year is None:
                continue

            if not filter_entry(entry, split=split, min_words=min_words, max_words=max_words):
                continue

            post_data = extract_columns(entry, None)
            
            # Replace flat author string with detailed object
            post_data["author"] = {
                "name": author_name,
                "age": None,
            }
            
            # Compute dynamic age if we have a birth year and timestamp
            created_utc = entry.get("created_utc")
            if created_utc and birth_year:
                try:
                    from datetime import datetime, timezone
                    if isinstance(created_utc, str):
                        created_utc = float(created_utc)
                    post_year = datetime.fromtimestamp(created_utc, timezone.utc).year
                    age_val = max(0, post_year - birth_year)
                    post_data["author"]["age"] = age_val
                except (ValueError, TypeError, OSError):
                    pass

            # Compute linguistic features if requested
            if include_features:
                features = apply_linguistic_features(post_data.get("selftext", ""))
                post_data.update(features)

            results.append(post_data)

    return results


def load_reddit_files_for_self_identification(
    input_dir: str,
    detector: SelfIdentificationDetector,
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    client=None
) -> List[Dict[str, Any]]:
    """Load and process Reddit files for self-identification using Dask."""
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_* files found inside {input_dir}")
    
    logger.info(f"Found {len(files)} JSONL files to scan for self-identification.")

    bag = db.from_sequence(files, npartitions=len(files))
    processed_bag = bag.map(
        lambda fp: process_reddit_file_for_self_identification(
            fp, detector=detector, split=split, min_words=min_words, max_words=max_words
        )
    ).flatten()

    with ProgressBar():
        results: List[Dict[str, Any]] = processed_bag.compute()

    return results


def load_reddit_files_for_user_posts(
    input_dir: str,
    user_birthyears: Dict[str, int],
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    include_features: bool = True,
    client=None
) -> List[Dict[str, Any]]:
    """Load and process Reddit files for user posts using Dask."""
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_* files found inside {input_dir}")
        
    logger.info(f"Scanning {len(files)} JSONL files for posts written by target users.")

    bag = db.from_sequence(files, npartitions=len(files))
    processed_bag = bag.map(
        lambda fp: process_reddit_file_for_user_posts(
            fp, user_birthyears=user_birthyears, split=split, 
            min_words=min_words, max_words=max_words, include_features=include_features
        )
    ).flatten()

    with ProgressBar():
        results = processed_bag.compute()

    return results