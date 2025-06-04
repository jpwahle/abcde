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
    output_file: str = None,
    max_entries_in_memory: int = 10000
) -> List[Dict[str, Any]]:
    """Stream a single JSONL Reddit file and return entries that contain self-identification.
    
    If output_file is provided, writes results directly to disk in batches to control memory usage.
    """
    import gc
    import pandas as pd
    import os
    from pathlib import Path
    
    results: List[Dict[str, Any]] = []
    
    # If we have an output file, prepare for direct writing
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        write_header = True
    
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
            
            # Write to disk if we have enough results to avoid memory buildup
            if output_file and len(results) >= max_entries_in_memory:
                # Convert to DataFrame and append to file
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_file, mode='a', index=False, header=write_header)
                    write_header = False  # Only write header once
                    
                # Clear results from memory
                results = []
                del results_df
                gc.collect()

    # Write any remaining results
    if output_file and results:
        results_df = pd.DataFrame(results)
        write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        results_df.to_csv(output_file, mode='a', index=False, header=write_header)
        del results_df
        gc.collect()
        
        # Return empty list since results were written to disk
        return []
    
    return results


def process_reddit_file_for_user_posts(
    file_path: str, 
    user_birthyears: Dict[str, int], 
    split: str, 
    min_words: int, 
    max_words: int,
    include_features: bool = True,
    output_file: str = None,
    max_entries_in_memory: int = 10000
) -> List[Dict[str, Any]]:
    """Process Reddit file to collect posts from specific users.
    
    If output_file is provided, writes results directly to disk in batches to control memory usage.
    """
    import gc
    import pandas as pd
    import os
    from pathlib import Path
    
    results: List[Dict[str, Any]] = []
    
    # If we have an output file, prepare for direct writing
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        write_header = True

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
            
            # Write to disk if we have enough results to avoid memory buildup
            if output_file and len(results) >= max_entries_in_memory:
                # Convert to DataFrame and append to file
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_file, mode='a', index=False, header=write_header)
                    write_header = False  # Only write header once
                    
                # Clear results from memory
                results = []
                del results_df
                gc.collect()

    # Write any remaining results
    if output_file and results:
        results_df = pd.DataFrame(results)
        write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        results_df.to_csv(output_file, mode='a', index=False, header=write_header)
        del results_df
        gc.collect()
        
        # Return empty list since results were written to disk
        return []

    return results


def process_reddit_file_for_self_identification_worker(
    file_path: str,
    split: str,
    min_words: int,
    max_words: int,
    output_dir: str,
    max_entries_in_memory: int = 10000
) -> str:
    """Process a single Reddit JSONL file for self-identification detection.
    
    Creates detector locally to avoid large object transfer to workers.
    Writes results directly to disk chunks.
    Returns the path to the written chunk file.
    """
    import os
    from pathlib import Path
    
    # Create detector locally on worker to avoid serialization overhead
    from self_identification import SelfIdentificationDetector
    detector = SelfIdentificationDetector()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique chunk filename
    worker_id = os.getpid()  # Use process ID as worker identifier
    file_basename = Path(file_path).stem
    chunk_filename = f"chunk_worker{worker_id}_{file_basename}.csv"
    chunk_path = os.path.join(output_dir, chunk_filename)
    
    # Process file and write directly to chunk
    process_reddit_file_for_self_identification(
        file_path=file_path,
        detector=detector,
        split=split,
        min_words=min_words,
        max_words=max_words,
        output_file=chunk_path,
        max_entries_in_memory=max_entries_in_memory
    )
    
    # Return the chunk file path (create empty file if no results)
    if not os.path.exists(chunk_path):
        Path(chunk_path).touch()
    
    return chunk_path


def load_reddit_files_for_self_identification(
    input_dir: str,
    detector: SelfIdentificationDetector,
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    client=None,
    output_dir: str = None,
    max_entries_in_memory: int = 10000
) -> List[Dict[str, Any]]:
    """Load and process Reddit files for self-identification using Dask.
    
    If output_dir is provided, workers write chunks directly to disk and results are
    concatenated from disk. Otherwise, uses in-memory processing.
    """
    import os
    import tempfile
    
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_* files found inside {input_dir}")
    
    logger.info(f"Found {len(files)} JSONL files to scan for self-identification.")

    # Create temporary output directory if not provided
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="reddit_chunks_")
        output_dir = temp_dir
        cleanup_temp = True
        use_disk_chunks = True
    else:
        cleanup_temp = False
        use_disk_chunks = True

    if use_disk_chunks:
        logger.info(f"Writing worker chunks to: {output_dir}")

        bag = db.from_sequence(files, npartitions=len(files))
        processed_bag = bag.map(
            lambda fp: process_reddit_file_for_self_identification_worker(
                fp, 
                split=split, 
                min_words=min_words, 
                max_words=max_words,
                output_dir=output_dir,
                max_entries_in_memory=max_entries_in_memory
            )
        )

        with ProgressBar():
            chunk_paths = processed_bag.compute()

        # Concatenate results from disk chunks
        logger.info(f"Concatenating {len(chunk_paths)} chunk files from disk...")
        all_results = []
        
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                try:
                    import pandas as pd
                    chunk_df = pd.read_csv(chunk_path)
                    if not chunk_df.empty:
                        chunk_results = chunk_df.to_dict('records')
                        all_results.extend(chunk_results)
                        del chunk_df, chunk_results
                except Exception as e:
                    logger.warning(f"Failed to read chunk {chunk_path}: {e}")
        
        # Clean up temporary files
        if cleanup_temp:
            import shutil
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {output_dir}: {e}")
        
        logger.info(f"Total results collected: {len(all_results)}")
        return all_results
    else:
        # Original in-memory processing
        bag = db.from_sequence(files, npartitions=len(files))
        processed_bag = bag.map(
            lambda fp: process_reddit_file_for_self_identification(
                fp, detector=detector, split=split, min_words=min_words, max_words=max_words
            )
        ).flatten()

        with ProgressBar():
            results: List[Dict[str, Any]] = processed_bag.compute()

        return results


def process_reddit_file_for_user_posts_worker(
    file_path: str,
    user_birthyears: Dict[str, int],
    split: str,
    min_words: int,
    max_words: int,
    include_features: bool,
    output_dir: str,
    max_entries_in_memory: int = 10000
) -> str:
    """Process a single Reddit JSONL file for user post collection.
    
    Writes results directly to disk chunks.
    Returns the path to the written chunk file.
    """
    import os
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique chunk filename
    worker_id = os.getpid()  # Use process ID as worker identifier
    file_basename = Path(file_path).stem
    chunk_filename = f"chunk_worker{worker_id}_{file_basename}.csv"
    chunk_path = os.path.join(output_dir, chunk_filename)
    
    # Process file and write directly to chunk
    process_reddit_file_for_user_posts(
        file_path=file_path,
        user_birthyears=user_birthyears,
        split=split,
        min_words=min_words,
        max_words=max_words,
        include_features=include_features,
        output_file=chunk_path,
        max_entries_in_memory=max_entries_in_memory
    )
    
    # Return the chunk file path (create empty file if no results)
    if not os.path.exists(chunk_path):
        Path(chunk_path).touch()
    
    return chunk_path


def load_reddit_files_for_user_posts(
    input_dir: str,
    user_birthyears: Dict[str, int],
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    include_features: bool = True,
    client=None,
    output_dir: str = None,
    max_entries_in_memory: int = 10000
) -> List[Dict[str, Any]]:
    """Load and process Reddit files for user posts using Dask.
    
    If output_dir is provided, workers write chunks directly to disk and results are
    concatenated from disk. Otherwise, uses in-memory processing.
    """
    import os
    import tempfile
    
    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_* files found inside {input_dir}")
        
    logger.info(f"Scanning {len(files)} JSONL files for posts written by target users.")

    # Create temporary output directory if not provided
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="reddit_user_chunks_")
        output_dir = temp_dir
        cleanup_temp = True
        use_disk_chunks = True
    else:
        cleanup_temp = False
        use_disk_chunks = True

    if use_disk_chunks:
        logger.info(f"Writing worker chunks to: {output_dir}")

        bag = db.from_sequence(files, npartitions=len(files))
        processed_bag = bag.map(
            lambda fp: process_reddit_file_for_user_posts_worker(
                fp, 
                user_birthyears=user_birthyears,
                split=split, 
                min_words=min_words, 
                max_words=max_words, 
                include_features=include_features,
                output_dir=output_dir,
                max_entries_in_memory=max_entries_in_memory
            )
        )

        with ProgressBar():
            chunk_paths = processed_bag.compute()

        # Concatenate results from disk chunks
        logger.info(f"Concatenating {len(chunk_paths)} chunk files from disk...")
        all_results = []
        
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                try:
                    import pandas as pd
                    chunk_df = pd.read_csv(chunk_path)
                    if not chunk_df.empty:
                        chunk_results = chunk_df.to_dict('records')
                        all_results.extend(chunk_results)
                        del chunk_df, chunk_results
                except Exception as e:
                    logger.warning(f"Failed to read chunk {chunk_path}: {e}")
        
        # Clean up temporary files
        if cleanup_temp:
            import shutil
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {output_dir}: {e}")
        
        logger.info(f"Total results collected: {len(all_results)}")
        return all_results
    else:
        # Original in-memory processing
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