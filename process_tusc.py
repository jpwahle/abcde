#!/usr/bin/env python3
"""
Process TUSC pipeline: detect self-identified users and collect posts with linguistic features.
"""
import argparse
import os
import signal
import psutil
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from helpers import (
    SelfIdentificationDetector,
    apply_linguistic_features,
    detect_self_identification_in_tusc_entry,
    detect_self_identification_in_tusc_entry_with_mappings,
    ensure_output_directory,
    write_results_to_csv,
    append_results_to_csv,
    print_banner,
    aggregate_user_demographics,
)


# Timeout wrapper for debugging pathological cases
class Timeout(Exception):
    pass

def timeout(sec):
    def deco(fn):
        def _wrap(*a, **kw):
            def handler(signum, frame):
                raise Timeout(f"Function {fn.__name__} timed out after {sec} seconds")
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(sec)
            try:
                return fn(*a, **kw)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        return _wrap
    return deco

# Create safe wrapped versions of potentially problematic functions
safe_detect_self_identification = timeout(5)(detect_self_identification_in_tusc_entry_with_mappings)
safe_apply_linguistic_features = timeout(5)(apply_linguistic_features)

def log_memory_usage(batch_num=None):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    prefix = f"Batch {batch_num}: " if batch_num is not None else ""
    print(f"{prefix}Memory: {memory.percent:.1f}% used ({memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB), "
          f"Swap: {swap.percent:.1f}% used ({swap.used/(1024**3):.1f}GB / {swap.total/(1024**3):.1f}GB)")


def determine_split(input_file: str) -> str:
    fname = os.path.basename(input_file).lower()
    return "city" if "city" in fname else "country"


def load_self_identified_users(csv_path: str) -> set:
    """Load user IDs from existing self-identified users CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Self-identified users file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep="\t", dtype=str)
    user_ids = set()

    # Handle different possible column names
    for col in ["Author", "UserID", "userID"]:
        if col in df.columns:
            user_ids.update(df[col].dropna().astype(str).tolist())
    # Normalize IDs as stripped, non-empty strings and remove potential trailing '.0'
    normalized_ids = set()
    for uid in user_ids:
        if uid is None:
            continue
        s = str(uid).strip()
        # If the string ends with '.0' and the rest are digits, strip the decimal part
        if s.endswith(".0") and s[:-2].isdigit():
            s = s[:-2]
        if s:
            normalized_ids.add(s)

    return normalized_ids



def get_author(entry: dict) -> str:
    """
    Return the user identifier as a clean string.
    Tries Author → UserID → userID → userName, skips NaN/None/''.
    """
    for key in ("Author", "UserID", "userID", "userName"):
        val = entry.get(key)
        if val is None or pd.isna(val):           # filters out np.nan
            continue
        # If it is a float that represents an int (e.g. 3.0) cast to int first
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        s = str(val).strip()
        if s and s.lower() != "nan":
            return s
    return ""

def main(input_file: str, output_dir: str, chunk_size: int, stages: str, task_id: int = 0, total_tasks: int = 1) -> None:
    print_banner()
    ensure_output_directory(os.path.join(output_dir, "_"))
    split = determine_split(input_file)

    # Log task information for array jobs
    if total_tasks > 1:
        print(f"Task {task_id + 1}/{total_tasks}: Processing TUSC {split} data")
        print(f"Input file: {input_file}")
        print(f"Chunk size: {chunk_size}")
        print(f"Stages: {stages}")

    user_ids = set()

    # Stage 1: Detect self-identified users
    if stages in ["1", "both"]:
        print("Stage 1: Detect self-identified users")
        detector = SelfIdentificationDetector()

        parquet_file = pq.ParquetFile(input_file)

        # Calculate total batches for progress bar
        total_rows = parquet_file.metadata.num_rows
        total_batches = (total_rows // chunk_size) + 1
        
        # Calculate how many batches this task will process
        batches_for_this_task = sum(1 for batch_idx in range(total_batches) if batch_idx % total_tasks == task_id)
        
        if total_tasks > 1:
            print(f"Task {task_id + 1}/{total_tasks}: Will process {batches_for_this_task} out of {total_batches} total batches")

        # Streaming setup to avoid memory issues
        BUFFER_SIZE = 200_000  # Write to disk every 200k results
        buffered_results = []
        written_count = 0
        timeout_count = 0
        processed_batches = 0

        # Log initial memory state
        print("Initial memory state:")
        log_memory_usage()

        for batch_idx, batch in enumerate(tqdm(
            parquet_file.iter_batches(batch_size=chunk_size), 
            total=total_batches,
            desc=f"Task {task_id + 1}/{total_tasks} Stage 1"
        )):
            # Skip batches not assigned to this task
            if batch_idx % total_tasks != task_id:
                continue
                
            processed_batches += 1
            df = batch.to_pandas()
            
            # Log memory usage every 50 processed batches (not every 50 total batches)
            if processed_batches % 50 == 0:
                log_memory_usage(f"Task {task_id + 1}/{total_tasks} - Processed batch {processed_batches}")
            
            for row_idx, row in df.iterrows():
                entry = row.to_dict()
                try:
                    age_matches, formatted_demographics = safe_detect_self_identification(
                        entry, detector
                    )
                except Timeout as e:
                    print(f"TIMEOUT in batch {batch_idx}, row {row_idx}: {e}")
                    print(f"Tweet content length: {len(str(entry.get('Tweet', '')))}")
                    print(f"Tweet preview: {str(entry.get('Tweet', ''))[:200]}...")
                    timeout_count += 1
                    continue
                except Exception as e:
                    print(f"ERROR in batch {batch_idx}, row {row_idx}: {e}")
                    continue
                
                if not age_matches:
                    continue
                rec = entry.copy()
                rec["self_identification"] = age_matches
                rec.update(formatted_demographics)
                buffered_results.append(rec)

                # Stream to disk when buffer is full
                if len(buffered_results) >= BUFFER_SIZE:
                    output_path = os.path.join(output_dir, f"{split}_users.tsv")
                    append_results_to_csv(
                        buffered_results,
                        output_path,
                        output_tsv=True,
                        data_source="tusc",
                        split=split,
                    )
                    written_count += len(buffered_results)
                    print(f"Task {task_id + 1}/{total_tasks}: Streamed {len(buffered_results)} results to disk. Total written: {written_count}")
                    buffered_results.clear()
                    
                    # Log memory after flush
                    log_memory_usage(f"Task {task_id + 1}/{total_tasks} - Batch {processed_batches} (after flush)")

        # Write remaining buffered results
        if buffered_results:
            output_path = os.path.join(output_dir, f"{split}_users.tsv")
            append_results_to_csv(
                buffered_results,
                output_path,
                output_tsv=True,
                data_source="tusc",
                split=split,
            )
            written_count += len(buffered_results)

        total_found = written_count
        print(f"Task {task_id + 1}/{total_tasks}: Found {total_found} self-identified users")
        if timeout_count > 0:
            print(f"WARNING: {timeout_count} entries timed out during processing")

        # For stage 2, we need to load the user IDs from the shared file
        if stages == "both":
            users_file = os.path.join(output_dir, f"{split}_users.tsv")
            user_ids = load_self_identified_users(users_file)
            print(f"Task {task_id + 1}/{total_tasks}: Loaded {len(user_ids)} user IDs for stage 2")
        
        # Final memory state
        print(f"Task {task_id + 1}/{total_tasks}: Final stage 1 memory state:")
        log_memory_usage()

    # Stage 2: Collect posts from self-identified users and compute features
    if stages in ["2", "both"]:
        print("Stage 2: Collect posts from self-identified users and compute features")

        # If we didn't run stage 1, load user IDs from shared file
        if stages == "2":
            users_file = os.path.join(output_dir, f"{split}_users.tsv")
            if not os.path.exists(users_file):
                raise FileNotFoundError(f"Self-identified users file not found: {users_file}")
            
            user_ids = load_self_identified_users(users_file)
            print(f"Task {task_id + 1}/{total_tasks}: Loaded {len(user_ids)} self-identified users from {users_file}")
        else:
            users_file = os.path.join(output_dir, f"{split}_users.tsv")

        # Load all user demographics for age calculation and feature enrichment
        df_users = pd.read_csv(users_file, sep="\t", dtype=str)
        df_users["Author"] = df_users["Author"].astype(str)

        # Aggregate demographics per user to handle duplicates
        df_users = aggregate_user_demographics(df_users, data_source="tusc")

        # Create user map after aggregation to ensure unique authors
        user_map = df_users.set_index("Author").to_dict(orient="index")

        # Streaming setup for stage 2
        BUFFER_SIZE = 100_000  # Smaller buffer for posts (they have more features)
        buffered_posts = []
        written_posts = 0
        timeout_count_stage2 = 0
        processed_batches_stage2 = 0

        parquet_file = pq.ParquetFile(input_file)
        total_rows = parquet_file.metadata.num_rows
        total_batches = (total_rows // chunk_size) + 1
        
        # Calculate how many batches this task will process
        batches_for_this_task = sum(1 for batch_idx in range(total_batches) if batch_idx % total_tasks == task_id)
        
        if total_tasks > 1:
            print(f"Task {task_id + 1}/{total_tasks}: Will process {batches_for_this_task} out of {total_batches} total batches for stage 2")

        # Log initial memory state for stage 2
        print("Stage 2 initial memory state:")
        log_memory_usage()

        for batch_idx, batch in enumerate(tqdm(
            parquet_file.iter_batches(batch_size=chunk_size), 
            total=total_batches,
            desc=f"Task {task_id + 1}/{total_tasks} Stage 2"
        )):
            # Skip batches not assigned to this task
            if batch_idx % total_tasks != task_id:
                continue
                
            processed_batches_stage2 += 1
            df = batch.to_pandas()
            
            # Log memory usage every 50 processed batches
            if processed_batches_stage2 % 50 == 0:
                log_memory_usage(f"Task {task_id + 1}/{total_tasks} Stage2 - Processed batch {processed_batches_stage2}")
            
            for row_idx, row in df.iterrows():
                entry = row.to_dict()
                # Only include posts by self-identified users
                author = get_author(entry)
                if author not in user_ids:
                    continue
                rec = entry.copy()
                rec["Author"] = author or ""
                
                try:
                    features = safe_apply_linguistic_features(entry["Tweet"])
                except Timeout as e:
                    print(f"TIMEOUT in stage 2 batch {batch_idx}, row {row_idx}: {e}")
                    print(f"Author: {author}")
                    print(f"Tweet content length: {len(str(entry.get('Tweet', '')))}")
                    print(f"Tweet preview: {str(entry.get('Tweet', ''))[:200]}...")
                    timeout_count_stage2 += 1
                    continue
                except Exception as e:
                    print(f"ERROR in stage 2 batch {batch_idx}, row {row_idx}: {e}")
                    print(f"Author: {author}")
                    continue
                
                rec.update(features)
                # Add all demographic data from the user map
                if author in user_map:
                    user_demographics = user_map[author]
                    for key, value in user_demographics.items():
                        if pd.notna(value):
                            rec[key] = value
                    # Compute age at post from birthyear mapping (assume birthdate Jan 1)
                    if "DMGMajorityBirthyear" in user_demographics and pd.notna(
                        user_demographics["DMGMajorityBirthyear"]
                    ):
                        birthyear = int(user_demographics["DMGMajorityBirthyear"])
                        year = int(rec.get("Year"))
                        rec["DMGAgeAtPost"] = year - birthyear
                    else:
                        rec["DMGAgeAtPost"] = ""
                else:
                    rec["DMGAgeAtPost"] = ""
                buffered_posts.append(rec)

                # Stream to disk when buffer is full
                if len(buffered_posts) >= BUFFER_SIZE:
                    output_path = os.path.join(output_dir, f"{split}_user_posts.tsv")
                    append_results_to_csv(
                        buffered_posts,
                        output_path,
                        output_tsv=True,
                        data_source="tusc",
                        split=split,
                    )
                    written_posts += len(buffered_posts)
                    print(f"Task {task_id + 1}/{total_tasks}: Streamed {len(buffered_posts)} posts to disk. Total written: {written_posts}")
                    buffered_posts.clear()
                    
                    # Log memory after flush
                    log_memory_usage(f"Task {task_id + 1}/{total_tasks} Stage2 - Batch {processed_batches_stage2} (after flush)")

        # Write remaining buffered posts
        if buffered_posts:
            output_path = os.path.join(output_dir, f"{split}_user_posts.tsv")
            append_results_to_csv(
                buffered_posts,
                output_path,
                output_tsv=True,
                data_source="tusc",
                split=split,
            )
            written_posts += len(buffered_posts)

        print(f"Task {task_id + 1}/{total_tasks}: Found {written_posts} posts from self-identified users")
        if timeout_count_stage2 > 0:
            print(f"WARNING: {timeout_count_stage2} entries timed out during stage 2 processing")
        
        # Final memory state
        print(f"Task {task_id + 1}/{total_tasks}: Final stage 2 memory state:")
        log_memory_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TUSC processing pipeline")
    parser.add_argument(
        "--input_file", required=True, help="Path to input Parquet file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write output TSVs"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of rows per batch when streaming Parquet",
    )
    parser.add_argument(
        "--stages",
        choices=["1", "2", "both"],
        default="both",
        help="Which stages to run: '1' for self-identification detection only, '2' for post collection only, 'both' for complete pipeline",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="ID of the current task (for array jobs)",
    )
    parser.add_argument(
        "--total_tasks",
        type=int,
        default=1,
        help="Total number of tasks (for array jobs)",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.chunk_size, args.stages, args.task_id, args.total_tasks)
