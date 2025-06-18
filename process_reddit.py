#!/usr/bin/env python3
"""
Process Reddit pipeline: detect self-identified users and collect posts with linguistic features.
"""
import argparse
import array
import itertools
import json
import math
import mmap
import os
import pathlib
import time
from typing import Optional

import pandas as pd

try:
    import numpy as np
    import orjson

    FAST_IO_AVAILABLE = True
except ImportError:
    FAST_IO_AVAILABLE = False
    print("Warning: numpy and/or orjson not available. Falling back to slower I/O.")

from datetime import datetime

from helpers import (
    SelfIdentificationDetector,
    append_results_to_csv,
    apply_linguistic_features,
    detect_self_identification_with_resolved_age,
    ensure_output_directory,
    extract_columns,
    filter_entry,
    format_demographic_detections_for_output,
    get_all_jsonl_files,
)

# Global detector for stage1 self-identification detection
_detector = SelfIdentificationDetector()
_user_ids = set()
_user_birthyear_map = {}


def log_with_timestamp(message: str) -> None:
    """Print a message with a timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def build_index(jsonl_path: str, index_path: Optional[str] = None) -> str:
    """Build index file for fast random access to JSONL lines."""
    if not FAST_IO_AVAILABLE:
        raise RuntimeError("Fast I/O requires numpy and orjson packages")

    index_path = index_path or str(pathlib.Path(jsonl_path).with_suffix(".idx"))

    log_with_timestamp(f"Building index for {jsonl_path}")
    offsets = array.array("Q")  # 64-bit unsigned
    pos = 0

    with (
        open(jsonl_path, "rb") as f,
        mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
        while True:
            offsets.append(pos)
            nl = mm.find(b"\n", pos)
            if nl == -1:
                break
            pos = nl + 1

    with open(index_path, "wb") as idx_file:
        offsets.tofile(idx_file)

    log_with_timestamp(f"Index built: {index_path} with {len(offsets)} lines")
    return index_path


def wait_for_index(index_path: str, timeout: int = 3600) -> bool:
    """Wait for index file to be created by another task."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(index_path):
            # Additional check: ensure the file is complete by checking if it's been stable for 1 second
            time.sleep(1)
            if os.path.exists(index_path):
                return True
        time.sleep(30)  # Check every 5 seconds
    return False


def read_lines_indexed(
    jsonl_path: str, index_path: str, start: int, stop: int
) -> list[str]:
    """Read lines using pre-built index for fast random access."""
    if not FAST_IO_AVAILABLE:
        raise RuntimeError("Fast I/O requires numpy and orjson packages")

    offs = np.memmap(index_path, dtype=np.uint64, mode="r")

    # Handle edge cases
    if start >= len(offs):
        return []
    if stop >= len(offs):
        stop = len(offs) - 1

    with open(jsonl_path, "rb") as f:
        f.seek(offs[start])
        nbytes = (offs[stop + 1] - offs[start]) if stop + 1 < len(offs) else None
        raw = f.read(nbytes)

    # Return raw lines as strings (to match existing interface)
    return [line.decode("utf-8") for line in raw.splitlines()]


def count_lines_indexed(index_path: str) -> int:
    """Count lines using index file."""
    if not FAST_IO_AVAILABLE:
        raise RuntimeError("Fast I/O requires numpy and orjson packages")

    offs = np.memmap(index_path, dtype=np.uint64, mode="r")
    return len(offs) - 1  # Last offset is end-of-file


def count_lines(path: str, output_dir: str = None) -> int:
    """Return the number of lines in a text file."""
    # Try to use index if available
    if output_dir and FAST_IO_AVAILABLE:
        filename = os.path.basename(path)
        index_path = os.path.join(output_dir, "indexes", f"{filename}.idx")
        if os.path.exists(index_path):
            try:
                return count_lines_indexed(index_path)
            except Exception as e:
                log_with_timestamp(
                    f"Error using index for line count: {e}. Falling back to sequential count."
                )

    # Fallback to original method
    count = 0
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b""):
            count += buf.count(b"\n")
    return count


def read_lines_range(
    path: str, start: int, n: int, output_dir: str = None
) -> list[str]:
    """Read a specific range of lines from a text file."""
    # Try to use index if available
    if output_dir and FAST_IO_AVAILABLE:
        filename = os.path.basename(path)
        index_path = os.path.join(output_dir, "indexes", f"{filename}.idx")
        if os.path.exists(index_path):
            try:
                return read_lines_indexed(path, index_path, start, start + n - 1)
            except Exception as e:
                log_with_timestamp(
                    f"Error using index for reading lines: {e}. Falling back to sequential read."
                )

    # Fallback to original method
    with open(path, "r", encoding="utf-8") as fh:
        lines = itertools.islice(fh, start, start + n)
        return list(lines)


def ensure_indexes_built(
    files: list[str], task_id: int, output_dir: str, overwrite: bool = False
) -> None:
    """Ensure all files have indexes built. Only task 0 builds them, others wait."""
    if not FAST_IO_AVAILABLE:
        log_with_timestamp("Fast I/O not available, skipping index building")
        return

    index_dir = os.path.join(output_dir, "indexes")
    os.makedirs(index_dir, exist_ok=True)

    # Create a status file to track index building progress
    status_file = os.path.join(index_dir, "index_status.json")

    # If index was built by another task, we can skip building it again
    if os.path.exists(status_file) and not overwrite:
        with open(status_file, "r") as f:
            status = json.load(f)
        if status.get("completed", False):
            log_with_timestamp("Index building already completed")
            return

    if task_id == 0:
        # Task 0 builds all indexes
        log_with_timestamp("Task 0: Building indexes for all files")

        status = {"files": {}, "completed": False}

        for file_path in files:
            filename = os.path.basename(file_path)
            index_path = os.path.join(index_dir, f"{filename}.idx")

            if not overwrite and os.path.exists(index_path):
                log_with_timestamp(
                    f"Index already exists for file {filename}. Skipping."
                )
                status["files"][filename] = "completed"
                continue

            if not os.path.exists(index_path):
                try:
                    build_index(file_path, index_path)
                    status["files"][filename] = "completed"
                except Exception as e:
                    log_with_timestamp(f"Error building index for {file_path}: {e}")
                    status["files"][filename] = "failed"
            else:
                log_with_timestamp(f"Index already exists: {index_path}")
                status["files"][filename] = "completed"

        status["completed"] = True
        with open(status_file, "w") as f:
            json.dump(status, f)

        log_with_timestamp("Task 0: Index building completed")

    else:
        # Other tasks wait for indexes to be ready
        log_with_timestamp(f"Task {task_id}: Waiting for indexes to be built by task 0")

        timeout = 14400  # 4 hours timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            if os.path.exists(status_file):
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)

                    if status.get("completed", False):
                        log_with_timestamp(f"Task {task_id}: Indexes are ready")
                        return
                except (json.JSONDecodeError, KeyError):
                    pass

            time.sleep(30)  # Check every 30 seconds

        log_with_timestamp(
            f"Task {task_id}: Timeout waiting for indexes. Proceeding without fast I/O."
        )


def process_chunk_stage1(task):
    path, lines, chunk_idx, total_chunks_for_task = task
    results_local: list[dict] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not filter_entry(entry, split="text", min_words=5, max_words=1000):
            continue

        # Get age-resolved detection first
        age_matches = detect_self_identification_with_resolved_age(entry, _detector)
        if not age_matches:
            continue

        # Get full demographic detections with mappings
        title = entry.get("title", "")
        selftext = entry.get("selftext", "")
        combined_text = f"{title} {selftext}"
        demographic_detections = _detector.detect_with_mappings(combined_text)

        # Format demographic fields for output
        formatted_demographics = format_demographic_detections_for_output(
            demographic_detections
        )

        author = entry.get("author")
        if not author or author in ("[deleted]", "AutoModerator", "Bot"):
            continue

        # Combine age-resolved data with other demographics
        result = {
            "author": author,
            "self_identification": age_matches,
            "post": extract_columns(entry, None),
        }
        result.update(formatted_demographics)
        results_local.append(result)

    log_with_timestamp(
        f"Processed chunk {chunk_idx + 1}/{total_chunks_for_task}: {len(lines)} posts from {path}. Found {len(results_local)} self-identified users."
    )
    return results_local


def process_file_stage1(file_path: str) -> list[dict]:
    results_local: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not filter_entry(entry, split="text", min_words=5, max_words=1000):
                continue

            # Get age-resolved detection first
            age_matches = detect_self_identification_with_resolved_age(entry, _detector)
            if not age_matches:
                continue

            # Get full demographic detections with mappings
            title = entry.get("title", "")
            selftext = entry.get("selftext", "")
            combined_text = f"{title} {selftext}"
            demographic_detections = _detector.detect_with_mappings(combined_text)

            # Format demographic fields for output
            formatted_demographics = format_demographic_detections_for_output(
                demographic_detections
            )

            author = entry.get("author")
            if not author or author in ("[deleted]", "AutoModerator", "Bot"):
                continue

            # Combine age-resolved data with other demographics
            result = {
                "author": author,
                "self_identification": age_matches,
                "post": extract_columns(entry, None),
            }
            result.update(formatted_demographics)
            results_local.append(result)

    return results_local


def process_chunk_stage2(task):
    path, lines, chunk_idx, total_chunks_for_task = task
    global _user_birthyear_map
    results_local: list[dict] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        author = entry.get("author")
        if author not in _user_ids:
            continue
        if not filter_entry(entry, split="text", min_words=5, max_words=1000):
            continue
        post = extract_columns(entry, None)
        features = apply_linguistic_features(post["selftext"])
        post.update(features)
        # Compute age at post from birthyear mapping
        if author in _user_birthyear_map and pd.notna(_user_birthyear_map[author]):
            birthyear = int(_user_birthyear_map[author])
            ts = post.get("created_utc")
            post_year = datetime.utcfromtimestamp(int(ts)).year
            post["DMGAgeAtPost"] = post_year - birthyear
        else:
            post["DMGAgeAtPost"] = None
        results_local.append(post)
    log_with_timestamp(
        f"Processed chunk {chunk_idx + 1}/{total_chunks_for_task}: {len(lines)} posts from {path}."
    )
    return results_local


def process_file_stage2(file_path) -> list[dict]:
    global _user_birthyear_map
    results_local: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            author = entry.get("author")
            if author not in _user_ids:
                continue
            if not filter_entry(entry, split="text", min_words=5, max_words=1000):
                continue
            post = extract_columns(entry, None)
            features = apply_linguistic_features(post["selftext"])
            post.update(features)
            # Compute age at post from birthyear mapping
            if author in _user_birthyear_map and pd.notna(_user_birthyear_map[author]):
                birthyear = int(_user_birthyear_map[author])
                ts = post.get("created_utc")
                post_year = datetime.utcfromtimestamp(int(ts)).year
                post["DMGAgeAtPost"] = post_year - birthyear
            else:
                post["DMGAgeAtPost"] = None
            results_local.append(post)
    return results_local


def main(
    input_dir: str,
    output_dir: str,
    workers: int = 1,
    chunk_size: int = 0,
    stages: str = "both",
    task_id: int = 0,
    total_tasks: int = 1,
    linecount_dir: str = None,
) -> None:

    log_with_timestamp(
        f"Running with {workers} workers, {chunk_size} chunk size, {stages} stages, {task_id} task ID, {total_tasks} total tasks, {linecount_dir} linecount directory"
    )

    if FAST_IO_AVAILABLE:
        log_with_timestamp("Fast I/O enabled (using numpy and orjson)")
    else:
        log_with_timestamp(
            "Fast I/O disabled (install numpy and orjson for better performance)"
        )

    ensure_output_directory(os.path.join(output_dir, "_"))

    files = get_all_jsonl_files(input_dir)

    # Build indexes for fast I/O (only task 0 builds, others wait)
    ensure_indexes_built(files, task_id, output_dir)

    def partition_files_by_size(
        paths: list[str], n: int
    ) -> tuple[list[list[str]], list[int]]:
        sizes = [(p, os.path.getsize(p)) for p in paths]
        sizes.sort(key=lambda x: x[1], reverse=True)
        groups = [[] for _ in range(n)]
        totals = [0] * n
        for path, sz in sizes:
            idx = totals.index(min(totals))
            groups[idx].append(path)
            totals[idx] += sz
        return groups, totals

    line_counts: dict[str, int] = {}
    total_chunks = 0

    if total_tasks > 1 and chunk_size == 0:
        # When processing whole files we can simply split the file list between
        # tasks to minimise duplicate work.  Each task handles a subset of files
        # exclusively.
        groups, totals = partition_files_by_size(files, total_tasks)
        files = groups[task_id]
        total_size_gb = totals[task_id] / 1024**3
        total_chunks_per_task = totals[task_id] / total_tasks
        log_with_timestamp(
            f"Task {task_id + 1}/{total_tasks} processing {total_chunks_per_task} chunks from {len(files)} files (~{total_size_gb:.2f} GB)"
        )
    else:
        # With chunked processing all tasks share the same list of files but
        # distribute chunks globally across tasks.
        for fp in files:
            if linecount_dir:
                filename = os.path.basename(fp)
                lc_path = os.path.join(linecount_dir, f"{filename}_linecount")
                if os.path.exists(lc_path):
                    try:
                        with open(lc_path, "r") as lc_f:
                            n_lines = int(lc_f.read().strip())
                            log_with_timestamp(
                                f"Linecount file {lc_path} found. {n_lines} lines."
                            )
                    except Exception:
                        log_with_timestamp(
                            f"Error reading linecount file {lc_path}. Counting lines for {fp}."
                        )
                        n_lines = count_lines(fp, output_dir)
                else:
                    log_with_timestamp(
                        f"Linecount file not found: {lc_path}. Counting lines for {fp}."
                    )
                    n_lines = count_lines(fp, output_dir)
            else:
                log_with_timestamp(
                    f"No linecount directory provided, counting lines for {fp}"
                )
                n_lines = count_lines(fp, output_dir)
            line_counts[fp] = n_lines
            total_chunks += math.ceil(n_lines / chunk_size) if chunk_size else 1
        total_size_gb = sum(os.path.getsize(p) for p in files) / 1024**3
        total_chunks_per_task = total_chunks / total_tasks
        log_with_timestamp(
            f"Task {task_id + 1}/{total_tasks} processing {total_chunks_per_task} chunks from {len(files)} files (~{total_size_gb:.2f} GB)"
        )

    def generate_tasks(paths: list[str]):
        """Yield (file_path, lines_or_none, chunk_idx, total_chunks_for_task) pairs assigned to this array task."""
        # First pass: calculate total chunks this task will process
        task_chunk_count = 0
        idx = 0
        for fp in paths:
            if chunk_size and chunk_size > 0:
                n_lines = line_counts[fp]
                n_chunks = math.ceil(n_lines / chunk_size)
                for chunk_idx in range(n_chunks):
                    if idx % total_tasks == task_id:
                        task_chunk_count += 1
                    idx += 1
            else:
                if idx % total_tasks == task_id:
                    task_chunk_count += 1
                idx += 1

        # Second pass: yield tasks with proper indexing
        idx = 0
        current_task_chunk = 0
        for fp in paths:
            if chunk_size and chunk_size > 0:
                n_lines = line_counts[fp]
                n_chunks = math.ceil(n_lines / chunk_size)
                for chunk_idx in range(n_chunks):
                    if idx % total_tasks == task_id:
                        start = chunk_idx * chunk_size
                        count = (
                            chunk_size if chunk_idx < n_chunks - 1 else n_lines - start
                        )
                        lines = read_lines_range(fp, start, count, output_dir)
                        yield fp, lines, current_task_chunk, task_chunk_count
                        current_task_chunk += 1
                    idx += 1
            else:
                if idx % total_tasks == task_id:
                    yield fp, None, current_task_chunk, task_chunk_count
                    current_task_chunk += 1
                idx += 1

    self_user_ids: list[str] = []
    users_path = os.path.join(output_dir, "reddit_users.tsv")

    # Stage 1: Detect self-identified users (by file or by chunk)
    if stages in ["1", "both"]:
        log_with_timestamp("Stage 1: Detect self-identified users")

        for fp, lines, chunk_idx, total_chunks_for_task in generate_tasks(files):
            if lines is None:
                part = process_file_stage1(fp)
            else:
                part = process_chunk_stage1(
                    (fp, lines, chunk_idx, total_chunks_for_task)
                )
            append_results_to_csv(
                part,
                users_path,
                output_tsv=True,
                data_source="reddit",
                split="text",
            )
            self_user_ids.extend([r["author"] for r in part])

        log_with_timestamp(
            f"Task {task_id} found {len(self_user_ids)} self-identified users"
        )

    # Stage 2: Collect posts by self-identified users and compute features
    if stages in ["2", "both"]:
        log_with_timestamp(
            "Stage 2: Collect posts from self-identified users and compute features"
        )

        global _user_ids
        # If we didn't run stage 1, or to load birthyear mapping, load user IDs and birthyear info
        self_users_file = os.path.join(output_dir, "reddit_users.tsv")
        if stages == "2":
            # Load the TSV once and use it for both user IDs and birthyear mapping
            df_users = pd.read_csv(self_users_file, sep="\t")

            # Extract user IDs from the dataframe (same logic as load_self_identified_users)
            user_ids = set()
            for col in ["author", "Author", "userID"]:
                if col in df_users.columns:
                    user_ids.update(df_users[col].dropna().astype(str))
            _user_ids = user_ids

            log_with_timestamp(
                f"Loaded {len(_user_ids)} self-identified users from {self_users_file}"
            )
        else:
            _user_ids = set(self_user_ids)
            # Still need to load df_users for birthyear mapping
            df_users = pd.read_csv(self_users_file, sep="\t")

        global _user_birthyear_map
        _user_birthyear_map = (
            df_users.set_index("Author")["DMGMajorityBirthyear"].dropna().to_dict()
        )

        posts_path = os.path.join(output_dir, "reddit_users_posts.tsv")
        total_posts = 0
        for fp, lines, chunk_idx, total_chunks_for_task in generate_tasks(files):
            if lines is None:
                part = process_file_stage2(fp)
            else:
                part = process_chunk_stage2(
                    (fp, lines, chunk_idx, total_chunks_for_task)
                )
            append_results_to_csv(
                part,
                posts_path,
                output_tsv=True,
                data_source="reddit",
                split="text",
            )
            total_posts += len(part)

        log_with_timestamp(
            f"Task {task_id} found {total_posts} posts from self-identified users"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reddit processing pipeline")
    parser.add_argument(
        "--input_dir", required=True, help="Directory with RS_*.jsonl files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write output TSVs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for file-level parallelism",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Split large JSONL files into chunks of this many lines",
    )
    parser.add_argument(
        "--stages",
        choices=["1", "2", "both"],
        default="both",
        help="Which stages to run: 1 for self-identification detection only, 2 for post collection only, both for complete pipeline",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="Task index when running as SLURM array",
    )
    parser.add_argument(
        "--total_tasks",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
        help="Total number of tasks in the SLURM array",
    )
    parser.add_argument(
        "--linecount_dir",
        type=str,
        help="Directory containing precomputed linecount files (filename_linecount format)",
    )

    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.workers,
        args.chunk_size,
        args.stages,
        args.task_id,
        args.total_tasks,
        args.linecount_dir,
    )

    log_with_timestamp(f"Done with Reddit pipeline for worker {args.task_id}")
