#!/usr/bin/env python3
"""
Process Reddit pipeline: detect self-identified users and collect posts with linguistic features.
"""
import os
import json
import argparse
import itertools
import math
import pandas as pd

from helpers import (
    get_all_jsonl_files,
    filter_entry,
    extract_columns,
    SelfIdentificationDetector,
    detect_self_identification_with_resolved_age,
    apply_linguistic_features,
    append_results_to_csv,
    ensure_output_directory,
)
from datetime import datetime

# Global detector for stage1 self-identification detection
_detector = SelfIdentificationDetector()
_user_ids = set()
_user_birthyear_map = {}


def log_with_timestamp(message: str) -> None:
    """Print a message with a timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def count_lines(path: str) -> int:
    """Return the number of lines in a text file."""
    count = 0
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b""):
            count += buf.count(b"\n")
    return count


def read_lines_range(path: str, start: int, n: int) -> list[str]:
    """Read a specific range of lines from a text file."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = itertools.islice(fh, start, start + n)
        return list(lines)


def load_self_identified_users(csv_path: str) -> set:
    """Load user IDs from existing self-identified users CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Self-identified users file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep="\t")
    user_ids = set()

    # Handle different possible column names
    for col in ["author", "Author", "userID"]:
        if col in df.columns:
            user_ids.update(df[col].dropna().astype(str))

    return user_ids


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
        matches = detect_self_identification_with_resolved_age(entry, _detector)
        if not matches:
            continue
        author = entry.get("author")
        if not author or author in ("[deleted]", "AutoModerator", "Bot"):
            continue
        results_local.append(
            {
                "author": author,
                "self_identification": matches,
                "post": extract_columns(entry, None),
            }
        )
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
            matches = detect_self_identification_with_resolved_age(entry, _detector)
            if not matches:
                continue
            author = entry.get("author")
            if not author or author in ("[deleted]", "AutoModerator", "Bot"):
                continue
            results_local.append(
                {
                    "author": author,
                    "self_identification": matches,
                    "post": extract_columns(entry, None),
                }
            )
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
        birthyear = int(_user_birthyear_map[author])
        ts = post.get("created_utc")
        post_year = datetime.utcfromtimestamp(int(ts)).year
        post["DMGAgeAtPost"] = post_year - birthyear
        results_local.append(post)
    log_with_timestamp(f"Processed chunk {chunk_idx + 1}/{total_chunks_for_task}: {len(lines)} posts from {path}.")
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
            birthyear = int(_user_birthyear_map[author])
            ts = post.get("created_utc")
            post_year = datetime.utcfromtimestamp(int(ts)).year
            post["DMGAgeAtPost"] = post_year - birthyear
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
    
    log_with_timestamp(f"Running with {workers} workers, {chunk_size} chunk size, {stages} stages, {task_id} task ID, {total_tasks} total tasks, {linecount_dir} linecount directory")

    ensure_output_directory(os.path.join(output_dir, "_"))

    files = get_all_jsonl_files(input_dir)

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
                            log_with_timestamp(f"Linecount file {lc_path} found. {n_lines} lines.")
                    except Exception:
                        log_with_timestamp(f"Error reading linecount file {lc_path}. Counting lines for {fp}.")
                        n_lines = count_lines(fp)
                else:
                    log_with_timestamp(f"Linecount file not found: {lc_path}. Counting lines for {fp}.")
                    n_lines = count_lines(fp)
            else:
                log_with_timestamp(f"No linecount directory provided, counting lines for {fp}")
                n_lines = count_lines(fp)
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
                        lines = read_lines_range(fp, start, count)
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
                part = process_chunk_stage1((fp, lines, chunk_idx, total_chunks_for_task))
            append_results_to_csv(
                part,
                users_path,
                output_tsv=True,
                data_source="reddit",
                split="text",
            )
            self_user_ids.extend([r["author"] for r in part])

        log_with_timestamp(f"Task {task_id} found {len(self_user_ids)} self-identified users")

    # Stage 2: Collect posts by self-identified users and compute features
    if stages in ["2", "both"]:
        log_with_timestamp("Stage 2: Collect posts from self-identified users and compute features")

        global _user_ids
        # If we didn't run stage 1, or to load birthyear mapping, load user IDs and birthyear info
        self_users_file = os.path.join(output_dir, "reddit_users.tsv")
        if stages == "2":
            _user_ids = load_self_identified_users(self_users_file)
            log_with_timestamp(
                f"Loaded {len(_user_ids)} self-identified users from {self_users_file}"
            )
        else:
            _user_ids = set(self_user_ids)

        # Load DMG birthyear mapping for age-at-post computation
        df_users = pd.read_csv(self_users_file, sep="\t")

        global _user_birthyear_map
        _user_birthyear_map = df_users.set_index("Author")[
            "DMGMajorityBirthyear"
        ].to_dict()

        posts_path = os.path.join(output_dir, "reddit_users_posts.tsv")
        total_posts = 0
        for fp, lines, chunk_idx, total_chunks_for_task in generate_tasks(files):
            if lines is None:
                part = process_file_stage2(fp)
            else:
                part = process_chunk_stage2((fp, lines, chunk_idx, total_chunks_for_task))
            append_results_to_csv(
                part,
                posts_path,
                output_tsv=True,
                data_source="reddit",
                split="text",
            )
            total_posts += len(part)

        log_with_timestamp(f"Task {task_id} found {total_posts} posts from self-identified users")


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
