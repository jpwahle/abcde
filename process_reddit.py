#!/usr/bin/env python3
"""
Process Reddit pipeline: detect self-identified users and collect posts with linguistic features.
"""
import os
import json
import argparse
import itertools
import multiprocessing
import pandas as pd

from helpers import (
    get_all_jsonl_files,
    filter_entry,
    extract_columns,
    SelfIdentificationDetector,
    detect_self_identification_in_entry,
    apply_linguistic_features,
    write_results_to_csv,
    ensure_output_directory,
)

# Global detector for stage1 self-identification detection
_detector = SelfIdentificationDetector()

# Global set of user IDs for stage2 filtering
_user_ids = set()


def load_self_identified_users(csv_path: str) -> set:
    """Load user IDs from existing self-identified users CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Self-identified users file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, sep='\t')
    user_ids = set()
    
    # Handle different possible column names
    for col in ['author', 'Author', 'userID']:
        if col in df.columns:
            user_ids.update(df[col].dropna().astype(str))
    
    return user_ids


def process_chunk_stage1(task):
    path, lines = task
    results_local: list[dict] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not filter_entry(entry, split="text", min_words=5, max_words=1000):
            continue
        matches = detect_self_identification_in_entry(entry, _detector)
        if not matches:
            continue
        author = entry.get("author")
        if not author or author in ("[deleted]", "AutoModerator", "Bot"):
            continue
        results_local.append(
            {"author": author, "self_identification": matches, "post": extract_columns(entry, None)}
        )
    print(f"Processed {len(results_local)} posts from {path}. Found {len(matches)} self-identified users.")
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
            matches = detect_self_identification_in_entry(entry, _detector)
            if not matches:
                continue
            author = entry.get("author")
            if not author or author in ("[deleted]", "AutoModerator", "Bot"):
                continue
            results_local.append(
                {"author": author, "self_identification": matches, "post": extract_columns(entry, None)}
            )
    return results_local


def process_chunk_stage2(task):
    path, lines = task
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
        features = apply_linguistic_features(post.get("selftext", ""))
        post.update(features)
        results_local.append(post)
    return results_local


def process_file_stage2(file_path: str) -> list[dict]:
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
            features = apply_linguistic_features(post.get("selftext", ""))
            post.update(features)
            results_local.append(post)
    return results_local


def main(input_dir: str, output_dir: str, workers: int = 1, chunk_size: int = 0, stages: str = "both") -> None:
    ensure_output_directory(os.path.join(output_dir, "_"))

    files = get_all_jsonl_files(input_dir)
    # helper to split a JSONL file into chunks of lines
    def read_jsonl_chunks(path: str) -> list[list[str]]:
        with open(path, "r", encoding="utf-8") as fh:
            for lines in iter(lambda: list(itertools.islice(fh, chunk_size)), []):
                yield lines

    self_results = []
    
    # Stage 1: Detect self-identified users (by file or by chunk)
    if stages in ["1", "both"]:
        print("Stage 1: Detect self-identified users")
        
        if chunk_size and chunk_size > 0:
            # split files into line-based chunks to limit memory per task
            tasks1 = [(fp, chunk) for fp in files for chunk in read_jsonl_chunks(fp)]
            if workers > 1:
                with multiprocessing.Pool(workers) as pool:
                    parts = pool.map(process_chunk_stage1, tasks1)
                self_results = [r for part in parts for r in part]
            else:
                self_results: list[dict] = []
                for task in tasks1:
                    self_results.extend(process_chunk_stage1(task))
        else:
            # process entire files
            if workers > 1:
                with multiprocessing.Pool(workers) as pool:
                    file_results = pool.map(process_file_stage1, files)
                self_results = [item for sublist in file_results for item in sublist]
            else:
                self_results: list[dict] = []
                for file_path in files:
                    self_results.extend(process_file_stage1(file_path))

        write_results_to_csv(
            self_results,
            os.path.join(output_dir, "reddit_users.tsv"),
            output_tsv=True,
            data_source="reddit",
            split="text",
        )
        
        print(f"Found {len(self_results)} self-identified users")

    # Stage 2: Collect posts by self-identified users and compute features
    if stages in ["2", "both"]:
        print("Stage 2: Collect posts from self-identified users and compute features")
        
        global _user_ids
        
        # If we didn't run stage 1, load user IDs from existing file
        if stages == "2":
            self_users_file = os.path.join(output_dir, "reddit_users.tsv")
            _user_ids = load_self_identified_users(self_users_file)
            print(f"Loaded {len(_user_ids)} self-identified users from {self_users_file}")
        else:
            _user_ids = {r["author"] for r in self_results}

        if chunk_size and chunk_size > 0:
            tasks2 = [(fp, chunk) for fp in files for chunk in read_jsonl_chunks(fp)]
            if workers > 1:
                with multiprocessing.Pool(workers) as pool:
                    parts = pool.map(process_chunk_stage2, tasks2)
                posts_results = [r for part in parts for r in part]
            else:
                posts_results: list[dict] = []
                for task in tasks2:
                    posts_results.extend(process_chunk_stage2(task))
        else:
            if workers > 1:
                with multiprocessing.Pool(workers) as pool:
                    file_results = pool.map(process_file_stage2, files)
                posts_results = [item for sublist in file_results for item in sublist]
            else:
                posts_results: list[dict] = []
                for file_path in files:
                    posts_results.extend(process_file_stage2(file_path))

        write_results_to_csv(
            posts_results,
            os.path.join(output_dir, "reddit_users_posts.tsv"),
            output_tsv=True,
            data_source="reddit",
            split="text",
        )
        
        print(f"Found {len(posts_results)} posts from self-identified users")


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
        "--stages", choices=["1", "2", "both"], default="both",
        help="Which stages to run: '1' for self-identification detection only, '2' for post collection only, 'both' for complete pipeline"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.workers, args.chunk_size, args.stages)
