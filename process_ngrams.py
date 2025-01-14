#!/usr/bin/env python3
"""
Distributed processing of Google Books Ngram corpus data for SLURM.

This script is designed to work with SLURM array jobs where each worker
processes specific chunks across all ngram files in a distributed manner.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from helpers import apply_linguistic_features


def strip_pos_tags(ngram: str) -> str:
    """Remove POS tags from ngram tokens.

    Example:
        "Poetical_NOUN and_CONJ Dramatic_NOUN Works_NOUN ._." -> "Poetical and Dramatic Works ."
    """
    tokens = ngram.split()
    cleaned_tokens = []
    for token in tokens:
        # Remove everything after the last underscore (POS tag)
        if "_" in token:
            cleaned_token = token.rsplit("_", 1)[0]
        else:
            cleaned_token = token
        cleaned_tokens.append(cleaned_token)
    return " ".join(cleaned_tokens)


def process_ngram_line(line: str) -> Optional[Dict[str, Any]]:
    """Process a single ngram line and extract fields.

    Args:
        line: Tab-separated line with format: ngram TAB year TAB match_count TAB book_count

    Returns:
        Dictionary with parsed fields or None if parsing fails
    """
    try:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            return None

        ngram, year_str, match_count_str, book_count_str = parts

        # Parse numeric fields
        year = int(year_str)
        match_count = int(match_count_str)
        book_count = int(book_count_str)

        return {
            "ngram": ngram,
            "year": year,
            "match_count": match_count,
            "book_count": book_count,
            "ngram_cleaned": strip_pos_tags(ngram),
        }
    except (ValueError, IndexError):
        return None


def get_file_line_count(file_path: Path) -> int:
    """Get the number of lines in a file."""
    line_count = 0
    with open(file_path, "rb") as f:
        for _ in f:
            line_count += 1
    return line_count


def build_task_index(
    input_dir: Path, pattern: str, chunk_size: int
) -> List[Tuple[Path, int, int]]:
    """Build an index of all tasks (file chunks) to process.

    Returns:
        List of tuples: (file_path, start_line, end_line)
    """
    tasks = []

    # Find all matching files
    ngram_files = sorted(input_dir.glob(pattern))

    for file_path in ngram_files:
        print(f"Indexing {file_path.name}...")
        line_count = get_file_line_count(file_path)

        # Create chunks for this file
        for start_line in range(0, line_count, chunk_size):
            end_line = min(start_line + chunk_size, line_count)
            tasks.append((file_path, start_line, end_line))

    return tasks


def process_file_chunk(
    file_path: Path,
    start_line: int,
    end_line: int,
    include_features: bool = True,
) -> List[Dict[str, Any]]:
    """Process a specific chunk of lines from a file.

    Args:
        file_path: Path to the ngram file
        start_line: Starting line number (0-based)
        end_line: Ending line number (exclusive)
        include_features: Whether to compute linguistic features

    Returns:
        List of processed ngram records with features
    """
    results = []

    with open(file_path, "r", encoding="utf-8") as f:
        # Skip to start line
        for _ in range(start_line):
            f.readline()

        # Process lines in chunk
        for _ in range(start_line, end_line):
            line = f.readline()
            if not line:
                break

            # Parse the ngram line
            ngram_data = process_ngram_line(line)
            if not ngram_data:
                continue

            # For 5-grams, we expect exactly 5 words
            word_count = len(ngram_data["ngram_cleaned"].split())
            if word_count != 5:
                continue

            # Compute linguistic features if requested
            if include_features:
                try:
                    features = apply_linguistic_features(ngram_data["ngram_cleaned"])
                    ngram_data.update(features)
                except Exception as e:
                    print(f"Error computing features for ngram: {e}")
                    continue

            results.append(ngram_data)

    return results


def main():
    parser = argparse.ArgumentParser(
        description=("Distributed processing of Google Books Ngram corpus for SLURM")
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing ngram files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output TSV files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*5gram*",
        help="File pattern to match (default: '*5gram*' for 5-grams)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of lines per chunk",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="Task ID for this worker (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--no_features",
        action="store_true",
        help="Skip linguistic feature extraction (only parse ngrams)",
    )

    args = parser.parse_args()

    # Validate directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task index
    print(f"Building task index for pattern '{args.pattern}'...")
    tasks = build_task_index(input_dir, args.pattern, args.chunk_size)

    if not tasks:
        print(f"Error: No files matching pattern '{args.pattern}' in {input_dir}")
        sys.exit(1)

    print(f"Total tasks (chunks): {len(tasks)}")

    # Check if task ID is valid
    if args.task_id >= len(tasks):
        print(f"Task ID {args.task_id} exceeds number of tasks ({len(tasks)})")
        sys.exit(0)

    # Get task for this worker
    file_path, start_line, end_line = tasks[args.task_id]
    chunk_lines = end_line - start_line

    print(f"\nTask {args.task_id}:")
    print(f"  File: {file_path.name}")
    print(f"  Lines: {start_line:,} - {end_line:,} ({chunk_lines:,} lines)")
    print(f"  Features: {'ENABLED' if not args.no_features else 'DISABLED'}")

    # Process the chunk
    start_time = datetime.now()
    results = process_file_chunk(
        file_path,
        start_line,
        end_line,
        include_features=not args.no_features,
    )

    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"\nProcessed {len(results):,} ngrams in {processing_time:.2f} seconds")

    # Write results to TSV
    # Use file name and chunk info in output filename
    output_file = output_dir / f"{file_path.stem}_chunk_{args.task_id:06d}.tsv"

    if results:
        # Determine field order - original fields first, then features
        base_fields = ["ngram", "year", "match_count", "book_count"]
        feature_fields = sorted(
            [
                k
                for k in results[0].keys()
                if k not in base_fields and k != "ngram_cleaned"
            ]
        )
        fieldnames = base_fields + feature_fields

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"Results written to: {output_file}")
    else:
        print("No valid ngrams found in this chunk")


if __name__ == "__main__":
    main()
