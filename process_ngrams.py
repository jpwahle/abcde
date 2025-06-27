#!/usr/bin/env python3
"""
Optimized distributed processing of Google Books Ngram corpus data for SLURM.

This version uses byte-offset indexing for fast random access to large files,
similar to the Reddit processing pipeline.
"""
from __future__ import annotations

import argparse
import array
import csv
import mmap
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    FAST_IO_AVAILABLE = True
except ImportError:
    FAST_IO_AVAILABLE = False
    print("Warning: numpy not available. Fast indexing requires numpy.")

from helpers import apply_linguistic_features


def log_with_timestamp(message: str) -> None:
    """Print a message with a timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


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


def build_index(ngram_path: str, index_path: Optional[str] = None) -> str:
    """Build index file for fast random access to ngram file lines."""
    if not FAST_IO_AVAILABLE:
        raise RuntimeError("Fast I/O requires numpy package")

    index_path = index_path or str(Path(ngram_path).with_suffix(".idx"))

    log_with_timestamp(f"Building index for {ngram_path}")
    offsets = array.array("Q")  # 64-bit unsigned
    pos = 0

    with (
        open(ngram_path, "rb") as f,
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
        time.sleep(30)  # Check every 30 seconds
    return False


def read_lines_indexed(
    ngram_path: str, index_path: str, start: int, stop: int
) -> List[str]:
    """Read lines using pre-built index for fast random access."""
    if not FAST_IO_AVAILABLE:
        raise RuntimeError("Fast I/O requires numpy package")

    offs = np.memmap(index_path, dtype=np.uint64, mode="r")

    # Handle edge cases
    if start >= len(offs):
        return []
    if stop >= len(offs):
        stop = len(offs) - 1

    with open(ngram_path, "rb") as f:
        f.seek(offs[start])
        nbytes = (offs[stop + 1] - offs[start]) if stop + 1 < len(offs) else None
        raw = f.read(nbytes)

    # Return raw lines as strings
    return [line.decode("utf-8") for line in raw.splitlines()]


def count_lines_indexed(index_path: str) -> int:
    """Count lines using index file."""
    if not FAST_IO_AVAILABLE:
        raise RuntimeError("Fast I/O requires numpy package")

    offs = np.memmap(index_path, dtype=np.uint64, mode="r")
    return len(offs) - 1  # Last offset is end-of-file


def get_file_line_count(file_path: Path, index_dir: Optional[Path] = None) -> int:
    """Get the number of lines in a file using index if available."""
    if index_dir and FAST_IO_AVAILABLE:
        index_path = index_dir / f"{file_path.name}.idx"
        if index_path.exists():
            try:
                return count_lines_indexed(str(index_path))
            except Exception as e:
                log_with_timestamp(
                    f"Error using index for line count: {e}. Falling back to sequential count."
                )

    # Fallback to sequential counting
    line_count = 0
    with open(file_path, "rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            line_count += buf.count(b"\n")
    return line_count


def build_task_index(
    input_dir: Path, pattern: str, chunk_size: int, index_dir: Optional[Path] = None
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
        line_count = get_file_line_count(file_path, index_dir)

        # Create chunks for this file
        for start_line in range(0, line_count, chunk_size):
            end_line = min(start_line + chunk_size, line_count)
            tasks.append((file_path, start_line, end_line))

    return tasks


def process_file_chunk_indexed(
    file_path: Path,
    index_path: Path,
    start_line: int,
    end_line: int,
    include_features: bool = True,
) -> List[Dict[str, Any]]:
    """Process a specific chunk of lines from a file using indexed access.

    Args:
        file_path: Path to the ngram file
        index_path: Path to the index file
        start_line: Starting line number (0-based)
        end_line: Ending line number (exclusive)
        include_features: Whether to compute linguistic features

    Returns:
        List of processed ngram records with features
    """
    results = []

    # Read lines using index
    lines = read_lines_indexed(str(file_path), str(index_path), start_line, end_line - 1)

    for line in lines:
        if not line:
            continue

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


def process_file_chunk_sequential(
    file_path: Path,
    start_line: int,
    end_line: int,
    include_features: bool = True,
) -> List[Dict[str, Any]]:
    """Process a specific chunk of lines from a file (fallback method).

    This is the original sequential method used when indexing is not available.
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
        description=(
            "Optimized distributed processing of Google Books Ngram corpus for SLURM"
        )
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
    parser.add_argument(
        "--build_indexes",
        action="store_true",
        help="Build indexes for all matching files before processing",
    )
    parser.add_argument(
        "--use_sequential",
        action="store_true",
        help="Use sequential reading instead of indexed access (slower)",
    )

    args = parser.parse_args()

    # Validate directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create index directory
    index_dir = output_dir / "indexes"
    index_dir.mkdir(exist_ok=True)

    # Build indexes if requested or if this is task 0
    if args.build_indexes or args.task_id == 0:
        if not args.use_sequential and FAST_IO_AVAILABLE:
            log_with_timestamp("Building indexes for all ngram files...")
            ngram_files = sorted(input_dir.glob(args.pattern))
            for file_path in ngram_files:
                index_path = index_dir / f"{file_path.name}.idx"
                if not index_path.exists():
                    build_index(str(file_path), str(index_path))

    # Build task index
    print(f"Building task index for pattern '{args.pattern}'...")
    tasks = build_task_index(
        input_dir, args.pattern, args.chunk_size, index_dir if not args.use_sequential else None
    )

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
    print(f"  Method: {'SEQUENTIAL' if args.use_sequential else 'INDEXED'}")

    # Process the chunk
    start_time = datetime.now()

    if args.use_sequential or not FAST_IO_AVAILABLE:
        results = process_file_chunk_sequential(
            file_path,
            start_line,
            end_line,
            include_features=not args.no_features,
        )
    else:
        # Wait for index if not task 0
        index_path = index_dir / f"{file_path.name}.idx"
        if args.task_id > 0 and not index_path.exists():
            log_with_timestamp(f"Waiting for index file: {index_path}")
            if not wait_for_index(str(index_path)):
                log_with_timestamp(
                    "Timeout waiting for index. Falling back to sequential processing."
                )
                results = process_file_chunk_sequential(
                    file_path,
                    start_line,
                    end_line,
                    include_features=not args.no_features,
                )
            else:
                results = process_file_chunk_indexed(
                    file_path,
                    index_path,
                    start_line,
                    end_line,
                    include_features=not args.no_features,
                )
        else:
            # Build index if needed
            if not index_path.exists():
                build_index(str(file_path), str(index_path))
            results = process_file_chunk_indexed(
                file_path,
                index_path,
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