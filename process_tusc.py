#!/usr/bin/env python3
"""
Process TUSC pipeline: detect self-identified users and collect posts with linguistic features.
"""
import os
import argparse

import pyarrow.parquet as pq
import pandas as pd

from helpers import (
    detect_self_identification_in_tusc_entry,
    SelfIdentificationDetector,
    apply_linguistic_features,
    write_results_to_csv,
    ensure_output_directory,
)


def determine_split(input_file: str) -> str:
    fname = os.path.basename(input_file).lower()
    return "city" if "city" in fname else "country"


def main(input_file: str, output_dir: str, chunk_size: int) -> None:
    ensure_output_directory(os.path.join(output_dir, "_"))
    split = determine_split(input_file)
    detector = SelfIdentificationDetector()

    # Stage 1: Detect self-identified users
    self_results: list[dict] = []
    parquet_file = pq.ParquetFile(input_file)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            entry = row.to_dict()
            matches = detect_self_identification_in_tusc_entry(entry, detector)
            if not matches:
                continue
            rec = entry.copy()
            rec["self_identification"] = matches
            self_results.append(rec)
    write_results_to_csv(
        self_results,
        os.path.join(output_dir, f"{split}_self_users.csv"),
        output_tsv=True,
        data_source="tusc",
        split=split,
    )

    # Stage 2: Collect posts from self-identified users and compute features
    user_ids = {r.get("Author") or r.get("userID") for r in self_results}
    posts_results: list[dict] = []
    for batch in pq.ParquetFile(input_file).iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            entry = row.to_dict()
            author = entry.get("UserID") or entry.get("userID")
            name = entry.get("UserName") or entry.get("userName")
            if (author not in user_ids) and (name not in user_ids):
                continue
            rec = entry.copy()
            rec["Author"] = author or ""
            rec["AuthorName"] = name or ""
            features = apply_linguistic_features(entry.get("Tweet", ""))
            rec.update(features)
            posts_results.append(rec)
    write_results_to_csv(
        posts_results,
        os.path.join(output_dir, f"{split}_user_posts.csv"),
        output_tsv=True,
        data_source="tusc",
        split=split,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TUSC processing pipeline"
    )
    parser.add_argument(
        "--input_file", required=True, help="Path to input Parquet file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write output TSVs"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100000,
        help="Number of rows per batch when streaming Parquet"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.chunk_size)