#!/usr/bin/env python3
"""
Process TUSC pipeline: detect self-identified users and collect posts with linguistic features.
"""
import os
import argparse

import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

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


def load_self_identified_users(csv_path: str) -> set:
    """Load user IDs from existing self-identified users CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Self-identified users file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, sep='\t')
    user_ids = set()
    
    # Handle different possible column names
    for col in ['Author', 'userID', 'UserID']:
        if col in df.columns:
            user_ids.update(df[col].dropna().astype(str))
    
    return user_ids


def main(input_file: str, output_dir: str, chunk_size: int, stages: str) -> None:
    ensure_output_directory(os.path.join(output_dir, "_"))
    split = determine_split(input_file)
    
    self_results = []
    user_ids = set()
    
    # Stage 1: Detect self-identified users
    if stages in ["1", "both"]:
        print("Stage 1: Detect self-identified users")
        detector = SelfIdentificationDetector()
        
        parquet_file = pq.ParquetFile(input_file)

        # Calculate total batches for progress bar
        total_rows = parquet_file.metadata.num_rows
        total_batches = (total_rows // chunk_size) + 1

        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), total=total_batches):
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
            os.path.join(output_dir, f"{split}_self_users.tsv"),
            output_tsv=True,
            data_source="tusc",
            split=split,
        )

        print(f"Found {len(self_results)} self-identified users")
        
        # Extract user IDs for stage 2
        user_ids = {r.get("Author") or r.get("userID") for r in self_results}

    # Stage 2: Collect posts from self-identified users and compute features
    if stages in ["2", "both"]:
        print("Stage 2: Collect posts from self-identified users and compute features")
        
        # If we didn't run stage 1, load user IDs from existing file
        if stages == "2":
            self_users_file = os.path.join(output_dir, f"{split}_self_users.tsv")
            user_ids = load_self_identified_users(self_users_file)
            print(f"Loaded {len(user_ids)} self-identified users from {self_users_file}")
        
        posts_results: list[dict] = []

        parquet_file = pq.ParquetFile(input_file)
        total_rows = parquet_file.metadata.num_rows
        total_batches = (total_rows // chunk_size) + 1
        
        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), total=total_batches):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                entry = row.to_dict()
                author = entry.get("UserID") or entry.get("userID") or entry.get("Author")
                name = entry.get("UserName") or entry.get("userName") or entry.get("AuthorName")
                if (author not in user_ids) or (name not in user_ids):
                    continue
                rec = entry.copy()
                rec["Author"] = author or ""
                rec["AuthorName"] = name or ""
                features = apply_linguistic_features(entry.get("PostText", ""))
                rec.update(features)
                posts_results.append(rec)
        
        write_results_to_csv(
            posts_results,
            os.path.join(output_dir, f"{split}_user_posts.tsv"),
            output_tsv=True,
            data_source="tusc",
            split=split,
        )

        print(f"Found {len(posts_results)} posts from self-identified users")


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
    parser.add_argument(
        "--stages", choices=["1", "2", "both"], default="both",
        help="Which stages to run: '1' for self-identification detection only, '2' for post collection only, 'both' for complete pipeline"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.chunk_size, args.stages)