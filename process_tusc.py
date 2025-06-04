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
from datetime import datetime


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
    for col in ['Author', 'userID']:
        if col in df.columns:
            user_ids.update(df[col].dropna().astype(str).tolist())
    # Normalize IDs as stripped, non-empty strings
    user_ids = {uid.strip() for uid in user_ids if uid.strip()}
    return user_ids


def main(input_file: str, output_dir: str, chunk_size: int, stages: str) -> None:
    ensure_output_directory(os.path.join(output_dir, "_"))
    split = determine_split(input_file)
    
    self_results = []
    user_ids = set()
    # Mapping from author to birthyear for age-at-post
    _user_birthyear_map: dict[str, int] = {}
    
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
        
        # Extract user IDs for stage 2 (normalized to strings)
        user_ids = {
            str(r.get("Author") or r.get("userID") or r.get("userName") or "").strip()
            for r in self_results
        }

    # Stage 2: Collect posts from self-identified users and compute features
    if stages in ["2", "both"]:
        print("Stage 2: Collect posts from self-identified users and compute features")

        # If we didn't run stage 1, load user IDs and birthyear mapping from existing file
        self_users_file = os.path.join(output_dir, f"{split}_self_users.tsv")
        if stages == "2":
            user_ids = load_self_identified_users(self_users_file)
            print(f"Loaded {len(user_ids)} self-identified users from {self_users_file}")
        
        # Load birthyear mapping for age calculation
        df_users = pd.read_csv(self_users_file, sep='\t')
        _user_birthyear_map = df_users.set_index('Author')['DMGMajorityBirthyear'].to_dict()
        
        print(f"_user_birthyear_map: {_user_birthyear_map}")

        posts_results: list[dict] = []

        parquet_file = pq.ParquetFile(input_file)
        total_rows = parquet_file.metadata.num_rows
        total_batches = (total_rows // chunk_size) + 1
        
        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), total=total_batches):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                entry = row.to_dict()
                # Only include posts by self-identified users
                author_val = entry.get("userID") or entry.get("Author")
                author = str(author_val).strip() if author_val is not None else ""
                if author not in user_ids:
                    continue
                rec = entry.copy()
                rec["Author"] = author or ""
                features = apply_linguistic_features(entry.get("PostText", ""))
                rec.update(features)
                # Compute age at post from birthyear mapping (assume birthdate Jan 1)
                birthyear = _user_birthyear_map.get(author)
                year = int(rec.get("Year", 0))
                rec["DMGAgeAtPost"] = year - int(birthyear) if birthyear is not None else ""
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