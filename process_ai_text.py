#!/usr/bin/env python3
"""
Process AI-generated text datasets to compute linguistic features.
"""
import argparse
import os
from datetime import datetime

import pandas as pd
from helpers import apply_linguistic_features, print_banner


def log_with_timestamp(message: str) -> None:
    """Print a message with a timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def main(input_file: str, output_dir: str, dataset_name: str) -> None:
    """
    Main processing function for AI text datasets.
    """
    print_banner()
    log_with_timestamp(
        f"Starting processing for dataset '{dataset_name}' from file: {input_file}"
    )

    # Mapping of dataset names to the columns containing text to be analyzed.
    # For datasets with multiple text fields (e.g., answer and reasoning),
    # a tuple of column names is provided.
    dataset_column_map = {
        "wildchat-1m": "ai_text",
        "lmsys-1m": "ai_text",
        "pippa": "ai_text",
        "hh-rlhf": "ai_text",
        "prism": "ai_text",
        "apt-paraphrase-dataset-gpt-3": "text",
        "anthropic-persuasiveness": "ai_text",
        "M4": "ai_text",
        "mage": "ai_text",
        "luar": "ai_text",
        "general_thoughts_430k": ("model_answer", "model_reasoning"),
        "reasoning_shield": ("model_answer", "model_reasoning"),
        "safechain": ("model_answer", "model_reasoning"),
        "star1": ("model_answer", "model_reasoning"),
        "tinystories": "ai_text",
        "raid": "ai_text",
    }

    if dataset_name not in dataset_column_map:
        log_with_timestamp(f"Error: Dataset '{dataset_name}' is not configured.")
        return

    text_columns = dataset_column_map[dataset_name]

    try:
        # Detect separator (CSV or TSV)
        sep = "\t" if input_file.endswith(".tsv") else ","
        if dataset_name == "wildchat-1m":
            df = pd.read_csv(input_file, sep=sep, low_memory=False, encoding="windows-1252")
        else:
            df = pd.read_csv(input_file, sep=sep, low_memory=False)
    except FileNotFoundError:
        log_with_timestamp(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        log_with_timestamp(f"Error reading input file: {e}")
        return

    if isinstance(text_columns, str):
        # Single text column to process
        if text_columns not in df.columns:
            log_with_timestamp(
                f"Error: Text column '{text_columns}' not found in the input file for dataset '{dataset_name}'."
            )
            return

        log_with_timestamp(f"Applying linguistic features to column '{text_columns}'...")
        features_list = []
        for text in df[text_columns]:
            if isinstance(text, str) and text.strip():
                try:
                    features = apply_linguistic_features(text)
                    features_list.append(features)
                except ValueError as e:
                    log_with_timestamp(f"Skipping row due to error: {e}")
                    features_list.append({})
            else:
                features_list.append({})
        
        features_df = pd.DataFrame(features_list)
        df_out = pd.concat([df, features_df], axis=1)

    else:
        # Multiple text columns to process (e.g., answer and reasoning)
        df_out = df
        for col_name in text_columns:
            if col_name not in df.columns:
                log_with_timestamp(
                    f"Error: Text column '{col_name}' not found in the input file for dataset '{dataset_name}'."
                )
                continue

            log_with_timestamp(f"Applying linguistic features to column '{col_name}'...")
            features_list = []
            for text in df[col_name]:
                if isinstance(text, str) and text.strip():
                    try:
                        features = apply_linguistic_features(text)
                        features_list.append(features)
                    except ValueError as e:
                        log_with_timestamp(f"Skipping row due to error: {e}")
                        features_list.append({})
                else:
                    features_list.append({})
            
            features_df = pd.DataFrame(features_list)
            # Add prefix to feature column names to distinguish them
            features_df = features_df.add_prefix(f"{col_name}_")
            df_out = pd.concat([df_out, features_df], axis=1)

    # Prepare output file path
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{base_filename}_features.tsv"
    output_path = os.path.join(output_dir, output_filename)

    log_with_timestamp(f"Writing output to {output_path}")
    df_out.to_csv(output_path, sep="\t", index=False)

    log_with_timestamp("Processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process AI-generated text datasets for linguistic features."
    )
    parser.add_argument(
        "--input_file", required=True, help="Path to the input CSV/TSV file."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write the output TSV file."
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset to process (e.g., 'wildchat-1m').",
    )

    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.dataset_name)
