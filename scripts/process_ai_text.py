#!/usr/bin/env python3
"""
Process AI-generated text datasets to compute linguistic features.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from abcde import apply_linguistic_features, print_banner


def _detect_text_columns(
    df: pd.DataFrame, dataset_name: str, dataset_ai_map
) -> tuple[list[str], list[str]]:
    """Return (ai_columns, human_columns) present in df for the given dataset.

    Heuristic:
    - Start from configured AI columns per dataset (string or tuple in map)
    - Add any additional common AI column names that exist
    - Detect common human/prompt column names if present
    """
    # Seed AI columns from configured map
    ai_cols: list[str] = []
    if dataset_name in dataset_ai_map:
        cfg = dataset_ai_map[dataset_name]
        if isinstance(cfg, str):
            ai_cols = [cfg] if cfg in df.columns else []
        else:
            ai_cols = [c for c in cfg if c in df.columns]

    # Known column name candidates
    ai_candidates = [
        "ai_text",
        "response",
        "assistant",
        "answer",
        "completion",
        "output",
        "model_answer",
        "model_reasoning",
        "generated_text",
        "assistant_response",
    ]
    human_candidates = [
        "prompt",
        "instruction",
        "input",
        "question",
        "human",
        "user_input",
        "user_prompt",
        "query",
    ]

    # Add present AI candidates not already included
    ai_cols_set = set(ai_cols)
    for col in ai_candidates:
        if col in df.columns and col not in ai_cols_set:
            ai_cols.append(col)
            ai_cols_set.add(col)

    # Detect human/prompt columns present
    human_cols = [c for c in human_candidates if c in df.columns]

    # Fallback: for datasets with a generic 'text' column and no explicit human columns
    if not ai_cols and "text" in df.columns:
        ai_cols = ["text"]

    return ai_cols, human_cols


def _build_long_text_df(
    df: pd.DataFrame, ai_cols: list[str], human_cols: list[str]
) -> pd.DataFrame:
    """Create a long-format DataFrame with columns ['text', 'text_type'].

    - Emits one row per non-empty string in provided AI/Human columns
    - text_type is "AI" for ai_cols, "Human" for human_cols
    """
    records: list[dict] = []
    # AI texts
    for col in ai_cols:
        if col not in df.columns:
            continue
        for val in df[col]:
            if isinstance(val, str) and val.strip():
                records.append({"text": val, "text_type": "AI"})

    # Human/prompt texts
    for col in human_cols:
        if col not in df.columns:
            continue
        for val in df[col]:
            if isinstance(val, str) and val.strip():
                records.append({"text": val, "text_type": "Human"})

    return (
        pd.DataFrame.from_records(records, columns=["text", "text_type"])
        if records
        else pd.DataFrame(columns=["text", "text_type"])
    )


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

    # Mapping of dataset names to AI text columns. Some datasets may also contain
    # additional prompt/human columns which will be detected heuristically.
    # For datasets with multiple AI text fields (e.g., answer and reasoning),
    # a tuple of column names is provided and all will be treated as AI text.
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
            df = pd.read_csv(
                input_file, sep=sep, low_memory=False, encoding="windows-1252"
            )
        else:
            df = pd.read_csv(input_file, sep=sep, low_memory=False)
    except FileNotFoundError:
        log_with_timestamp(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        log_with_timestamp(f"Error reading input file: {e}")
        return

    # Detect AI and Human text columns present in the input
    ai_cols, human_cols = _detect_text_columns(df, dataset_name, dataset_column_map)
    if not ai_cols and not human_cols:
        log_with_timestamp(
            "Error: No recognizable text columns found (AI or Human/prompt)."
        )
        return

    # Build long-format DataFrame with a single 'text' column and 'text_type'
    df_long = _build_long_text_df(df, ai_cols, human_cols)
    log_with_timestamp(
        f"Prepared long-format text with {len(df_long)} rows from AI cols={ai_cols} and Human cols={human_cols}"
    )

    # Apply linguistic features on the unified 'text' column
    log_with_timestamp("Applying linguistic features to unified 'text' column…")
    features_list = []
    for text in df_long["text"]:
        if isinstance(text, str) and text.strip():
            try:
                features_list.append(apply_linguistic_features(text))
            except ValueError as e:
                log_with_timestamp(f"Skipping row due to error: {e}")
                features_list.append({})
        else:
            features_list.append({})
    features_df = pd.DataFrame(features_list)
    df_out = pd.concat([df_long, features_df], axis=1)

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
