#!/usr/bin/env bash
#SBATCH --job-name=tusc_pii
#SBATCH --output=logs/tusc_pii.%j.out
#SBATCH --error=logs/tusc_pii.%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

set -euxo pipefail

INPUT_FILE="${INPUT_FILE:-/beegfs/wahle/datasets/tusc/tusc-city.parquet}"
OUTPUT_DIR="/beegfs/wahle/github/abcde/outputs_tusc"
CHUNK_SIZE=${CHUNK_SIZE:-100000}

export PYTHONUNBUFFERED=1

uv run python process_tusc.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --stages none \
    --collect_pii_posts
