#!/usr/bin/env bash
#SBATCH --job-name=tusc_pipeline
#SBATCH --output=logs/tusc_pipeline.%j.out
#SBATCH --error=logs/tusc_pipeline.%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

set -euxo pipefail

INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-city.parquet"
OUTPUT_DIR="/beegfs/wahle/github/abcde/outputs_tusc"
STAGES="2"

export PYTHONUNBUFFERED=1

uv run python process_tusc.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --stages "$STAGES"