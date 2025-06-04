#!/usr/bin/env bash
#SBATCH --job-name=tusc_pipeline
#SBATCH --output=logs/tusc_pipeline.%j.out
#SBATCH --error=logs/tusc_pipeline.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64GB

set -euxo pipefail

INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-country.parquet"
OUTPUT_DIR="/beegfs/wahle/github/abcde/outputs_tusc"

srun python process_tusc.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"