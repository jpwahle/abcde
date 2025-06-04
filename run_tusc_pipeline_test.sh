#!/usr/bin/env bash
#SBATCH --job-name=tusc_pipeline_test
#SBATCH --output=logs/tusc_pipeline_test.%j.out
#SBATCH --error=logs/tusc_pipeline_test.%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8GB

set -euxo pipefail

INPUT_FILE=/path/to/sample_tusc.parquet
OUTPUT_DIR=/path/to/output/tusc_test

srun python process_tusc.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"