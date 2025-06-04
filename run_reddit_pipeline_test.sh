#!/usr/bin/env bash
#SBATCH --job-name=reddit_pipeline_test
#SBATCH --output=logs/reddit_pipeline_test.%j.out
#SBATCH --error=logs/reddit_pipeline_test.%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

set -euxo pipefail

INPUT_DIR=/path/to/sample/jsonl
OUTPUT_DIR=/path/to/output/reddit_test

srun python process_reddit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK"