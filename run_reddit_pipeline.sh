#!/usr/bin/env bash
#SBATCH --job-name=reddit_pipeline
#SBATCH --output=logs/reddit_pipeline.%j.out
#SBATCH --error=logs/reddit_pipeline.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=128

set -euxo pipefail

INPUT_DIR=/path/to/reddit/jsonl
OUTPUT_DIR=/path/to/output/reddit

srun python process_reddit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK"