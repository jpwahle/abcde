#!/usr/bin/env bash
#SBATCH --job-name=reddit_pipeline
#SBATCH --output=logs/reddit_pipeline.%j.out
#SBATCH --error=logs/reddit_pipeline.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=256

set -euxo pipefail

INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit
# number of lines per chunk for large JSONL files (0 = process whole file at once)
CHUNK_SIZE=${CHUNK_SIZE:-100000}


uv run python process_reddit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --chunk_size "$CHUNK_SIZE"