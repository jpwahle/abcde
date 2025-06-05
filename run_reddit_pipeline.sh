#!/usr/bin/env bash
#SBATCH --job-name=reddit_pipeline
#SBATCH --output=logs/reddit_pipeline.%A_%a.out
#SBATCH --error=logs/reddit_pipeline.%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-511

set -euxo pipefail

INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit
LINECOUNT_DIR=/beegfs/wahle/github/abcde/reddit_linecounts
# number of lines per chunk for large JSONL files (0 = process whole file at once)
CHUNK_SIZE=${CHUNK_SIZE:-100000}
STAGES="1"

uv run python process_reddit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --chunk_size "$CHUNK_SIZE" \
    --stages "$STAGES" \
    --task_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --total_tasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --linecount_dir "$LINECOUNT_DIR"
