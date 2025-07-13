#!/usr/bin/env bash
#SBATCH --job-name=reddit_pii_detection
#SBATCH --output=logs/reddit_pii_detection.%A_%a.out
#SBATCH --error=logs/reddit_pii_detection.%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --array=0-155

set -euxo pipefail

INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2022/extracted/
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit_pii
# number of lines per chunk for large JSONL files (0 = process whole file at once)
CHUNK_SIZE=${CHUNK_SIZE:-10000}

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export PYTHONUNBUFFERED=1

echo "Starting PII detection job array task ${SLURM_ARRAY_TASK_ID} at $(date)"

uv run python process_reddit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --chunk_size "$CHUNK_SIZE" \
    --task_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --total_tasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --pii

echo "Completed PII detection job array task ${SLURM_ARRAY_TASK_ID} at $(date)"