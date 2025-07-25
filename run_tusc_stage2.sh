#!/usr/bin/env bash
#SBATCH --job-name=tusc_stage2
#SBATCH --output=logs/tusc_stage2.%A_%a.out
#SBATCH --error=logs/tusc_stage2.%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-127

set -euxo pipefail

# Configuration
INPUT_FILE=${INPUT_FILE:-"/beegfs/wahle/datasets/tusc/tusc-city.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/beegfs/wahle/github/abcde/outputs_tusc"}
CHUNK_SIZE=${CHUNK_SIZE:-100000}

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export PYTHONUNBUFFERED=1

echo "Starting TUSC Stage 2 job array task ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE"

# Run Stage 2 only
uv run python process_tusc.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --stages "2" \
    --task_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --total_tasks "${SLURM_ARRAY_TASK_COUNT:-1}"

echo "Completed TUSC Stage 2 job array task ${SLURM_ARRAY_TASK_ID} at $(date)" 