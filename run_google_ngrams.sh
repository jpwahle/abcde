#!/bin/bash
#SBATCH --job-name=ngrams_distributed
#SBATCH --output=logs/ngrams_dist_%A_%a.out
#SBATCH --error=logs/ngrams_dist_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-256

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
INPUT_DIR="$HOME/datasets/google-books-ngrams/extracted"
OUTPUT_DIR="outputs_google_ngrams"
CHUNK_SIZE=100000  # Lines per chunk
PATTERN="*5gram*"  # Process 5-grams

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM Array Task Count: $SLURM_ARRAY_TASK_COUNT"
echo "Start time: $(date)"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE lines"
echo "Pattern: $PATTERN"

# Process the assigned chunk
# Use the fast version with byte-offset indexing
uv run python process_ngrams.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pattern "$PATTERN" \
    --chunk_size $CHUNK_SIZE \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_workers $SLURM_ARRAY_TASK_COUNT \
    --build_indexes

echo "End time: $(date)"
