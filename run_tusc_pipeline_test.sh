#!/bin/bash
#SBATCH --job-name=tusc_test
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/tusc_test_%j.out
#SBATCH --error=logs/tusc_test_%j.err

# TUSC Data Processing Pipeline - Test Mode (Two-Stage Approach)
# Stage 1: Identify self-identified users
# Stage 2: Collect all posts from those users and compute linguistic features
#
# Usage: sbatch run_tusc_pipeline_test.sh
# Or: sbatch --export=INPUT_FILE=/path/to/file.parquet,SPLIT=city run_tusc_pipeline_test.sh
#
# Quick-validation run: small subset of TUSC data to ensure everything works
# before committing large resources on the full dataset.
# This launcher job starts Python scripts that spin up their own Dask
# clusters via dask-jobqueue/SLURMCluster (see --use_slurm flag).

# Default parameters (can be overridden via --export)
INPUT_FILE=${INPUT_FILE:-"/beegfs/wahle/datasets/tusc/tusc-country.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/beegfs/wahle/github/abcde/outputs_tusc_test"}
CHUNK_SIZE=1000
N_WORKERS=128
MEM_PER_WORKER=4GB
TEST_SAMPLES=500000

echo "Starting TUSC processing pipeline - TEST MODE (Two-Stage Approach)"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo "Number of workers: $N_WORKERS"
echo "Memory per worker: $MEM_PER_WORKER"
echo "Test samples: $TEST_SAMPLES"
echo "----------------------------------------"

# Create output and log directories
mkdir -p $OUTPUT_DIR logs

# Set pipefail to catch any command failures
set -euo pipefail

# Generate output filenames based on input filename
INPUT_NAME=$(basename "$INPUT_FILE" .parquet)
SELF_ID_CSV="$OUTPUT_DIR/${INPUT_NAME}_self_users_test.tsv"
FINAL_OUTPUT_CSV="$OUTPUT_DIR/${INPUT_NAME}_user_posts_test.tsv"

echo "Stage 1: Identifying self-identified users (TEST MODE)..."
echo "Output will be written to: $SELF_ID_CSV"

# Build the stage 1 command arguments
STAGE1_ARGS=(
    --data_source tusc
    --input_file "$INPUT_FILE"
    --output_csv "$SELF_ID_CSV"
    --chunk_size "$CHUNK_SIZE"
    --n_workers "$N_WORKERS"
    --memory_per_worker "$MEM_PER_WORKER"
    --use_slurm
    --output_tsv
    --test_mode
    --test_samples "$TEST_SAMPLES"
)

# Stage 1: Find self-identified users
uv run python identify_self_users.py "${STAGE1_ARGS[@]}"

echo "[$(date)] Stage 1 completed ✔"

# Check if we found any self-identified users before proceeding
if [ ! -s "$SELF_ID_CSV" ] || [ $(wc -l < "$SELF_ID_CSV") -le 1 ]; then
    echo "WARNING: No self-identified users found in test data. Skipping Stage 2."
    echo "This is normal for small test samples - try increasing TEST_SAMPLES or use full pipeline."
    echo "[$(date)] TUSC test pipeline completed (Stage 1 only) ✔"
    exit 0
fi

echo "Stage 2: Collecting posts from self-identified users and computing linguistic features (TEST MODE)..."
echo "Output will be written to: $FINAL_OUTPUT_CSV"

# Build the stage 2 command arguments  
STAGE2_ARGS=(
    --input_file "$INPUT_FILE"
    --self_identified_csv "$SELF_ID_CSV"
    --output_csv "$FINAL_OUTPUT_CSV"
    --chunk_size "$CHUNK_SIZE"
    --n_workers "$N_WORKERS"
    --memory_per_worker "$MEM_PER_WORKER"
    --use_slurm
    --output_tsv
    --test_mode
    --test_samples "$TEST_SAMPLES"
)

# Stage 2: Collect user posts and compute features
uv run python collect_user_posts_tusc.py "${STAGE2_ARGS[@]}"

echo "[$(date)] TUSC test processing pipeline completed ✔"
echo "Final output: $FINAL_OUTPUT_CSV"

# Show some basic stats
echo "----------------------------------------"
echo "Pipeline Results Summary:"
echo "Self-identified users found: $(tail -n +2 "$SELF_ID_CSV" | wc -l)"
if [ -f "$FINAL_OUTPUT_CSV" ]; then
    echo "Total posts processed: $(tail -n +2 "$FINAL_OUTPUT_CSV" | wc -l)"
else
    echo "No final output file generated"
fi
echo "----------------------------------------"