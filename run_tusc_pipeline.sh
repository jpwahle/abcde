#!/bin/bash
#SBATCH --job-name=tusc_pipeline
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/tusc_%j.out
#SBATCH --error=logs/tusc_%j.err

# TUSC Data Processing Pipeline (Two-Stage Approach)
# Stage 1: Identify self-identified users
# Stage 2: Collect all posts from those users and compute linguistic features
#
# Usage: sbatch run_tusc_pipeline.sh
# Or: sbatch --export=INPUT_FILE=/path/to/file.parquet,SPLIT=city run_tusc_pipeline.sh
#
# This launcher job starts Python scripts that spin up their own Dask
# clusters via dask-jobqueue/SLURMCluster (see --use_slurm flag).
# The *real* heavy lifting is handled by the dynamically allocated Dask workers.

# Default parameters (can be overridden via --export)
INPUT_FILE=${INPUT_FILE:-"/shared/tusc/tusc-country.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/beegfs/wahle/github/abcde/outputs_tusc"}
SPLIT=${SPLIT:-"country"}
CHUNK_SIZE=10000
N_WORKERS=256
MEM_PER_WORKER=4GB

echo "Starting TUSC processing pipeline (Two-Stage Approach)"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Split type: $SPLIT"
echo "Chunk size: $CHUNK_SIZE"
echo "Number of workers: $N_WORKERS"
echo "Memory per worker: $MEM_PER_WORKER"
echo "Test mode: $TEST_MODE"
if [ "$TEST_MODE" = "true" ]; then
    echo "Test samples: $TEST_SAMPLES"
fi
echo "----------------------------------------"

# Create output and log directories
mkdir -p $OUTPUT_DIR logs

# Set pipefail to catch any command failures
set -euo pipefail

# Generate output filenames based on input filename
INPUT_NAME=$(basename "$INPUT_FILE" .parquet)
SELF_ID_CSV="$OUTPUT_DIR/${INPUT_NAME}_self_users.csv"
FINAL_OUTPUT_CSV="$OUTPUT_DIR/${INPUT_NAME}_user_posts.csv"

if [ "$TEST_MODE" = "true" ]; then
    SELF_ID_CSV="$OUTPUT_DIR/${INPUT_NAME}_self_users_test.csv"
    FINAL_OUTPUT_CSV="$OUTPUT_DIR/${INPUT_NAME}_user_posts_test.csv"
fi

echo "Stage 1: Identifying self-identified users..."
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
)

# Add split if specified (will be auto-determined if not)
if [ -n "${SPLIT:-}" ]; then
    STAGE1_ARGS+=(--split "$SPLIT")
fi

# Add test mode arguments if enabled
if [ "$TEST_MODE" = "true" ]; then
    STAGE1_ARGS+=(--test_mode)
    if [ -n "${TEST_SAMPLES:-}" ]; then
        STAGE1_ARGS+=(--test_samples "$TEST_SAMPLES")
    fi
fi

# Stage 1: Find self-identified users
uv run python identify_self_users.py "${STAGE1_ARGS[@]}"

echo "[$(date)] Stage 1 completed ✔"
echo "Stage 2: Collecting posts from self-identified users and computing linguistic features..."
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
)

# Add split if specified (will be auto-determined if not)
if [ -n "${SPLIT:-}" ]; then
    STAGE2_ARGS+=(--split "$SPLIT")
fi

# Add test mode arguments if enabled
if [ "$TEST_MODE" = "true" ]; then
    STAGE2_ARGS+=(--test_mode)
    if [ -n "${TEST_SAMPLES:-}" ]; then
        STAGE2_ARGS+=(--test_samples "$TEST_SAMPLES")
    fi
fi

# Stage 2: Collect user posts and compute features
uv run python collect_user_posts_tusc.py "${STAGE2_ARGS[@]}"

echo "[$(date)] TUSC processing pipeline completed ✔"
echo "Final output: $FINAL_OUTPUT_CSV"