#!/bin/bash
#SBATCH --job-name=tusc_processing
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/tusc_%j.out
#SBATCH --error=logs/tusc_%j.err

# TUSC Data Processing Pipeline with Dask Parallelization
# Usage: sbatch run_tusc_pipeline.sh
# Or: sbatch --export=INPUT_FILE=/path/to/file.parquet,SPLIT=city run_tusc_pipeline.sh
#
# This launcher job starts a Python script that spins up its own Dask
# cluster via dask-jobqueue/SLURMCluster (see --use_slurm flag).
# The *real* heavy lifting is handled by the dynamically allocated Dask workers.

# Default parameters (can be overridden via --export)
INPUT_FILE=${INPUT_FILE:-"/shared/tusc/tusc-country.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/beegfs/wahle/github/abcde/outputs_tusc"}
SPLIT=${SPLIT:-"country"}
CHUNK_SIZE=10000
N_WORKERS=256
MEM_PER_WORKER=4GB

echo "Starting TUSC processing pipeline with Dask parallelization"
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

# Build the command arguments
ARGS=(
    --input_file "$INPUT_FILE"
    --output_dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --chunk_size "$CHUNK_SIZE"
    --n_workers "$N_WORKERS"
    --memory_per_worker "$MEM_PER_WORKER"
    --use_slurm
)

# Add test mode arguments if enabled
if [ "$TEST_MODE" = "true" ]; then
    ARGS+=(--test_mode)
    if [ -n "${TEST_SAMPLES:-}" ]; then
        ARGS+=(--test_samples "$TEST_SAMPLES")
    fi
fi

# Run processing with Dask parallelization
echo "Processing TUSC data with Dask..."
uv run python process_tusc_data.py "${ARGS[@]}"

echo "[$(date)] TUSC processing pipeline completed ✔"