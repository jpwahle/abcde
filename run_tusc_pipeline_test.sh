#!/bin/bash
#SBATCH --job-name=tusc_test
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/tusc_test_%j.out
#SBATCH --error=logs/tusc_test_%j.err

# TUSC Data Processing Pipeline - Test Mode with Dask Parallelization
# Usage: sbatch run_tusc_pipeline_test.sh
# Or: sbatch --export=INPUT_FILE=/path/to/file.parquet,SPLIT=city run_tusc_pipeline_test.sh
#
# Quick-validation run: small subset of TUSC data to ensure everything works
# before committing large resources on the full dataset.
# This launcher job starts a Python script that spins up its own Dask
# cluster via dask-jobqueue/SLURMCluster (see --use_slurm flag).

# Default parameters (can be overridden via --export)
INPUT_FILE=${INPUT_FILE:-"/beegfs/wahle/datasets/tusc/tusc-country.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/beegfs/wahle/github/abcde/outputs_tusc_test"}
SPLIT=${SPLIT:-"country"}
CHUNK_SIZE=1000
N_WORKERS=16
MEM_PER_WORKER=4GB
TEST_SAMPLES=16000

echo "Starting TUSC processing pipeline - TEST MODE with Dask"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Split type: $SPLIT"
echo "Chunk size: $CHUNK_SIZE"
echo "Number of workers: $N_WORKERS"
echo "Memory per worker: $MEM_PER_WORKER"
echo "Test samples: $TEST_SAMPLES"
echo "----------------------------------------"

# Create output and log directories
mkdir -p $OUTPUT_DIR logs

# Set pipefail to catch any command failures
set -euo pipefail

# Run processing with Dask parallelization in test mode
echo "Processing TUSC data in test mode with Dask..."
uv run python process_tusc_data.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --chunk_size "$CHUNK_SIZE" \
    --n_workers "$N_WORKERS" \
    --memory_per_worker "$MEM_PER_WORKER" \
    --use_slurm \
    --test_mode \
    --test_samples "$TEST_SAMPLES"

echo "[$(date)] TUSC test processing pipeline completed ✔"