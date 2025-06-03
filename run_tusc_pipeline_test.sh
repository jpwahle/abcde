#!/bin/bash
#SBATCH --job-name=tusc_test
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/tusc_test_%j.out
#SBATCH --error=logs/tusc_test_%j.err

# TUSC Data Processing Pipeline - Test Mode
# Usage: sbatch run_tusc_pipeline_test.sh
# Or: sbatch --export=INPUT_FILE=/path/to/file.parquet,SPLIT=city run_tusc_pipeline_test.sh

# Default parameters (can be overridden via --export)
INPUT_FILE=${INPUT_FILE:-"/shared/tusc/tusc-country.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/test"}
SPLIT=${SPLIT:-"country"}
BATCH_SIZE=${BATCH_SIZE:-1000}
TEST_SAMPLES=${TEST_SAMPLES:-10000}

echo "Starting TUSC processing pipeline - TEST MODE"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Split type: $SPLIT"
echo "Batch size: $BATCH_SIZE"
echo "Test samples: $TEST_SAMPLES"
echo "----------------------------------------"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Activate environment and run processing in test mode
echo "Processing TUSC data in test mode..."
uv run python process_tusc_data.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --test_mode \
    --test_samples "$TEST_SAMPLES"

echo "TUSC test processing pipeline completed"