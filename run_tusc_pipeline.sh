#!/bin/bash
#SBATCH --job-name=tusc_processing
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/tusc_%j.out
#SBATCH --error=logs/tusc_%j.err

# TUSC Data Processing Pipeline
# Usage: sbatch run_tusc_pipeline.sh
# Or: sbatch --export=INPUT_FILE=/path/to/file.parquet,SPLIT=city run_tusc_pipeline.sh

# Default parameters (can be overridden via --export)
INPUT_FILE=${INPUT_FILE:-"/shared/tusc/tusc-country.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
SPLIT=${SPLIT:-"country"}
BATCH_SIZE=${BATCH_SIZE:-10000}
TEST_MODE=${TEST_MODE:-false}
TEST_SAMPLES=${TEST_SAMPLES:-10000}

echo "Starting TUSC processing pipeline"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Split type: $SPLIT"
echo "Batch size: $BATCH_SIZE"
echo "Test mode: $TEST_MODE"
if [ "$TEST_MODE" = "true" ]; then
    echo "Test samples: $TEST_SAMPLES"
fi
echo "----------------------------------------"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Activate environment and run processing
echo "Processing TUSC data..."
if [ "$TEST_MODE" = "true" ]; then
    uv run python process_tusc_data.py \
        --input_file "$INPUT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --split "$SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --test_mode \
        --test_samples "$TEST_SAMPLES"
else
    uv run python process_tusc_data.py \
        --input_file "$INPUT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --split "$SPLIT" \
        --batch_size "$BATCH_SIZE"
fi

echo "TUSC processing pipeline completed"