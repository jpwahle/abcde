#!/usr/bin/env bash

# Full TUSC pipeline with proper stage sequencing
# This script submits stage 1, waits for completion, then submits stage 2

set -euo pipefail

# Configuration
INPUT_FILE=${INPUT_FILE:-"/beegfs/wahle/datasets/tusc/tusc-city.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/beegfs/wahle/github/abcde/outputs_tusc"}
CHUNK_SIZE=${CHUNK_SIZE:-100000}
ARRAY_SIZE=${ARRAY_SIZE:-"0-127"}

echo "Starting TUSC full pipeline"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo "Array size: $ARRAY_SIZE"

# Export variables for the stage scripts
export INPUT_FILE
export OUTPUT_DIR
export CHUNK_SIZE

# Submit Stage 1
echo "Submitting Stage 1 job array..."
STAGE1_JOB=$(sbatch --array="$ARRAY_SIZE" --parsable run_tusc_stage1.sh)
echo "Stage 1 submitted with job ID: $STAGE1_JOB"

# Submit Stage 2 with dependency on Stage 1 completion
echo "Submitting Stage 2 job array (dependent on Stage 1)..."
STAGE2_JOB=$(sbatch --array="$ARRAY_SIZE" --dependency=afterok:$STAGE1_JOB --parsable run_tusc_stage2.sh)
echo "Stage 2 submitted with job ID: $STAGE2_JOB"

echo ""
echo "Pipeline submitted successfully!"
echo "Stage 1 job ID: $STAGE1_JOB"
echo "Stage 2 job ID: $STAGE2_JOB"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  tail -f logs/tusc_stage1.${STAGE1_JOB}_*.out"
echo "  tail -f logs/tusc_stage2.${STAGE2_JOB}_*.out"
echo ""
echo "Final output files will be:"
echo "  $OUTPUT_DIR/city_users.tsv"
echo "  $OUTPUT_DIR/city_user_posts.tsv"
