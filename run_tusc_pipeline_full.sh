#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

# Export environment variables for sbatch
export INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-city.parquet"
jobid1=$(sbatch --parsable --export=INPUT_FILE run_tusc_pipeline.sh)

export INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-country.parquet"
jobid2=$(sbatch --parsable --export=INPUT_FILE run_tusc_pipeline.sh)

echo ""
echo "Full pipeline submitted successfully!"
echo "============================================"
echo "Stage 1 Job ID: $jobid1"
echo "Stage 2 Job ID: $jobid2"
echo ""
echo "To check job status: squeue --user=\$USER"
echo "To cancel jobs: scancel $jobid1 $jobid2"
echo "============================================"
