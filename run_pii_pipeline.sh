#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

echo "Submitting Reddit PII extraction job..."
jobid_reddit=$(sbatch --parsable run_reddit_pii_pipeline.sh)

echo "Submitting TUSC city PII extraction job..."
jobid_city=$(INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-city.parquet" sbatch --parsable run_tusc_pii_pipeline.sh)

echo "Submitting TUSC country PII extraction job..."
jobid_country=$(INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-country.parquet" sbatch --parsable run_tusc_pii_pipeline.sh)

echo ""
echo "PII-only pipeline submitted successfully!"
echo "============================================"
echo "Reddit Job ID: $jobid_reddit"
echo "TUSC City Job ID: $jobid_city"
echo "TUSC Country Job ID: $jobid_country"
echo ""
echo "To check job status: squeue --user=\$USER"
echo "To cancel jobs: scancel $jobid_reddit $jobid_city $jobid_country"
echo "============================================"
