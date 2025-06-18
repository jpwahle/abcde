#!/usr/bin/env bash
#SBATCH --job-name=tusc_pipeline_full
#SBATCH --output=logs/tusc_pipeline_full.%j.out
#SBATCH --error=logs/tusc_pipeline_full.%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

set -euxo pipefail

INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-city.parquet"
jobid1=$(sbatch --parsable run_tusc_pipeline.sh)

INPUT_FILE="/beegfs/wahle/datasets/tusc/tusc-country.parquet"
jobid2=$(sbatch --parsable run_tusc_pipeline.sh)

echo ""
echo "Full pipeline submitted successfully!"
echo "============================================"
echo "Stage 1 Job ID: $jobid1"
echo "Stage 2 Job ID: $jobid2"
echo ""
echo "To check job status: squeue --user=\$USER"
echo "To cancel jobs: scancel $jobid1 $jobid2"
echo "============================================"
