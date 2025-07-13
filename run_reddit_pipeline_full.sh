#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

# Submit the first job
jobid1=$(STAGES=1 sbatch --parsable run_reddit_pipeline.sh)

echo "Submitted STAGES=1 job with Job ID: $jobid1"

# Submit the second job with dependency on successful completion of the first
jobid2=$(STAGES=2 sbatch --dependency=afterok:$jobid1 --parsable run_reddit_pipeline.sh)

echo "Submitted STAGES=2 job with dependency on $jobid1. Job ID: $jobid2"

echo ""
echo "============================================"
echo "Full pipeline submitted successfully!"
echo "============================================"
echo "Stage 1 Job ID: $jobid1"
echo "Stage 2 Job ID: $jobid2 (depends on $jobid1)"
echo "Monitor 1 Job ID: $monitor1_jobid (starts with Stage 1, logs: logs/monitor_stage1_${jobid1}.log)"
echo "Monitor 2 Job ID: $monitor2_jobid (starts with Stage 2, logs: logs/monitor_stage2_${jobid2}.log)"
echo ""
echo "To check job status: squeue --user=\$USER"
echo "To check monitor logs: tail -f logs/monitor_stage1_${jobid1}.log"
echo "To cancel jobs: scancel $jobid1 $jobid2"
echo "============================================"
