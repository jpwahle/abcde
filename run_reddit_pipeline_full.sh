#!/bin/bash

# Submit the first job
jobid1=$(STAGES=1 sbatch --parsable run_reddit_pipeline.sh)

echo "Submitted STAGES=1 job with Job ID: $jobid1"

# Submit the second job with dependency on successful completion of the first
jobid2=$(STAGES=2 sbatch --dependency=afterok:$jobid1 --parsable run_reddit_pipeline.sh)

echo "Submitted STAGES=2 job with dependency on $jobid1. Job ID: $jobid2"
