#!/usr/bin/env bash
#SBATCH --job-name=reddit_pipeline_test
#SBATCH --output=logs/reddit_pipeline_test.%j.out
#SBATCH --error=logs/reddit_pipeline_test.%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

set -euxo pipefail

# sample paths for testing
INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted/RS_2010-01
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit_test
# number of lines per chunk for large JSONL files (0 = process whole file at once)
CHUNK_SIZE=${CHUNK_SIZE:-100000}

srun python process_reddit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --chunk_size "$CHUNK_SIZE"