#!/bin/bash
#SBATCH --job-name=reddit-pipeline-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Quick-validation run: small subset of Reddit dump (e.g. a single year or 1% sample).
# You can safely start this on the login / short queue to ensure everything works
# before committing large resources on the full dump.
#
# Resources for the *launcher* job – the heavy work happens in SLURMCluster workers.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00

# ------------------ CONFIGURATION ----------------------------------------
# Choose ONE of the following two possibilities:
# 1) Use a single-year directory (uncomment):
#     INPUT_DIR=/shared/reddit/2010
# 2) Use a random 1% sample generated via sample.py (uncomment both SAMPLE_* lines):
#     SAMPLE_INPUT_DIR=/shared/reddit/full
#     SAMPLE_OUTPUT_DIR=/scratch/$USER/reddit_sample_1pct
# ------------------------------------------------------------------------

# Pick split to test (text only is usually enough):
SPLIT=text

INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted/RS_2015-01

# Number of Dask workers (keep low for testing)
N_WORKERS=64
MEM_PER_WORKER=4GB

set -euo pipefail
mkdir -p logs

OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/outputs_test}
mkdir -p "$OUTPUT_DIR"

SELF_USERS_CSV=$OUTPUT_DIR/self_users_test.csv
ALL_POSTS_CSV=$OUTPUT_DIR/self_users_posts_test.csv

# ------------------ Stage 1 ---------------------------------------------
uv run python identify_self_users.py \
  --input_dir "$INPUT_DIR" \
  --output_csv "$SELF_USERS_CSV" \
  --split "$SPLIT" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm

# ------------------ Stage 2 ---------------------------------------------
uv run python collect_user_posts.py \
  --input_dir "$INPUT_DIR" \
  --self_identified_csv "$SELF_USERS_CSV" \
  --output_csv "$ALL_POSTS_CSV" \
  --split "$SPLIT" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm

echo "[$(date)] Test pipeline finished ✔" 