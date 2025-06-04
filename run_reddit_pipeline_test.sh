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
# Choose a single-year directory (uncomment):
#     INPUT_DIR=/shared/reddit/2010


# Pick split to test (text only is usually enough):
SPLIT=text

INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted/RS_2010-01
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit_test

# Number of Dask workers (keep low for testing)
N_WORKERS=16
MEM_PER_WORKER=4GB

set -euo pipefail
mkdir -p $OUTPUT_DIR logs

SELF_USERS_TSV=$OUTPUT_DIR/reddit_users_test.tsv
ALL_POSTS_TSV=$OUTPUT_DIR/reddit_users_posts_test.tsv

# ------------------ Stage 1 ---------------------------------------------
uv run python identify_self_users.py \
  --input_dir "$INPUT_DIR" \
  --output_csv "$SELF_USERS_TSV" \
  --split "$SPLIT" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm \
  --output_tsv

# ------------------ Stage 2 ---------------------------------------------
uv run python collect_user_posts.py \
  --input_dir "$INPUT_DIR" \
  --self_identified_csv "$SELF_USERS_TSV" \
  --output_csv "$ALL_POSTS_TSV" \
  --split "$SPLIT" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm \
  --output_tsv

echo "[$(date)] Test pipeline finished ✔" 