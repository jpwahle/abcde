#!/bin/bash
#SBATCH --job-name=reddit-pipeline-%a
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- core resources for the *launcher* job ---------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=72:00:00

# ------------------------ CONFIGURATION -----------------------------
# Adjust paths
INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted              # directory containing RS_*.jsonl
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit
N_WORKERS=128                               # how many Dask workers to start
MEM_PER_WORKER=16GB                          # memory per worker for SLURMCluster
# --------------------------------------------------------------------

mkdir -p $OUTPUT_DIR logs

set -euo pipefail

SELF_USERS_CSV=$OUTPUT_DIR/reddit_users.tsv
ALL_POSTS_CSV=$OUTPUT_DIR/reddit_users_posts.tsv

# --------------------------------------------------------------------
# Stage 1 – detect self-identified users
uv run python identify_self_users_reddit.py \
  --input_dir "$INPUT_DIR" \
  --output_csv "$SELF_USERS_CSV" \
  --split "text" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm \
  --output_tsv

# --------------------------------------------------------------------
# Stage 2 – collect all posts of those users & annotate features
uv run python collect_user_posts_reddit.py \
  --input_dir "$INPUT_DIR" \
  --self_identified_csv "$SELF_USERS_CSV" \
  --output_csv "$ALL_POSTS_CSV" \
  --split "text" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm \
  --output_tsv

echo "[$(date)] Pipeline finished ✔" 