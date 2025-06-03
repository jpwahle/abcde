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
#SBATCH --array=0-1

# --------------------------------------------------------------------
# Array index 0 → text   | 1 → multimodal
# Each array task starts a Python script that spins up its own Dask
# cluster via dask-jobqueue/SLURMCluster (see --use_slurm flag).
# This means the *real* heavy lifting is handled by the dynamically
# allocated Dask workers, not by the launcher job itself.

# ------------------------ CONFIGURATION -----------------------------
# Adjust paths
INPUT_DIR=/beegfs/wahle/datasets/reddit-2010-2020/extracted              # directory containing RS_*.jsonl
OUTPUT_DIR=/beegfs/wahle/github/abcde/outputs_reddit
N_WORKERS=128                               # how many Dask workers to start
MEM_PER_WORKER=4GB                          # memory per worker for SLURMCluster
# --------------------------------------------------------------------

# Map array index → split value
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    SPLIT=text
else
    SPLIT=multimodal
fi

mkdir -p $OUTPUT_DIR logs

set -euo pipefail

SELF_USERS_CSV=$OUTPUT_DIR/self_users_${SPLIT}.csv
ALL_POSTS_CSV=$OUTPUT_DIR/self_users_posts_${SPLIT}.csv

# --------------------------------------------------------------------
# Stage 1 – detect self-identified users
uv run python identify_self_users.py \
  --input_dir "$INPUT_DIR" \
  --output_csv "$SELF_USERS_CSV" \
  --split "$SPLIT" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm

# --------------------------------------------------------------------
# Stage 2 – collect all posts of those users & annotate features
uv run python collect_user_posts.py \
  --input_dir "$INPUT_DIR" \
  --self_identified_csv "$SELF_USERS_CSV" \
  --output_csv "$ALL_POSTS_CSV" \
  --split "$SPLIT" \
  --n_workers $N_WORKERS \
  --memory_per_worker $MEM_PER_WORKER \
  --use_slurm

echo "[$(date)] Pipeline for split='$SPLIT' finished ✔" 