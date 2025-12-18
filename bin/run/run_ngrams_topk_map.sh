#!/usr/bin/env bash
#SBATCH --job-name=ngrams_map
#SBATCH --output=logs/ngrams_map.%A_%a.out
#SBATCH --error=logs/ngrams_map.%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-127

set -euxo pipefail

INPUT_DIR=${INPUT_DIR:-/beegfs/wahle/datasets/google-books-ngrams/extracted}
OUT_DIR=${OUT_DIR:-/beegfs/wahle/github/abcde/outputs_google_ngrams/topk_map}
PATTERN=${PATTERN:-*5gram*}
FLUSH_EVERY=${FLUSH_EVERY:-5000000}

mkdir -p "$OUT_DIR" logs

uv run python /beegfs/wahle/github/abcde/ngrams_topk_map.py \
  --input_dir "$INPUT_DIR" \
  --pattern "$PATTERN" \
  --output_dir "$OUT_DIR" \
  --task_id "${SLURM_ARRAY_TASK_ID:-0}" \
  --total_tasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --flush_every "$FLUSH_EVERY"


