#!/usr/bin/env bash
#SBATCH --job-name=ngrams_filter
#SBATCH --output=logs/ngrams_filter.%j.out
#SBATCH --error=logs/ngrams_filter.%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

set -euxo pipefail

# Point to the annotated outputs produced by run_google_ngrams
INPUT_DIR=${INPUT_DIR:-/beegfs/wahle/github/abcde/outputs_google_ngrams}
# Match only annotated chunk TSVs, exclude map-reduce artifacts like topk_map and topk_global_counts.tsv
PATTERN=${PATTERN:-googlebooks-eng-fiction-all-5gram-20120701-*.tsv}
TOPK_PATH=${TOPK_PATH:-/beegfs/wahle/github/abcde/outputs_google_ngrams/topk_global_counts.tsv}
OUT_PATH=${OUT_PATH:-/beegfs/wahle/github/abcde/outputs_google_ngrams/googlebooks-eng-fiction-top1M-5gram.tsv}

mkdir -p "$(dirname "$OUT_PATH")" logs

uv run python /beegfs/wahle/github/abcde/ngrams_topk_filter_rows.py \
  --input_dir "$INPUT_DIR" \
  --pattern "$PATTERN" \
  --topk_path "$TOPK_PATH" \
  --output_path "$OUT_PATH"


