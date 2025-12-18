#!/usr/bin/env bash
#SBATCH --job-name=ngrams_reduce
#SBATCH --output=logs/ngrams_reduce.%j.out
#SBATCH --error=logs/ngrams_reduce.%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

set -euxo pipefail

MAP_DIR=${MAP_DIR:-/beegfs/wahle/github/abcde/outputs_google_ngrams/topk_map}
OUT_PATH=${OUT_PATH:-/beegfs/wahle/github/abcde/outputs_google_ngrams/topk_global_counts.tsv}
TOP_K=${TOP_K:-5000000}
MEM_UNIQUES=${MEM_UNIQUES:-50000000}

mkdir -p "$(dirname "$OUT_PATH")" logs

uv run python /beegfs/wahle/github/abcde/ngrams_topk_reduce.py \
  --map_dir "$MAP_DIR" \
  --output_path "$OUT_PATH" \
  --top_k "$TOP_K" \
  --memory_cap_unique "$MEM_UNIQUES"


