#!/usr/bin/env bash
#SBATCH --job-name=extract_reddit_test_data
#SBATCH --output=logs/extract_reddit_test_data.%A_%a.out
#SBATCH --error=logs/extract_reddit_test_data.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-155   

set -euo pipefail

# 1) Source and destination directories
SRC_DIR="/beegfs/wahle/datasets/reddit-2010-2022/extracted"
DST_DIR="/beegfs/wahle/github/abcde/data/test/reddit"

# Ensure destination exists (mkdir -p is safe to call from multiple array jobs)
mkdir -p "$DST_DIR"

# 2) Build an array of all RS_YYYY-MM files (globs are sorted lexicographically by default)
files=( "$SRC_DIR"/RS_????-?? )

# 3) Select the file corresponding to this array task
file="${files[$SLURM_ARRAY_TASK_ID]}"
base="$(basename "$file")"

# 4) Define the same Perl-compatible regex (escaped for single-quoted bash string)
regex='\bI(?:\s+am|'\''m)\s+([1-9]\d?)(?=\s*(?:years?(?:\s+old|-old)?|yo|yrs?)?\b)(?!\s*[%\$°#@&*+=<>()[\]{}|\\~`^_])'

# 5) Grep up to 100 matches and write to the destination
grep -P -m 100 "$regex" "$file" > "$DST_DIR/${base}_test"