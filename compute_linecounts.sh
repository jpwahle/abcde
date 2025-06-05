#!/usr/bin/env bash
#SBATCH --job-name=compute_linecounts
#SBATCH --output=logs/compute_linecounts.%A_%a.out
#SBATCH --error=logs/compute_linecounts.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-155   

# Compute line counts for reddit dataset files in parallel using SLURM array.
# Generates a file with suffix _linecount for each dataset file.
set -euo pipefail

DATA_DIR="/beegfs/wahle/datasets/reddit-2010-2022/extracted/"
OUTPUT_DIR="/beegfs/wahle/github/abcde/reddit_linecounts"

mkdir -p "$OUTPUT_DIR"

# Create array of files
mapfile -t files < <(find "$DATA_DIR" -type f \( -name 'RS_*' -o -name 'RC_*' \) | sort)

# Get the file for this array task
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#files[@]}" ]; then
    echo "Array task ID ${SLURM_ARRAY_TASK_ID} is beyond the number of files (${#files[@]}). Exiting."
    exit 0
fi

file="${files[$SLURM_ARRAY_TASK_ID]}"
filename=$(basename "$file")

echo "Processing file: $file"
# Process the file
count=$(wc -l < "$file")
echo "$count" > "${OUTPUT_DIR}/${filename}_linecount"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUTPUT_DIR}/${filename}_linecount written with $count lines"
