#!/usr/bin/env bash
# Compute line counts for reddit dataset files.
# Usage: ./compute_linecounts.sh <data_dir>
# Generates a file with suffix _linecount for each dataset file.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATA_DIR>" >&2
    exit 1
fi

DATA_DIR="$1"

find "$DATA_DIR" -type f \( -name 'RS_*.jsonl' -o -name 'RC_*.jsonl' \) | while read -r file; do
    count=$(wc -l < "$file")
    echo "$count" > "${file}_linecount"
    echo "${file}_linecount written with $count lines"
done
