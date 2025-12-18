#!/usr/bin/env bash
#SBATCH --job-name=extract_tusc_test_data
#SBATCH --output=logs/extract_tusc_test_data.%A_%a.out
#SBATCH --error=logs/extract_tusc_test_data.%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-1

set -euo pipefail

# 1) Source and destination directories
SRC_DIR="/beegfs/wahle/datasets/tusc"
DST_DIR="/beegfs/wahle/github/abcde/data/test/tusc"
mkdir -p "$DST_DIR"

# 2) Array of the two Parquet files
files=(
  "$SRC_DIR/tusc-city.parquet"
  "$SRC_DIR/tusc-country.parquet"
)

# 3) Pick the file for this array task
file="${files[$SLURM_ARRAY_TASK_ID]}"
base="$(basename "$file" .parquet)"
out_parquet="$DST_DIR/${base}_test.parquet"

# 4) Inline Python: use PyArrow to iterate in batches and stop after first 100 matches
uv run python - "$file" "$out_parquet" << 'EOF'
import re
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

infile = sys.argv[1]
outfile = sys.argv[2]

# Compile the same regex:
pattern = re.compile(
    r"\bI(?:\s+am|'m)\s+(?:[1-9]\d?)"
    r"(?=\s*(?:years?(?:\s+old|-old)?|yo|yrs?)?\b)"
    r"(?!\s*[%\$°#@&*+=<>()[\]{}|\\~`^_])",
    re.I
)

print(f"Loading {infile}...")

# Open ParquetFile for batch iteration:
pqfile = pq.ParquetFile(infile)

print(f"Loaded {infile}...")

matched_chunks = []
count = 0

print(f"Iterating over {infile}...")

# Iterate over row-groups or batches:
for batch in pqfile.iter_batches(batch_size=10000):
    # Convert RecordBatch to a Pandas DataFrame
    table = pa.Table.from_batches([batch], schema=batch.schema)
    df = table.to_pandas()

    # Filter rows where the 'text' column matches the regex
    mask = df['Tweet'].str.contains(pattern, na=False)
    if mask.any():
        print(f"Found {len(df[mask])} matches...")
        matches = df[mask]
        matched_chunks.append(matches)
        count += len(matches)
        print(f"Count: {count}")
        if count >= 100:
            break

# Concatenate and take exactly the first 100 (or fewer if not enough):
if matched_chunks:
    result_df = pd.concat(matched_chunks).head(100)
else:
    # If no matches at all, create an empty DataFrame with the same schema:
    # (read a tiny slice just to get columns)
    # Or assume schema from first batch:
    first_batch = next(pqfile.iter_batches(batch_size=1))
    cols = list(pa.Table.from_batches([first_batch], schema=first_batch.schema).to_pandas().columns)
    result_df = pd.DataFrame(columns=cols)

# Write out to Parquet
result_df.to_parquet(outfile)
EOF
