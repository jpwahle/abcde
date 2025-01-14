#!/bin/bash

set -euo pipefail

# Determine the directory the script lives in so it works regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="$SCRIPT_DIR/google-books-ngram-urls.html"
OUTPUT_FILE="$SCRIPT_DIR/google-books-ngram-urls.txt"

# Verify that the source HTML file exists
if [[ ! -f "$HTML_FILE" ]]; then
  echo "Error: $HTML_FILE not found." >&2
  exit 1
fi

# Extract all href values and save them line-by-line to the output file
# 1. grep -oE finds every occurrence of href="..." in the html
# 2. sed removes the leading href=" and trailing " leaving only the URL
grep -oE 'href="[^"]+"' "$HTML_FILE" | sed -E 's/href="([^"]+)"/\1/' > "$OUTPUT_FILE"

echo "Wrote $(wc -l < "$OUTPUT_FILE" | tr -d ' ') URLs to $OUTPUT_FILE"

