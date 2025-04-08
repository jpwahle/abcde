#!/bin/bash

# Initialize total line counter
total_lines=0

# Loop through all files excluding .zst files
for file in *; do
    # Skip files with .zst extension
    if [[ ! "$file" =~ \.zst$ && -f "$file" ]]; then
        echo "Counting lines in $file..."
        line_count=$(wc -l < "$file")
        echo "$file: $line_count lines"
        total_lines=$((total_lines + line_count))
    fi
done

# Display the total number of lines
if [[ $total_lines -eq 0 ]]; then
    echo "No files without .zst extension found."
else
    echo "Total lines across all files: $total_lines"
fi