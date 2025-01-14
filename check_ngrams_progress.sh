#!/bin/bash

# Script to check progress of Google Books Ngram processing

INPUT_DIR="$HOME/datasets/google-books-ngrams/extracted"
OUTPUT_DIR="$HOME/datasets/google-books-ngrams/processed"

# Count total input files
TOTAL_FILES=$(find "$INPUT_DIR" -name "googlebooks-eng-fiction-all-*gram-*" -type f | wc -l)

# Count processed files
PROCESSED_FILES=$(find "$OUTPUT_DIR" -name "*_processed.tsv" -type f 2>/dev/null | wc -l)

# Calculate percentage
if [ $TOTAL_FILES -gt 0 ]; then
    PERCENTAGE=$(awk "BEGIN {printf \"%.2f\", ($PROCESSED_FILES/$TOTAL_FILES)*100}")
else
    PERCENTAGE=0
fi

echo "Google Books Ngram Processing Progress"
echo "======================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Total ngram files: $TOTAL_FILES"
echo "Processed files: $PROCESSED_FILES"
echo "Progress: ${PERCENTAGE}%"
echo ""

# Show file size statistics if there are processed files
if [ $PROCESSED_FILES -gt 0 ]; then
    echo "Output file statistics:"
    echo "----------------------"
    
    # Total size
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "Total output size: $TOTAL_SIZE"
    
    # Average file size
    AVG_SIZE=$(find "$OUTPUT_DIR" -name "*_processed.tsv" -type f -exec du -b {} + | \
               awk '{sum+=$1; count++} END {if(count>0) printf "%.2f MB", sum/count/1024/1024}')
    echo "Average file size: $AVG_SIZE"
    
    # Show 5 most recent files
    echo ""
    echo "5 most recently processed files:"
    find "$OUTPUT_DIR" -name "*_processed.tsv" -type f -printf "%T@ %p\n" | \
        sort -rn | head -5 | cut -d' ' -f2- | xargs -I {} basename {} | \
        sed 's/^/  - /'
fi

# Check for currently running jobs
echo ""
echo "Currently running SLURM jobs:"
echo "----------------------------"
squeue -u $USER -n process_ngrams,process_ngrams_batch -h | wc -l | xargs -I {} echo "Active jobs: {}"

# Show sample of unprocessed files
if [ $PROCESSED_FILES -lt $TOTAL_FILES ]; then
    echo ""
    echo "Sample of unprocessed files:"
    echo "---------------------------"
    
    # Get list of all files and processed files, find the difference
    comm -23 <(find "$INPUT_DIR" -name "googlebooks-eng-fiction-all-*gram-*" -type f -exec basename {} \; | sort) \
             <(find "$OUTPUT_DIR" -name "*_processed.tsv" -type f -exec basename {} \; | sed 's/_processed\.tsv$//' | sort) | \
             head -5 | sed 's/^/  - /'
fi