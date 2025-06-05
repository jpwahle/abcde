#!/usr/bin/env bash
#SBATCH --job-name=extract_data
#SBATCH --output=logs/extract_data.out
#SBATCH --error=logs/extract_data.err
#SBATCH --time=8:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

# Function to extract a single .zst file
extract_file() {
    file="$1"
    extracted_file="${file%.zst}"

    # If the extracted file exists, delete it to allow overwriting
    if [[ -f "$extracted_file" ]]; then
        echo "Overwriting existing extracted file: $extracted_file"
        rm -f "$extracted_file"
    fi

    # Extract the .zst file
    echo "Extracting $file..."
    unzstd --keep --long=31 "$file"
    if [[ $? -ne 0 ]]; then
        echo "Error extracting $file"
    else
        echo "$file extracted successfully."
    fi
}

# Export the function so it can be used by xargs
export -f extract_file

# Find all .zst files and process them with 8 parallel tasks
find . -maxdepth 1 -type f -name "*.zst" | xargs -I {} -P 8 bash -c 'extract_file "$@"' _ {}