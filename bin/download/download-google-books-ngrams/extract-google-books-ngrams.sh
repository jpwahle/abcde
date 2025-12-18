#!/usr/bin/env bash
#SBATCH --job-name=extract_data
#SBATCH --output=logs/extract_data.out
#SBATCH --error=logs/extract_data.err
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-668

# Function to extract a single .gz file
extract_file() {
    file="$1"
    extracted_file="${file%.gz}"

    # If the extracted file exists, delete it to allow overwriting
    if [[ -f "$extracted_file" ]]; then
        echo "Overwriting existing extracted file: $extracted_file"
        rm -f "$extracted_file"
    fi

    # Extract the .gz file
    echo "Extracting $file..."
    gunzip --keep "$file"
    if [[ $? -ne 0 ]]; then
        echo "Error extracting $file"
    else
        echo "$file extracted successfully."
    fi
}

# Get list of all .gz files into an array
mapfile -t gz_files < <(find . -maxdepth 1 -type f -name "quadarcs.*.gz" | sort)

# Get the total number of files
total_files=${#gz_files[@]}

# Check if we have files to process
if [[ $total_files -eq 0 ]]; then
    echo "No .gz files found to extract"
    exit 0
fi

# Check if SLURM_ARRAY_TASK_ID is within bounds
if [[ $SLURM_ARRAY_TASK_ID -ge $total_files ]]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID is greater than number of files ($total_files). Nothing to do."
    exit 0
fi

# Get the file for this array task
file_to_process="${gz_files[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID processing file: $file_to_process"

# Process the assigned file
extract_file "$file_to_process"