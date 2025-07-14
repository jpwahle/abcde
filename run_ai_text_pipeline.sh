#!/bin/bash
#SBATCH --job-name=ai_text_pipeline
#SBATCH --output=logs/ai_text_pipeline_%A_%a.out
#SBATCH --error=logs/ai_text_pipeline_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-15

set -e

# Check for SLURM_ARRAY_TASK_ID
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. This script is designed to be run as a SLURM array job."
  exit 1
fi

INPUT_DIR="$HOME/datasets/ai-text-datasets"
OUTPUT_DIR="outputs_ai_text_pipeline"

mkdir -p "$OUTPUT_DIR"

# List of datasets to process
# Format: "logical_name:filename"
DATASETS=(
    "wildchat-1m:wildchat_data.csv"
    "lmsys-1m:lmsys_data.csv"
    "pippa:pippa_data.csv"
    "hh-rlhf:hh-rlhf_data.csv"
    "prism:prism_data.csv"
    "apt-paraphrase-dataset-gpt-3:apt-paraphrase-dataset-gpt-3.tsv"
    "anthropic-persuasiveness:anthropic_persuasiveness_data.csv"
    "M4:m4_data.csv"
    "mage:mage_data.csv"
    "luar:luar_lwd_data.csv"
    "general_thoughts_430k:general_thoughts_430k_data.csv"
    "reasoning_shield:reasoning_shield_data.csv"
    "safechain:safechain_data.csv"
    "star1:star1_data.csv"
    "raid:raid_data.csv"
    "tinystories:tinystories_data.csv"
)

# Get the task ID from the environment variable
TASK_ID=$SLURM_ARRAY_TASK_ID

# Check if the task ID is valid
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID $TASK_ID is out of bounds for the DATASETS array (0 to ${#DATASETS[@]}-1)."
    exit 1
fi

# Select the dataset for this task
ITEM=${DATASETS[$TASK_ID]}
IFS=':' read -r dataset_name filename <<< "$ITEM"

echo "SLURM Task ID: $TASK_ID"
echo "Processing dataset: $dataset_name"

input_file="$INPUT_DIR/$filename"

if [ -f "$input_file" ]; then
    echo "Processing $dataset_name from $input_file..."
    python3 process_ai_text.py \
        --input_file "$input_file" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name "$dataset_name"
    echo "Finished processing $dataset_name."
else
    echo "Warning: Input file not found for dataset '$dataset_name'. Skipping."
    echo "         (expected at $input_file)"
fi

echo "AI text processing task finished."

