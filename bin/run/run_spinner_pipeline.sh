#!/bin/bash
#SBATCH --job-name=spinner_pipeline
#SBATCH --output=logs/spinner_pipeline_%A_%a.out
#SBATCH --error=logs/spinner_pipeline_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-13

set -euxo pipefail

# Check for SLURM_ARRAY_TASK_ID
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. This script is designed to be run as a SLURM array job."
  exit 1
fi

INPUT_DIR="/beegfs/wahle/datasets/spinner/extracted/blogs"
OUTPUT_DIR="outputs_spinner"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# List of tiergroups to process (based on the directory structure provided)
TIERGROUPS=(
    "tiergroup-1"
    "tiergroup-2"
    "tiergroup-3"
    "tiergroup-4"
    "tiergroup-5"
    "tiergroup-6"
    "tiergroup-7"
    "tiergroup-8"
    "tiergroup-9"
    "tiergroup-10"
    "tiergroup-11"
    "tiergroup-12"
    "tiergroup-13"
    "tiergroup-none"
)

# Get the task ID from the environment variable
TASK_ID=$SLURM_ARRAY_TASK_ID

# Check if the task ID is valid
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#TIERGROUPS[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID $TASK_ID is out of bounds for the TIERGROUPS array (0 to ${#TIERGROUPS[@]}-1)."
    exit 1
fi

# Select the tiergroup for this task
TIERGROUP=${TIERGROUPS[$TASK_ID]}

echo "SLURM Task ID: $TASK_ID"
echo "Processing tiergroup: $TIERGROUP"

tiergroup_dir="$INPUT_DIR/$TIERGROUP"

if [ -d "$tiergroup_dir" ]; then
    echo "Processing $TIERGROUP from $tiergroup_dir..."
    
    # Create a separate output directory for this tiergroup
    tiergroup_output_dir="$OUTPUT_DIR/$TIERGROUP"
    mkdir -p "$tiergroup_output_dir"
    
    uv run python3 process_spinner.py \
        --input_dir "$tiergroup_dir" \
        --output_dir "$tiergroup_output_dir"
    
    echo "Finished processing $TIERGROUP."
else
    echo "Warning: Input directory not found for tiergroup '$TIERGROUP'. Skipping."
    echo "         (expected at $tiergroup_dir)"
fi

echo "Spinner processing task finished for $TIERGROUP." 