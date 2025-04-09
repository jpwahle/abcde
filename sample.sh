#!/bin/bash
#SBATCH --job-name=reddit_sampler
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks-per-node=1           # Run a single task per node
#SBATCH --cpus-per-task=64            # Use all 64 cores
#SBATCH --mem=240G                    # Request 240GB of RAM (leaving some for OS)
#SBATCH --time=6:00:00               # Request 24 hours of runtime

uv run python reddit_sampler.py \
  --input_dir=/path/to/input/directory \
  --output_jsonl=/path/to/output/directory/sampled_data.jsonl \
  --split=text \
  --sample_percentage=1.0 \
  --min_words=10 \
  --max_words=200 \
  --n_workers=60 \
  --memory_per_worker=4GB \
  --use_slurm \
  --seed=42 \
  --verbose