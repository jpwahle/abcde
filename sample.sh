#!/bin/bash
#SBATCH --job-name=reddit_sampler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=6:00:00

uv run python sample.py \
  --input_dir=/beegfs/wahle/datasets/reddit-2010-2020/extracted \
  --output_jsonl=./sampled_data_1_pct.jsonl \
  --split=text \
  --sample_percentage=1.0 \
  --min_words=10 \
  --max_words=200 \
  --n_workers=30 \
  --memory_per_worker=4GB \
  --use_slurm \
  --seed=42 \
  --verbose
