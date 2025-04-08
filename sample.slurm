#!/bin/sh
#SBATCH --job-name=sample_reddit
#SBATCH --partition=normal
#SBATCH --time=1-6:0:0 # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

uv run python sample.py --input_dir ~/datasets/reddit-2010-2020/extracted/ --output_jsonl filtered.jsonl
