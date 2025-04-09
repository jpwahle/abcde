#!/bin/bash
#SBATCH --job-name=annotate_reddit
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB

uv run python sample.py --input_dir ~/datasets/reddit-2010-2020/extracted/ --output_jsonl filtered.jsonl --n_workers 16 --memory_per_worker 8GB
