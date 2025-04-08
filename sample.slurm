#!/bin/bash
#SBATCH --job-name=annotate_reddit
#SBATCH --partition=normal
#SBATCH --time=1-06:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --mem=256GB

uv run python sample.py --input_dir ~/datasets/reddit-2010-2020/extracted/ --output_jsonl filtered.jsonl --n_workers 64