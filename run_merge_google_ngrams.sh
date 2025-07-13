#!/usr/bin/env bash
#SBATCH --job-name=merge_google_ngrams
#SBATCH --output=logs/merge_google_ngrams.out
#SBATCH --error=logs/merge_google_ngrams.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

set -euxo pipefail
export PYTHONUNBUFFERED=1

uv run python merge_google_ngrams.py