#!/usr/bin/env bash
#SBATCH --job-name=download_reddit
#SBATCH --output=logs/download_reddit.out
#SBATCH --error=logs/download_reddit.err
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

cat reddit-posts-urls.txt | xargs -n 1 -P 4 wget
