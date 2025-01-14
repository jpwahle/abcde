#!/usr/bin/env bash
#SBATCH --job-name=download_google_books_ngrams
#SBATCH --output=logs/download_google_books_ngrams.out
#SBATCH --error=logs/download_google_books_ngrams.err
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

cat google-books-ngram-urls.txt | xargs -n 1 -P 4 wget
