# Download Folder: Downloading the Google Books Ngrams Dataset

This folder contains helper scripts to fetch and extract the Google Books Ngrams archives from the web.

## Contents

| File                              | Description                                                                       |
|-----------------------------------|-----------------------------------------------------------------------------------|
| `create-google-books-ngram-urls.sh` | Extracts download URLs from `google-books-ngram-urls.html` into `google-books-ngram-urls.txt`. |
| `google-books-ngram-urls.html`   | HTML file containing the source URLs for Google Books Ngrams archives.           |
| `google-books-ngram-urls.txt`    | List of download URLs for ngrams archives (`.gz` files).                         |
| `download-google-books-ngrams.sh` | Downloads all `.gz` archives in parallel using `wget`. Designed for SLURM job scheduling. |
| `extract-google-books-ngrams.sh` | Extracts downloaded `.gz` files using `gunzip` in parallel with SLURM job arrays. |

## Quickstart

Run the following commands from this folder to download and prepare the raw ngrams data:

```bash
# 1. Generate URL list from HTML file (do this once):
bash create-google-books-ngram-urls.sh

# 2. Download all archives (adjust -P for parallel jobs if needed):
bash download-google-books-ngrams.sh

# 3. Extract files (designed for SLURM - adjust array size based on number of files):
bash extract-google-books-ngrams.sh
```

**Note:** The download and extraction scripts are configured for SLURM job scheduling with specific resource allocations. Modify the `#SBATCH` directives as needed for your environment.

After extraction, the ngrams files will be available in this folder for use by the Google Books Ngrams processing pipeline. 