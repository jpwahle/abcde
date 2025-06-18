# Download Folder: Downloading the Pushshift Reddit Dataset

This folder contains helper scripts to fetch and extract the Pushshift Reddit archives (submissions + comments) from archive.org.

## Contents

| File                      | Description                                                                       |
|---------------------------|-----------------------------------------------------------------------------------|
| `create-reddit-urls.sh`   | Generates monthly URL lists (`reddit-posts-urls.txt` and `reddit-comment-urls.txt`) for each year/month archive.
| `reddit-posts-urls.txt`   | List of download URLs for monthly submissions archives (`.zst`).                  |
| `reddit-comment-urls.txt` | List of download URLs for monthly comments archives (`.zst`).                     |
| `download-reddit.sh`      | Downloads all `.zst` archives in parallel using `wget`. Logs to `download.log`.    |
| `extract-data.sh`         | Extracts downloaded `.zst` files to `.jsonl` format using `unzstd` in parallel.    |

## Quickstart

Run the following commands from this folder to download and prepare the raw JSONL data:

```bash
# 1. Generate URL lists (do this once):
bash create-reddit-urls.sh

# 2. Download all archives (adjust -P for parallel jobs in download-reddit.sh if needed):
bash download-reddit.sh

# 3. Extract JSONL files (adjust -P for parallel jobs in extract-data.sh if needed):
bash extract-data.sh
```

After extraction, the `.jsonl` files will be available in this folder for use by the Reddit processing pipeline.
