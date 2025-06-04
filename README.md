# Dataset Annotation Pipeline

This repository now provides an **end-to-end workflow** that automatically …

## 1. Overview

The pipeline detects users that *self-identify* with demographic traits (e.g. age, gender, …) in Reddit posts and then gathers **all** of the posts written by these users. Every post is enriched with a rich set of linguistic features such as emotion, VAD, tense usage, social warmth, **etc.**

The code is intentionally *data-source agnostic* – only the file crawler (which searches for `RS_*.jsonl`) is Reddit-specific. The self-identification detector and the feature annotator work on plain text and can therefore be re-used for Twitter, blogs, books, or synthetic text in the future.

```mermaid
graph TD
    A[Dump of RS_*.jsonl files (≈3 TB)] -->|Dask-parallel| B[identify_self_users.py]
    B -->|self-identified authors| C[self_users.csv]
    C --> D[collect_user_posts.py]
    D -->|enriched posts| E[self_users_posts.csv]
```

## 2. Downloading the raw Reddit dump

The pipeline expects the uncompressed monthly `RS_YYYY-MM.jsonl` (posts) – and, if you are also interested in comments, `RC_YYYY-MM.jsonl`.  
We bundle helper scripts in `download/` that automate fetching and extracting the **Pushshift Reddit** archive (2010-2022).

```bash
# (optional) regenerate the URL lists – already versioned in the repo
bash download/create-reddit-urls.sh

# (1) Download all monthly Pushshift submissions (~800 GB compressed, ~3 TB uncompressed)
bash download/download-reddit.sh
#   ↳ files will be saved in the current directory as RS_YYYY-MM.zst (and RC_YYYY-MM.zst)

# (2) Decompress the .zst archives (requires the zstd CLI)
bash download/extract-data.sh
#   ↳ produces RS_YYYY-MM.jsonl which are consumed by the pipeline
```

Tips:
* To limit the download to certain years or months, simply edit the `reddit-*-urls.txt` files before step 1.
* Make sure you have **at least 3 TB** of free disk space *after* decompression.
* Install `zstd` if `unzstd`/`zstd` is not available: `sudo apt install zstd` (Debian/Ubuntu).
* You can move the extracted `.jsonl` files anywhere – just pass their directory via `--input_dir` in the next section.

## 3. Quickstart (local execution)

> **Warning**  Scanning the full 3 TB Reddit dump is a multi-hour job. Start with a subset to validate the setup.

```bash
# (1) Find self-identified users (outputs CSV with flattened structure and resolved age)
python identify_self_users.py \
  --input_dir /path/to/reddit/ \
  --output_csv outputs/self_users.csv \
  --n_workers 32

# (2) Collect all posts written by these users (outputs CSV with linguistic features)
python collect_user_posts.py \
  --input_dir /path/to/reddit/ \
  --self_identified_csv outputs/self_users.csv \
  --output_csv outputs/self_users_posts.csv \
  --n_workers 32
```

## 4. Running on a SLURM cluster

Pass `--use_slurm` to either script. Internally we use `dask-jobqueue` to spin up a temporary cluster. Adjust `--n_workers` and `--memory_per_worker` as needed:

```bash
python identify_self_users.py \
  --input_dir /shared/reddit \
  --output_csv outputs/self_users.csv \
  --n_workers 128 \
  --memory_per_worker 8GB \
  --use_slurm
```

## 5. Project layout

```
.
├── sample.py                  # Existing random sampler
├── self_identification.py     # Regex-based detector (data-source agnostic)
├── identify_self_users.py     # Stage 1: find self-identified authors
├── collect_user_posts.py      # Stage 2: gather all posts by those authors & annotate features
├── compute_features.py        # (Lightweight) wrapper around NRC-based feature extraction
└── README.md                  # ← you are here
```

## 6. Extending to other data sources

1. **Crawler** – replace `get_all_jsonl_files` with a function that yields your documents.
2. **Entry adapter** – ensure every record exposes `title`, `selftext`, `author`, and `id` – you can use a small wrapper if field names differ.
3. Everything else (regex detection, feature annotation) works unchanged.

## 7. Feature computation internals

See `compute_features.py` for the current minimal implementation. The stub can be replaced by the full code once the NRC lexicon files are placed under `data/`.

## 8. Development tips

* Use `--verbose` on any script to switch logging to `DEBUG` level.
* The Dask dashboard link is printed at start-up – great for monitoring progress.
* Try running the pipeline on the *sampled* subset (`sample.py`) first to verify everything.
