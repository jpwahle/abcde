# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a linguistic analysis pipeline that identifies users who self-identify demographic traits in their posts, then collects all posts by those users to compute linguistic features (emotions, VAD, tense usage, social warmth, etc.). The pipeline supports both Reddit (large-scale, 3TB+ datasets) and TUSC (smaller, ~12GB datasets) data sources.

## Architecture

The pipeline consists of two main stages with data-source specific entry points:

### Reddit Processing (Large-scale, parallel)
1. **Self-identification detection** (`identify_self_users_reddit.py`) - Scans Reddit JSONL dumps for users who mention demographic traits like age
2. **Feature extraction** (`collect_user_posts_reddit.py`) - Collects all posts by identified users and enriches them with linguistic features

### TUSC Processing (Parallel-capable)
1. **Self-identification detection** (`identify_self_users_tusc.py`) - Scans TUSC parquet files for users who mention demographic traits like age  
2. **Feature extraction** (`collect_user_posts_tusc.py`) - Collects all posts by identified users and enriches them with linguistic features

### Shared Core Modules
- `core/` - Common processing logic, cluster management, and I/O utilities
  - `data_processing.py` - Shared self-identification and feature extraction logic
  - `cluster.py` - Dask cluster setup and management utilities
  - `io_utils.py` - Common I/O operations and file format handling
- `reddit/` - Reddit-specific data loading and user management
  - `data_loader.py` - Reddit JSONL file processing with Dask parallelization
  - `user_loader.py` - Reddit user data loading utilities
- `tusc/` - TUSC-specific data loading and user management
  - `data_loader.py` - TUSC parquet file processing with chunked parallel support
  - `user_loader.py` - TUSC user data loading utilities

## Dependencies and Environment

Uses `uv` for dependency management. Core dependencies:
- `dask[distributed]` for parallel processing
- `dask-jobqueue` for SLURM cluster support
- Standard libraries for text processing

## Common Commands

### Reddit Processing

#### Local execution:
```bash
# Stage 1: Find self-identified users (outputs CSV with majority-voted age and flattened structure)
uv run python identify_self_users_reddit.py --input_dir /path/to/reddit/ --output_csv outputs/self_users.csv --n_workers 32

# Stage 2: Collect posts and compute features (outputs CSV with linguistic features)
uv run python collect_user_posts_reddit.py --input_dir /path/to/reddit/ --self_identified_csv outputs/self_users.csv --output_csv outputs/self_users_posts.csv --n_workers 32
```

#### SLURM cluster execution:
Add `--use_slurm` flag and adjust workers/memory:
```bash
uv run python identify_self_users_reddit.py --input_dir /shared/reddit --output_csv outputs/self_users.csv --n_workers 128 --memory_per_worker 8GB --use_slurm
```

#### Full Reddit pipeline (SLURM):
```bash
sbatch run_reddit_pipeline.sh
```

### TUSC Processing

#### Local execution (single-machine, test mode):
```bash
# Stage 1: Find self-identified users
uv run python identify_self_users_tusc.py --input_file /path/to/tusc.parquet --output_csv outputs/tusc_self_users.csv --test_mode

# Stage 2: Collect posts and compute features
uv run python collect_user_posts_tusc.py --input_file /path/to/tusc.parquet --self_identified_csv outputs/tusc_self_users.csv --output_csv outputs/tusc_user_posts.csv --test_mode
```

#### SLURM cluster execution (full-scale parallel):
```bash
# Stage 1: Find self-identified users (parallel processing)
uv run python identify_self_users_tusc.py --input_file /path/to/tusc.parquet --output_csv outputs/tusc_self_users.csv --n_workers 128 --memory_per_worker 4GB --use_slurm

# Stage 2: Collect posts and compute features (parallel processing)
uv run python collect_user_posts_tusc.py --input_file /path/to/tusc.parquet --self_identified_csv outputs/tusc_self_users.csv --output_csv outputs/tusc_user_posts.csv --n_workers 128 --memory_per_worker 4GB --use_slurm
```

#### Full TUSC pipeline (SLURM):
```bash
sbatch run_tusc_pipeline.sh
```

#### Test TUSC pipeline (SLURM, small sample):
```bash
sbatch run_tusc_pipeline_test.sh
```

## Data Requirements

### Reddit
- Expects uncompressed Reddit Pushshift files (`RS_YYYY-MM.jsonl`) in input directory
- Uses both `author_id` (preferred) and `author` (fallback) for user identification
- Supports `text` vs `multimodal` content filtering

### TUSC
- Expects TUSC parquet files (either city or country splits)
- Split type is auto-determined from filename if not specified
- Uses UserID/userID and UserName/userName fields for identification

### Shared
- NRC lexicon files should be placed in `data/` directory (graceful fallback if missing)
- All outputs support both CSV and TSV formats

## Key Design Patterns

- **Modular architecture**: Clear separation between Reddit and TUSC data sources with parallel processing capabilities for both
- **Shared core logic**: Common functions for self-identification detection and feature computation
- **Graceful degradation**: Feature computation falls back to empty dicts if lexicon files missing
- **Dual identification**: Uses both stable user IDs and usernames for user matching across dataset inconsistencies
- **Configurable processing**: Different scaling strategies for different data sources
- **Distributed processing**: Dask-based parallelization with SLURM support for both Reddit and TUSC data sources