# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Reddit linguistic analysis pipeline that identifies users who self-identify demographic traits in their posts, then collects all posts by those users to compute linguistic features (emotions, VAD, tense usage, social warmth, etc.). The pipeline is data-source agnostic - only the file crawler is Reddit-specific.

## Architecture

The pipeline consists of two main stages:
1. **Self-identification detection** (`identify_self_users.py`) - Scans Reddit dumps for users who mention demographic traits like age
2. **Feature extraction** (`collect_user_posts.py`) - Collects all posts by identified users and enriches them with linguistic features

Core modules:
- `self_identification.py` - Regex-based detector for demographic self-identification (data-source agnostic)
- `compute_features.py` - NRC lexicon-based feature computation with safe fallbacks for missing data files
- `helpers.py` - Shared utilities for file processing, filtering, and media handling

## Dependencies and Environment

Uses `uv` for dependency management. Core dependencies:
- `dask[distributed]` for parallel processing
- `dask-jobqueue` for SLURM cluster support
- Standard libraries for text processing

## Common Commands

### Local execution:
```bash
# Stage 1: Find self-identified users
uv run python identify_self_users.py --input_dir /path/to/reddit/ --output_jsonl outputs/self_users.jsonl --n_workers 32

# Stage 2: Collect posts and compute features
uv run python collect_user_posts.py --input_dir /path/to/reddit/ --self_identified_jsonl outputs/self_users.jsonl --output_jsonl outputs/self_users_posts.jsonl --n_workers 32
```

### SLURM cluster execution:
Add `--use_slurm` flag and adjust workers/memory:
```bash
uv run python identify_self_users.py --input_dir /shared/reddit --output_jsonl outputs/self_users.jsonl --n_workers 128 --memory_per_worker 8GB --use_slurm
```

### Full pipeline (SLURM):
```bash
sbatch run_reddit_pipeline.sh
```

## Data Requirements

- Expects uncompressed Reddit Pushshift files (`RS_YYYY-MM.jsonl`) in input directory
- NRC lexicon files should be placed in `data/` directory (graceful fallback if missing)
- Uses both `author_id` (preferred) and `author` (fallback) for user identification

## Key Design Patterns

- **Graceful degradation**: Feature computation falls back to empty dicts if lexicon files missing
- **Dual identification**: Uses both stable `author_id` and username for user matching across dataset inconsistencies
- **Configurable splits**: `text` vs `multimodal` content filtering
- **Distributed processing**: Dask-based parallelization with SLURM support