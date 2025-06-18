# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dataset annotation pipeline that processes Reddit and Twitter/X (TUSC) data to identify users who self-identify demographic information and enriches their posts with linguistic features. The pipeline outputs TSV files suitable for downstream analysis.

## Commands

### Development Environment
```bash
# Install dependencies using UV (required)
uv sync

# Install pre-commit hooks (one-time setup)
uv run pre-commit install

# Run tests
uv run pytest tests/

# Run a specific test
uv run pytest tests/test_pipeline.py::test_reddit_pipeline

# Run pre-commit manually on all files
uv run pre-commit run --all-files
```

### Running the Pipeline

#### Reddit Processing
```bash
# Process Reddit JSONL files
uv run python process_reddit.py \
    --input_dir path/to/reddit/jsonl \
    --output_dir path/to/output \
    --chunk_size 100000 \
    --workers 4 \
    --stages both  # or 'stage1' or 'stage2'
```

#### TUSC (Twitter/X) Processing
```bash
# Process TUSC parquet files
uv run python process_tusc.py \
    --input_file path/to/tusc.parquet \
    --output_dir path/to/output \
    --stages both  # or 'stage1' or 'stage2'
```

### SLURM Submission
```bash
# Submit to SLURM cluster
sbatch submit_reddit.sh
sbatch submit_tusc.sh
```

## Architecture

### Two-Stage Processing Pipeline

**Stage 1 - Self-Identification Detection**
- Scans all posts to find users who self-identify demographics (age, gender, location, occupation, religion)
- Uses regex patterns defined in `SelfIdentificationDetector` class
- Outputs a users TSV with demographic information

**Stage 2 - Feature Extraction**
- Collects all posts from self-identified users (from Stage 1)
- Applies linguistic features using NRC lexicons and custom analyzers
- Outputs a posts TSV with all features

### Key Components

**`helpers.py`** - Core functionality:
- `SelfIdentificationDetector`: Detects demographic self-identification using regex patterns
- `detect_with_mappings()`: New method that returns both raw extractions and mapped values
- `compute_vad_and_emotions()`: Computes linguistic features from NRC lexicons
- Data loading functions for demographic mappings (DMG-* files)

**Demographic Mappings**:
- Cities → Countries (from geonames CSV)
- Religions → Main Religion & Category
- Occupations → SOC Titles
- Countries → Nationalities (generated)

### Data Processing Flow
1. Read raw data (JSONL for Reddit, Parquet for TUSC)
2. For each post, detect self-identification statements
3. Aggregate to user level, resolve multiple age extractions
4. Filter users (age 13-99, valid birth year)
5. Collect all posts from valid users
6. Apply linguistic features to each post
7. Output TSV files with proper field naming conventions

## Important Implementation Details

### Age Resolution
- Multiple age extractions are resolved using clustering and confidence scoring
- Birth year is calculated based on post timestamp
- Ages outside 13-99 range are filtered out

### Linguistic Features
All features are computed per post using NRC lexicons:
- VAD scores (valence, arousal, dominance)
- Emotion categories (10 emotions)
- Worry/calmness indicators
- Moral trustworthiness
- Social warmth
- Pronoun usage (1st, 2nd, 3rd person)
- Body part mentions
- Tense analysis

### Output Field Naming
Demographic fields follow specific naming conventions:
- `DMGRawExtractedCity` → `DMGCountryMappedFromExtractedCity`
- `DMGRawExtractedReligion` → `DMGMainReligionMappedFromExtractedReligion`
- `DMGRawExtractedOccupation` → `DMGSOCTitleMappedFromExtractedOccupation`

### Performance Considerations
- Use `--chunk_size` to control memory usage (default 100k for Reddit)
- Set `--workers` based on available CPU cores
- Reddit processing uses fast indexing with numpy/orjson
- TUSC processing leverages parquet's columnar format

### Data Quality Filters
- Text length: 5-1000 words
- Excludes: deleted users, AutoModerator, adult content
- Requires valid self-identification patterns
- Age must be resolvable and within valid range
