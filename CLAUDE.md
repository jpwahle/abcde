# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dataset annotation pipeline that processes Reddit, Twitter/X (TUSC), and Google Books Ngrams data to identify users who self-identify demographic information and enriches their posts with linguistic features. The pipeline outputs TSV files suitable for downstream analysis.

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

#### Google Ngrams Processing
```bash
# Process Google Books Ngrams with fast byte-offset indexing
uv run python process_ngrams.py \
    --input_dir path/to/ngrams/extracted \
    --output_dir path/to/output \
    --pattern "*5gram*" \
    --chunk_size 100000 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --build_indexes  # Builds byte-offset indexes on first run
```

### SLURM Submission
```bash
# Submit to SLURM cluster
sbatch submit_reddit.sh
sbatch submit_tusc.sh
sbatch run_google_ngrams.sh
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
- Google Ngrams processing uses byte-offset indexing for fast random access

### Data Quality Filters
- Text length: 5-1000 words
- Excludes: deleted users, AutoModerator, adult content
- Requires valid self-identification patterns
- Age must be resolvable and within valid range

## Data Download and Preparation

### Reddit Data Download
The Reddit data is downloaded using the Academic Torrents dataset. The download script:
- Downloads JSONL files via torrent (requires `aria2c`)
- Supports resumable downloads
- Extracts .zst compressed files to JSONL
- Located in `download-reddit/download_reddit.py`

```bash
# Download Reddit data
cd download-reddit
python download_reddit.py --output_dir /path/to/reddit/data
```

### Google Books Ngrams Download
The Google Ngrams download process:
- Downloads the entire Google Books Ngrams corpus (Version 3)
- Handles 1-gram through 5-gram datasets
- Supports parallel downloads with resumability
- Verifies file integrity using MD5 checksums
- Located in `download-google-books-ngrams/`

```bash
# Download Google Ngrams (interactive menu)
cd download-google-books-ngrams
python download_google_ngrams.py

# Or download specific n-gram type
python download_specific_ngrams.py --ngram-type 5
```

Key features:
- Progress tracking with real-time download speeds
- Automatic retry on failures
- MD5 checksum verification
- Supports downloading to custom directories

### Google Ngrams Processing Optimization
The `process_ngrams.py` script has been optimized with byte-offset indexing:

**Original approach**: Sequential line reading requiring O(n) time to reach line n
**Optimized approach**: Direct byte-offset seeking using memory-mapped index files

Key improvements:
1. **Index Building**: Creates `.idx` files mapping line numbers to byte offsets
2. **Fast Random Access**: Uses `numpy.memmap` for efficient index reading
3. **Parallel Processing**: SLURM tasks can directly jump to their assigned chunks
4. **Memory Efficiency**: Processes chunks without loading entire files

Performance benefits:
- Near-instant access to any line in multi-GB files
- Linear speedup with number of SLURM array tasks
- Minimal memory footprint regardless of file size
- Index files are reusable across runs

The indexing approach matches the Reddit pipeline's performance characteristics.

## Memories and Notes

### Data Processing Notes
- But the google ngrams don't use user extraction (this is only true for social media, I will add more data like LLM generated and blogs to the processing but they won't have user processing)