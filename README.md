# ABCDE Repository README

## Overview

This repository, named **ABCDE** (likely standing for something like "Age, Birth, Country, Demography Extraction" based on the code), is designed for processing large textual datasets to extract demographic self-identification (e.g., age, gender, occupation, location, religion) and compute linguistic features (e.g., emotions, valence-arousal-dominance, warmth-competence, tense, body part mentions). It supports datasets like Reddit (posts and comments), Google Books Ngrams, and Twitter (via the TUSC dataset).

Key functionalities:
- **Self-Identification Detection**: Uses regex patterns to detect statements like "I am 25 years old" or "I live in London". Includes mappings (e.g., city to country, religion to category).
- **Linguistic Feature Extraction**: Computes features from various lexicons (e.g., NRC emotions, VAD, worry words) and custom lists (e.g., body parts, tenses).
- **Data Processing Pipelines**: Scripts for downloading, extracting, and processing datasets in parallel (e.g., via SLURM for HPC environments).
- **Output**: Generates TSV files for users (aggregated demographics) and posts (per-entry features with demographics).

The code is Python-based, with Bash scripts for downloading/extraction and SLURM job management. It's optimized for large-scale data (e.g., Reddit's 2010-2022 dumps).

This README is structured to help you understand the repo, run it, and extend/refactor it in the future. For refactors:
- Centralize lexicon loading in `helpers.py` for easy swaps.
- Patterns are in `SelfIdentificationDetector` – add new categories by extending `self.patterns`.
- Feature computation is modular in `apply_linguistic_features` – add new features by extending the function.
- Use tests in `tests/` to validate changes.

## Setup and Installation

1. **Clone the Repository**:
   ```
   git clone <repo-url>
   cd abcde
   ```

2. **Python Environment**:
   - Requires Python 3.10+.
   - Install dependencies with `uv` (recommended for speed) or `pip`:
     ```
     pip install uv  # If not installed
     uv sync  # Installs from pyproject.toml
     ```
   - Key libraries: `pandas`, `nltk`, `presidio_analyzer`, `re` (built-in), etc.
   - Download NLTK data:
     ```
     python -c "import nltk; nltk.download('stopwords')"
     ```

3. **Data Directories**:
   - Lexicons are in `data/` (e.g., `NRC-Emotion-Lexicon.txt`).
   - Input datasets go in `~/datasets/` (configurable in scripts).
   - Outputs go in configurable directories (e.g., `/beegfs/wahle/github/abcde/` in examples).

4. **HPC/SLURM Setup** (Optional):
   - Scripts like `run_reddit_pipeline.sh` submit SLURM jobs.
   - Adjust `--mem`, `--time`, etc., based on your cluster.

## Data Downloading

The repo processes three main datasets: Reddit, Google Books Ngrams, and Twitter (TUSC). Scripts are provided for downloading Reddit and Google Books. TUSC is manual.

### 1. Reddit (2010-2022)
   - Source: Pushshift dumps archived on Internet Archive.
   - Files: Monthly ZST-compressed JSONL (RS_* for submissions/posts, RC_* for comments).
   - Download Script: `download-reddit/download-reddit.sh`
     - Generates URLs in `reddit-posts-urls.txt` and `reddit-comment-urls.txt` via `create-reddit-urls.sh`.
     - Runs: `./download-reddit/download-reddit.sh` (parallel downloads with `wget`).
     - Output: `~/datasets/reddit-2010-2022/downloaded/` (ZST files).
   - Extraction: `./download-reddit/extract-reddit.sh` (uses `zstd -d` to extract to `extracted/`).
   - Full Pipeline: Use `run_reddit_pipeline_full.sh` (downloads, extracts, processes).
   - Size: ~10TB compressed; plan storage accordingly.
   - Tip: For testing, use `run_reddit_pipeline_test.sh` on sample data.

### 2. Google Books Ngrams (English Fiction)
   - Source: Google Books Ngram Viewer (v2 or v3; script uses v2 for fiction).
   - Files: Ngram files (1-5 grams) in gzipped TSV.
   - Download Script: `download-google-books-ngrams/download-google-books-ngrams.sh`
     - Generates URLs in `google-books-ngram-urls.txt` via `create-google-books-ngram-urls.sh`.
     - Runs: `./download-google-books-ngrams/download-google-books-ngrams.sh` (parallel with `wget`).
     - Output: `~/datasets/google-books-ngrams/downloaded/` (gz files).
   - Extraction: `./download-google-books-ngrams/extract-google-books-ngrams.sh` (uses `gunzip` to `extracted/`).
   - Processing: `run_google_ngrams.sh` (SLURM job for `process_ngrams.py`).
   - Monitoring: `check_ngrams_progress.sh` shows progress.
   - Size: ~100GB extracted.

### 3. Twitter (TUSC Dataset)
   - Source: https://github.com/Priya22/EmotionDynamics (TUSC: Twitter User Self-Identification Corpus).
   - Files: `tusc-city.parquet` and `tusc-country.parquet` (Parquet format with tweets).
   - Download: Manual – clone the repo and copy Parquet files to `~/datasets/tusc/`.
   - Processing: `run_tusc_pipeline.sh` or `run_tusc_pipeline_full.sh` (uses `process_tusc.py`).
   - Tip: For testing, use `extract_tusc_test_data.sh` to create sample data in `data/test/tusc/`.

## Lexicons Used

All lexicons are stored in `data/` and loaded in `helpers.py` via functions like `_load_lexicon()`. They power feature extraction in `apply_linguistic_features()` and demographic detection in `SelfIdentificationDetector`.

### 1. NRC Lexicons (Emotion, VAD, Warmth/Competence, Anxiety/Calmness)
   - **Source**: Dr. Saif Mohammad's NRC lab (https://saifmohammad.com/).
   - **Files**:
     - `NRC-Emotion-Lexicon.txt`: Word-emotion associations (anger, joy, etc.).
       - Link: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
       - Usage: Computes flags/counts/averages for 10 emotions (e.g., `NRCHasJoyWord`, `NRCCountPositiveWords`).
     - `NRC-VAD-Lexicon.txt`: Valence (pleasantness), Arousal (activation), Dominance (control).
       - Link: https://saifmohammad.com/WebPages/nrc-vad.html
       - Usage: Averages (e.g., `NRCAvgValence`) and high/low thresholds (e.g., `NRCHasHighValenceWord` at >0.75).
     - `NRC-Warmth-Lexicon.txt` & `NRC-SocialWarmth-Lexicon.txt`: Warmth (sociability, kindness) and competence (ability).
       - Link: https://saifmohammad.com/WebPages/warmth.html
       - Usage: Averages/flags for warmth (e.g., `NRCAvgWarmthWord`, `NRCHasHighSocialWarmthWord`).
     - `NRC-WorryWords-Lexicon.txt`: Anxiety/calmness words (e.g., "worried" vs. "relaxed").
       - Link: https://saifmohammad.com/worrywords.html (part of affect intensity).
       - Usage: Averages/flags for anxiety (e.g., `NRCAvgAnxiety`, `NRCHasHighAnxietyWord`).
     - `NRC-MoralTrustworthy-Lexicon.txt`: Moral trustworthiness (integrity, reliability).
       - Link: Derived from warmth/competence work.
       - Usage: Averages/flags (e.g., `NRCAvgMoralTrustWord`).
   - **Loading**: Via `_load_nrc_*()` functions. Words are lowercased.
   - **Refactor Tip**: To add a new NRC lexicon, create a loader like `_load_nrc_emotion_lexicon()` and integrate into `compute_vad_and_emotions()`.

### 2. Tense Lexicon
   - **Source**: UniMorph (English inflections): https://github.com/unimorph/eng
   - **File**: `TIME-eng-word-tenses.txt` (processed from UniMorph).
   - **Usage**: Detects past/present/future tenses (e.g., "walked" as past).
     - Features: `TIMEHasPastWord`, `TIMECountPresentWords`, etc.
   - **Loading**: `_load_eng_tenses_lexicon()` (key: base word, value: list of inflections).
   - **Refactor Tip**: Extend for other languages by swapping the lexicon and adjusting regex in `apply_linguistic_features()`.

### 3. Body Part Mentions (BPMs)
   - **Sources**:
     - Collins Dictionary: https://www.collinsdictionary.com/us/word-lists/bodyparts-of-the-body
     - Enchanted Learning: https://www.enchantedlearning.com/wordlist/body.shtml
   - **File**: Combined into `BPM-body-part-list.txt` (one word per line).
   - **Usage**: Detects mentions with pronouns (e.g., "my head" → `MyBPM` list includes "head").
     - Features: `HasBPM` (binary), `MyBPM`, `YourBPM`, etc. (lists of parts).
   - **Loading**: Simple line-split in `apply_linguistic_features()`.
   - **Refactor Tip**: To filter false positives, add context regex (e.g., exclude "head of state").

### 4. Occupations
   - **Source**: U.S. Bureau of Labor Statistics (SOC 2018): https://www.bls.gov/soc/2018/#materials
   - **File**: `DMG-occupation-list.txt` (processed list of job titles).
   - **Usage**: Detection patterns in `SelfIdentificationDetector` (e.g., "I am a software engineer").
     - Mapping: `dmg_occupation_to_soc` (raw term → SOC title, e.g., "software engineer" → "Software Developers").
     - Output: `DMGRawExtractedOccupation`, `DMGSOCTitleMappedFromExtractedOccupation`.
   - **Loading**: `_load_lexicon()` with lowercasing.
   - **Refactor Tip**: For updates, download new SOC data and regenerate the list/mapping.

### 5. Gender, Country, Religion
   - **Sources** (Wikipedia):
     - Gender: https://en.wikipedia.org/wiki/List_of_gender_identities → `DMG-gender-list.txt`
     - Country: https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population → `DMG-country-list.txt`
     - Religion: https://en.wikipedia.org/wiki/List_of_religions_and_spiritual_traditions → `DMG-religion-list.csv` (with substrains, main religions, categories).
   - **Usage**: Detection patterns for self-ID.
     - Religion Mappings: Raw → Main (e.g., "catholic" → "Christianity"), Category (e.g., "Abrahamic Religions").
     - Output: `DMGRawExtractedGender`, `DMGRawExtractedCountry`, `DMGRawExtractedReligion`, etc.
   - **Loading**: Line-split or CSV in `helpers.py`.
   - **Refactor Tip**: Use APIs (e.g., Wikipedia dumps) for auto-updates; add confidence scores to patterns.

### 6. Cities
   - **Source**: GeoNames (population >1000): https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/table/?disjunctive.cou_name_en&sort=name
   - **File**: `DMG-geonames-all-cities-with-a-population-1000.csv` (CSV with cities, countries, alternates).
   - **Usage**: Detection + Mapping (city → country, e.g., "london" → "United Kingdom").
     - Filtered: Excludes stopwords, countries, short names (<4 chars).
     - Output: `DMGRawExtractedCity`, `DMGCountryMappedFromExtractedCity`.
   - **Loading**: Pandas in `_load_dmg_cities()` (sorts by population for precedence).
   - **Refactor Tip**: Integrate GeoPandas for lat/long if needed; handle ambiguities (e.g., multiple "Springfield"s).

**General Lexicon Tips**:
- All loaded as dicts/sets in `helpers.py`.
- Preprocessing: Lowercased, filtered (e.g., stopwords for cities).
- To add a lexicon: Create a loader, integrate into detection/features, update tests.
- Licensing: NRC/Wikipedia are public; cite sources in papers.

## Running the Pipelines

### Reddit
- Full: `./run_reddit_pipeline_full.sh` (download + process).
- Process Only: `python process_reddit.py --input_dir <extracted> --output_dir <out> --stages both`.
- Outputs: `reddit_users.tsv` (user demographics), `reddit_users_posts.tsv` (posts with features).
- Stages: `detect` (demographics), `features` (linguistic), `both`.

### Google Books Ngrams
- Full: `./run_google_ngrams.sh` (processes extracted ngrams).
- Outputs: Processed TSVs in `~/datasets/google-books-ngrams/processed/`.

### TUSC (Twitter)
- Full: `./run_tusc_pipeline_full.sh`.
- Process: `python process_tusc.py --input_file <parquet> --output_dir <out> --stages both`.
- Outputs: Separate for city/country splits (e.g., `city_users.tsv`).

### Monitoring
- Reddit: `./monitor_reddit_pipeline.sh`.
- Ngrams: `./check_ngrams_progress.sh`.
- Line Counts: `./compute_linecounts.sh` (for Reddit).

### Testing
- Run: `uv run pytest`.
- Extracts samples: `./extract_reddit_test_data.sh`, `./extract_tusc_test_data.sh`.
- Verifies: `test_output_verification.py`.

## Output Format

- **Users TSV**: User ID, aggregated demographics (e.g., `DMGMajorityBirthyear`, lists of extracted cities).
- **Posts TSV**: Post ID, User ID, Text, Demographics (e.g., `DMGAgeAtPost`), Linguistic Features (e.g., `NRCAvgValence`, `TIMEHasPastWord`).
- All values lowercased; lists as comma-separated.
- Use `get_csv_fieldnames()` in `helpers.py` for exact columns.

If issues arise, check logs in `logs/` or SLURM outputs. For questions, refer to code comments or open an issue.