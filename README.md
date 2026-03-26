# ABCDE - Affect, Body, Cognition, Demographics, and Emotion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2512.17752-b31b1b.svg)](https://arxiv.org/abs/2512.17752)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-abcde-orange)](https://huggingface.co/datasets/jpwahle/abcde)

## Overview

**ABCDE** (Affect, Body, Cognition, Demographics, and Emotion) is a Python library for processing large textual datasets to extract demographic self-identification (e.g., age, gender, occupation, location, religion) and compute linguistic features (e.g., emotions, valence-arousal-dominance, warmth-competence, tense, body part mentions).

### Key Features

- **Self-Identification Detection**: Uses regex patterns to detect statements like "I am 25 years old" or "I live in London". Includes mappings (e.g., city to country, religion to category).
- **Linguistic Feature Extraction**: Computes features from various lexicons (e.g., NRC emotions, VAD, worry words) and custom lists (e.g., body parts, tenses).
- **Data Processing Pipelines**: Scripts for downloading, extracting, and processing datasets in parallel (e.g., via SLURM for HPC environments).
- **Output**: Generates TSV files for users (aggregated demographics) and posts (per-entry features with demographics).

### Supported Datasets

- **Reddit** (2010-2022): Posts and comments from Pushshift dumps
- **TUSC** (Twitter): Geolocated tweets from the TUSC dataset
- **Google Books Ngrams**: English fiction 5-grams
- **AI-Generated Text**: WildChat, LMSYS, PIPPA, HH-RLHF, RAID, and more
- **Blog Spinner Data**: XML-based blog posts

## Repository Structure

```
abcde/
├── src/abcde/              # Main Python package
│   ├── __init__.py
│   ├── core/               # Core detection and feature extraction
│   │   ├── detector.py     # SelfIdentificationDetector class
│   │   ├── features.py     # Linguistic feature computation
│   │   └── pii.py          # PII detection
│   ├── lexicons/           # Lexicon loading utilities
│   │   └── loaders.py
│   ├── io/                 # I/O utilities
│   │   ├── csv_utils.py    # CSV/TSV reading/writing
│   │   ├── jsonl_utils.py  # JSONL file handling
│   │   └── demographics.py # Demographics aggregation
│   └── utils/              # Utility functions
│       ├── banner.py
│       ├── dates.py
│       └── text.py
├── scripts/                # Processing scripts (CLI tools)
│   ├── process_reddit.py
│   ├── process_tusc.py
│   ├── process_ngrams.py
│   ├── process_ai_text.py
│   ├── process_spinner.py
│   └── merge_dataset.py
├── bin/                    # Shell scripts
│   ├── download/           # Data download scripts
│   │   ├── download-reddit/
│   │   └── download-google-books-ngrams/
│   └── run/                # Pipeline execution scripts
│       ├── run_reddit_pipeline.sh
│       ├── run_tusc_pipeline_full.sh
│       └── ...
├── lexicons/               # Lexicon data files
│   ├── NRC-*.txt           # NRC emotion/VAD/warmth lexicons
│   ├── DMG-*.txt           # Demographic data (countries, cities, etc.)
│   ├── BPM-*.txt           # Body part mentions
│   ├── COG-*.json          # Cognitive/thinking words
│   └── TIME-*.txt          # Tense data
├── tests/                  # Test files
├── helpers.py              # Backward-compatible imports
├── pyproject.toml          # Package configuration
├── README.md               # This file
└── DATASET.md              # Dataset documentation
```

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/abcde-project/abcde.git
cd abcde

# Install with uv (recommended)
pip install uv
uv sync

# Or install with pip
pip install -e .
```

### Requirements

- Python 3.10+
- Dependencies are managed via `pyproject.toml`

### Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### As a Library

```python
from abcde import SelfIdentificationDetector, apply_linguistic_features

# Detect demographic self-identification
detector = SelfIdentificationDetector()
text = "I am 25 years old and I live in London. I'm a software engineer."
matches = detector.detect(text)
print(matches)
# {'age': ['25'], 'city': ['london'], 'occupation': ['software engineer']}

# Get mappings (city -> country, etc.)
detailed = detector.detect_with_mappings(text)
print(detailed['city'])
# {'raw': ['london'], 'country_mapped': ['United Kingdom']}

# Extract linguistic features
features = apply_linguistic_features(text)
print(features['NRCAvgValence'])  # Average valence score
print(features['WordCount'])      # Word count
```

### Processing Scripts

```bash
# Process Reddit data
python scripts/process_reddit.py \
    --input_dir /path/to/reddit/extracted \
    --output_dir /path/to/output \
    --stages both

# Process TUSC data
python scripts/process_tusc.py \
    --input_file /path/to/tusc-city.parquet \
    --output_dir /path/to/output \
    --stages both

# Process AI-generated text
python scripts/process_ai_text.py \
    --input_file /path/to/wildchat.csv \
    --output_dir /path/to/output \
    --dataset_name wildchat

# Process Google Books Ngrams
python scripts/process_ngrams.py \
    --input_path /path/to/ngrams \
    --output_path /path/to/output
```

### SLURM/HPC Usage

```bash
# Submit Reddit pipeline
sbatch bin/run/run_reddit_pipeline.sh

# Submit TUSC pipeline
sbatch bin/run/run_tusc_pipeline_full.sh

# Submit AI text pipeline
sbatch bin/run/run_ai_text_pipeline.sh
```

## Data Downloading

### Reddit (2010-2022)

```bash
# Generate URLs and download
cd bin/download/download-reddit
./create-reddit-urls.sh
./download-reddit.sh

# Extract
./extract-reddit.sh
```

### Google Books Ngrams

```bash
cd bin/download/download-google-books-ngrams
./create-google-books-ngram-urls.sh
./download-google-books-ngrams.sh
./extract-google-books-ngrams.sh
```

### TUSC (Twitter)

Download from https://github.com/Priya22/EmotionDynamics and place parquet files in your data directory.

## Lexicons

All lexicons are stored in `lexicons/` and include:

| Lexicon | Description | Source |
|---------|-------------|--------|
| NRC-Emotion-Lexicon.txt | Word-emotion associations | [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) |
| NRC-VAD-Lexicon.txt | Valence-Arousal-Dominance scores | [NRC](https://saifmohammad.com/WebPages/nrc-vad.html) |
| NRC-WorryWords-Lexicon.txt | Anxiety/calmness words | [NRC](http://saifmohammad.com/worrywords.html) |
| NRC-*Warmth-Lexicon.txt | Social warmth scores | [NRC](https://saifmohammad.com/WebPages/lexicons.html) |
| DMG-country-list.txt | Country names | Wikipedia |
| DMG-geonames-*.csv | City data with countries | [GeoNames](https://public.opendatasoft.com) |
| DMG-religion-list.csv | Religion mappings | Wikipedia |
| TIME-eng-word-tenses.txt | English verb tenses | [UniMorph](https://github.com/unimorph/eng) |
| BPM-bodywords-full.txt | Body part words | Collins/Enchanted Learning |
| COG-thinking-words-categorized.json | Cognitive verbs | Custom |

## Output Format

### Users TSV
Contains aggregated demographics per user:
- `Author`: User ID
- `DMGMajorityBirthyear`: Resolved birth year
- `DMGRawExtractedCity`, `DMGCountryMappedFromExtractedCity`
- `DMGRawExtractedReligion`, `DMGMainReligionMappedFromExtractedReligion`
- `DMGRawExtractedOccupation`, `DMGSOCTitleMappedFromExtractedOccupation`

### Posts TSV
Contains per-post data with linguistic features:
- Post metadata (ID, text, timestamp, etc.)
- `DMGAgeAtPost`: Age when the post was created
- NRC features (`NRCAvgValence`, `NRCHasJoyWord`, etc.)
- Pronoun features (`PRNHasI`, `PRNHasWe`, etc.)
- Tense features (`TIMEHasPastVerb`, `TIMECountPresentVerbs`, etc.)
- Cognitive features (`COGHasAnalyzingEvaluatingWord`, etc.)
- Body part mentions (`MyBPM`, `HasBPM`, etc.)

See [DATASET.md](DATASET.md) for complete feature documentation.

## Testing

```bash
# Run tests
uv run pytest

# Run with timeout
uv run pytest --timeout=120
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{wahle2025affect,
  title={Affect, Body, Cognition, Demographics, and Emotion: The ABCDE of Text Features for Computational Affective Science},
  author={Wahle, Jan Philip and Vishnubhotla, Krishnapriya and Gipp, Bela and Mohammad, Saif M},
  journal={arXiv preprint arXiv:2512.17752},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NRC lexicons by Dr. Saif Mohammad
- UniMorph project for tense data
- GeoNames for city data
- Pushshift for Reddit archives
