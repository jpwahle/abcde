#!/usr/bin/env python3
"""
Backward-compatible helpers module.

This module re-exports all functionality from the new package structure
to maintain compatibility with existing code.

For new code, please import directly from the abcde package:
    from abcde import SelfIdentificationDetector, apply_linguistic_features
    from abcde.core import PIIDetector
    from abcde.io import write_results_to_csv
"""

# Re-export everything from the new package structure
from abcde.utils.banner import print_banner
from abcde.utils.dates import parse_tusc_created_at_year
from abcde.utils.text import clean_text_newlines, build_term_group

from abcde.lexicons.loaders import (
    get_lexicons_dir,
    load_nrc_vad_lexicon as _load_nrc_vad_lexicon,
    load_nrc_emotion_lexicon as _load_nrc_emotion_lexicon,
    load_nrc_worrywords_lexicon as _load_nrc_worrywords_lexicon,
    load_nrc_moraltrust_lexicon as _load_nrc_moraltrust_lexicon,
    load_nrc_socialwarmth_lexicon as _load_nrc_socialwarmth_lexicon,
    load_nrc_warmth_lexicon as _load_nrc_warmth_lexicon,
    load_eng_tenses_lexicon as _load_eng_tenses_lexicon,
    load_cog_lexicon as _load_cog_lexicon,
    load_body_parts,
    load_dmg_countries as _load_dmg_countries,
    load_dmg_genders as _load_dmg_genders,
    load_dmg_cities as _load_dmg_cities,
    load_dmg_religions as _load_dmg_religions,
    load_dmg_occupations as _load_dmg_occupations,
)

from abcde.core.detector import (
    SelfIdentificationDetector,
    detect_self_identification_in_entry,
    detect_self_identification_with_mappings_in_entry,
    detect_self_identification_with_resolved_age,
    detect_self_identification_in_tusc_entry,
    detect_self_identification_in_tusc_entry_with_mappings,
)

from abcde.core.pii import (
    PIIDetector,
    detect_pii_in_post,
)

from abcde.core.features import (
    apply_linguistic_features,
    compute_vad_and_emotions,
    compute_individual_pronouns,
    compute_prefixed_body_part_mentions,
    compute_tense_features,
    compute_cognitive_features,
    compute_embodied_cognitive_verbs,
    DEFAULT_COGNITIVE_VERBS,
)

from abcde.io.csv_utils import (
    flatten_result_to_csv_row,
    get_csv_fieldnames,
    ensure_output_directory,
    write_results_to_csv,
    append_results_to_csv,
)

from abcde.io.jsonl_utils import (
    get_all_jsonl_files,
    filter_entry,
    extract_columns,
)

from abcde.io.demographics import (
    format_demographic_detections_for_output,
    aggregate_user_demographics,
)

# Legacy variable names for backward compatibility
# These are loaded lazily when first accessed
_lexicons_loaded = False
vad_dict = None
emotion_dict = None
worry_dict = None
moraltrust_dict = None
socialwarmth_dict = None
warmth_dict = None
tense_dict = None
cog_dict = None
emotions = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "negative",
    "positive",
    "sadness",
    "surprise",
    "trust",
]
BODY_PARTS = None

# Demographic data
dmg_countries = None
dmg_genders = None
dmg_city_to_country = None
dmg_religion_to_main = None
dmg_religion_to_category = None
dmg_occupation_to_soc = None
dmg_nationalities = None
dmg_cities = None
dmg_religions = None
dmg_religion_adherents = None
dmg_occupations = None


def _ensure_legacy_data_loaded():
    """Load legacy module-level variables for backward compatibility."""
    global _lexicons_loaded
    global vad_dict, emotion_dict, worry_dict, moraltrust_dict
    global socialwarmth_dict, warmth_dict, tense_dict, cog_dict, BODY_PARTS
    global dmg_countries, dmg_genders, dmg_city_to_country
    global dmg_religion_to_main, dmg_religion_to_category, dmg_occupation_to_soc
    global dmg_nationalities, dmg_cities, dmg_religions, dmg_religion_adherents
    global dmg_occupations

    if _lexicons_loaded:
        return

    vad_dict = _load_nrc_vad_lexicon()
    emotion_dict = _load_nrc_emotion_lexicon()
    worry_dict = _load_nrc_worrywords_lexicon()
    moraltrust_dict = _load_nrc_moraltrust_lexicon()
    socialwarmth_dict = _load_nrc_socialwarmth_lexicon()
    warmth_dict = _load_nrc_warmth_lexicon()
    tense_dict = _load_eng_tenses_lexicon()
    cog_dict = _load_cog_lexicon()
    BODY_PARTS = load_body_parts()

    dmg_countries = _load_dmg_countries()
    dmg_genders = _load_dmg_genders()
    dmg_city_to_country = _load_dmg_cities()
    dmg_religion_to_main, dmg_religion_to_category = _load_dmg_religions()
    dmg_occupation_to_soc = _load_dmg_occupations()

    # Build derived data
    from abcde.core.detector import _build_nationalities_from_countries, _build_religion_adherents
    dmg_nationalities = _build_nationalities_from_countries(dmg_countries)
    dmg_cities = list(dmg_city_to_country.keys())
    dmg_religions = list(
        set(list(dmg_religion_to_main.keys()) + list(dmg_religion_to_main.values()))
    )
    dmg_religion_adherents = _build_religion_adherents(dmg_religions)
    dmg_occupations = list(dmg_occupation_to_soc.keys())

    _lexicons_loaded = True


# For scripts that access these module-level variables directly,
# we need to load them immediately
# Note: This can be slow on import, so new code should use lazy loading
try:
    _ensure_legacy_data_loaded()
except Exception:
    # If lexicons aren't available (e.g., during installation), skip
    pass


__all__ = [
    # Banner
    "print_banner",
    # Dates
    "parse_tusc_created_at_year",
    # Text
    "clean_text_newlines",
    "build_term_group",
    # Detector
    "SelfIdentificationDetector",
    "detect_self_identification_in_entry",
    "detect_self_identification_with_mappings_in_entry",
    "detect_self_identification_with_resolved_age",
    "detect_self_identification_in_tusc_entry",
    "detect_self_identification_in_tusc_entry_with_mappings",
    # PII
    "PIIDetector",
    "detect_pii_in_post",
    # Features
    "apply_linguistic_features",
    "compute_vad_and_emotions",
    "compute_individual_pronouns",
    "compute_prefixed_body_part_mentions",
    "compute_tense_features",
    "compute_cognitive_features",
    "compute_embodied_cognitive_verbs",
    "DEFAULT_COGNITIVE_VERBS",
    # CSV
    "flatten_result_to_csv_row",
    "get_csv_fieldnames",
    "ensure_output_directory",
    "write_results_to_csv",
    "append_results_to_csv",
    # JSONL
    "get_all_jsonl_files",
    "filter_entry",
    "extract_columns",
    # Demographics
    "format_demographic_detections_for_output",
    "aggregate_user_demographics",
    # Body parts
    "load_body_parts",
    "BODY_PARTS",
    # Legacy lexicon variables
    "vad_dict",
    "emotion_dict",
    "worry_dict",
    "moraltrust_dict",
    "socialwarmth_dict",
    "warmth_dict",
    "tense_dict",
    "cog_dict",
    "emotions",
    # Legacy demographic data
    "dmg_countries",
    "dmg_genders",
    "dmg_city_to_country",
    "dmg_religion_to_main",
    "dmg_religion_to_category",
    "dmg_occupation_to_soc",
    "dmg_nationalities",
    "dmg_cities",
    "dmg_religions",
    "dmg_religion_adherents",
    "dmg_occupations",
]
