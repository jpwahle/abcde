"""Lexicon loading utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import pandas as pd


def get_lexicons_dir() -> Path:
    """Get the lexicons directory path.
    
    Searches for the lexicons directory in multiple locations:
    1. Environment variable ABCDE_LEXICONS_DIR
    2. Relative to package: ../../lexicons (when installed as package)
    3. Relative to CWD: lexicons/
    4. Legacy: data/ directory
    """
    # Check environment variable first
    env_path = os.environ.get("ABCDE_LEXICONS_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Check relative to this file (package installation)
    pkg_path = Path(__file__).parent.parent.parent.parent / "lexicons"
    if pkg_path.exists():
        return pkg_path

    # Check relative to CWD
    cwd_path = Path.cwd() / "lexicons"
    if cwd_path.exists():
        return cwd_path

    # Legacy: check for data/ directory
    legacy_path = Path.cwd() / "data"
    if legacy_path.exists():
        return legacy_path

    # Default fallback
    return Path("lexicons")


_LEXICONS_DIR: Path | None = None


def _get_lexicons_dir() -> Path:
    """Get cached lexicons directory."""
    global _LEXICONS_DIR
    if _LEXICONS_DIR is None:
        _LEXICONS_DIR = get_lexicons_dir()
    return _LEXICONS_DIR


def _safe_read(path: Path) -> List[str]:
    """Safely read a file and return lines."""
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _load_lexicon(
    filename: str,
    key_col: int = 0,
    value_col: int = 2,
    skip_header: bool = False,
    value_type: str = "int",
    key_transform: Callable[[str], str] = lambda x: x.lower(),
    value_transform: Callable[[str], Any] = lambda x: x,
    accumulate: bool = False,
) -> Dict[str, Any]:
    """Generic lexicon loading function."""
    lexicons_dir = _get_lexicons_dir()
    lines = _safe_read(lexicons_dir / filename)
    if skip_header and lines:
        lines = lines[1:]
    result: Any
    if value_type in ("set", "list") and accumulate:
        from collections import defaultdict

        if value_type == "set":
            result = defaultdict(set)
        else:
            result = defaultdict(list)
    else:
        result = {}
    for line in lines:
        if not line or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) <= max(key_col, value_col):
            continue
        key = key_transform(parts[key_col])
        raw = value_transform(parts[value_col])
        if value_type == "int":
            val = int(raw)
        elif value_type == "float":
            val = float(raw)
        elif value_type == "str":
            val = str(raw)
        elif value_type == "set" and accumulate:
            if len(parts) > value_col + 1 and int(parts[value_col + 1]) == 1:
                result[key].add(raw)
            continue
        elif value_type == "list" and accumulate:
            result.setdefault(key, []).append(raw)
            continue
        else:
            val = raw
        result[key] = val
    return result


def load_nrc_vad_lexicon() -> Dict[str, Dict[str, float]]:
    """Load NRC VAD (Valence-Arousal-Dominance) lexicon."""
    lexicons_dir = _get_lexicons_dir()
    vad_dict: Dict[str, Dict[str, float]] = {}
    for line in _safe_read(lexicons_dir / "NRC-VAD-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        w, v, a, d = line.split("\t")
        vad_dict[w.lower()] = {
            "valence": float(v),
            "arousal": float(a),
            "dominance": float(d),
        }
    return vad_dict


def load_nrc_emotion_lexicon() -> Dict[str, set]:
    """Load NRC Emotion lexicon."""
    from collections import defaultdict

    lexicons_dir = _get_lexicons_dir()
    em: Dict[str, set] = defaultdict(set)
    for line in _safe_read(lexicons_dir / "NRC-Emotion-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        w, emo, flag = line.split("\t")
        if int(flag) == 1:
            em[w.lower()].add(emo)
    return em


def load_nrc_worrywords_lexicon() -> Dict[str, int]:
    """Load NRC WorryWords (Anxiety/Calmness) lexicon."""
    return _load_lexicon(
        "NRC-WorryWords-Lexicon.txt", skip_header=True, value_type="int"
    )


def load_nrc_moraltrust_lexicon() -> Dict[str, int]:
    """Load NRC Moral Trustworthiness lexicon."""
    return _load_lexicon(
        "NRC-MoralTrustworthy-Lexicon.txt", skip_header=True, value_type="int"
    )


def load_nrc_socialwarmth_lexicon() -> Dict[str, int]:
    """Load NRC Social Warmth lexicon."""
    return _load_lexicon(
        "NRC-SocialWarmth-Lexicon.txt", skip_header=True, value_type="int"
    )


def load_nrc_warmth_lexicon() -> Dict[str, int]:
    """Load NRC Combined Warmth lexicon."""
    return _load_lexicon(
        "NRC-CombinedWarmth-Lexicon.txt", skip_header=True, value_type="int"
    )


def load_eng_tenses_lexicon() -> Dict[str, List[str]]:
    """Load English word tenses lexicon."""
    return _load_lexicon(
        "TIME-eng-word-tenses.txt",
        key_col=1,
        value_col=2,
        value_type="list",
        accumulate=True,
    )


def load_cog_lexicon() -> Dict[str, Set[str]]:
    """Load cognitive/thinking words lexicon."""
    lexicons_dir = _get_lexicons_dir()
    path = lexicons_dir / "COG-thinking-words-categorized.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cog_dict: Dict[str, Set[str]] = {}
    for item in data:
        cat = (
            item["category"]
            .replace(" ", "")
            .replace("&", "")
            .replace("/", "")
            .replace(",", "")
        )
        terms = {term.lower() for term in item["terms"]}
        cog_dict[cat] = terms
    return cog_dict


def load_body_parts(filepath: str | None = None) -> List[str]:
    """Load body parts word list."""
    if filepath is None:
        lexicons_dir = _get_lexicons_dir()
        filepath = str(lexicons_dir / "BPM-bodywords-full.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip()]


def load_dmg_countries() -> List[str]:
    """Load country names from DMG-country-list.txt."""
    lexicons_dir = _get_lexicons_dir()
    countries = []
    for line in _safe_read(lexicons_dir / "DMG-country-list.txt"):
        if line:
            countries.append(line)
    return countries


def load_dmg_genders() -> List[str]:
    """Load gender terms from DMG-gender-list.txt."""
    lexicons_dir = _get_lexicons_dir()
    genders = []
    for line in _safe_read(lexicons_dir / "DMG-gender-list.txt"):
        if line:
            genders.append(line)
    return genders


def load_dmg_cities(max_cities: int = 50000) -> Dict[str, str]:
    """Load cities from DMG-geonames CSV and create city->country mapping.
    
    Args:
        max_cities: Maximum number of cities to load (by population).
                   Defaults to 50000 to keep regex patterns manageable.
    """
    lexicons_dir = _get_lexicons_dir()
    city_to_country = {}

    csv_path = lexicons_dir / "DMG-geonames-all-cities-with-a-population-1000.csv"
    try:
        # Read CSV with semicolon separator
        df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False)

        # Sort by population (descending) to prioritize larger cities
        df["Population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(0)
        df_sorted = df.sort_values("Population", ascending=False)
        
        # Limit to top N cities by population to keep regex manageable
        if max_cities and len(df_sorted) > max_cities:
            df_sorted = df_sorted.head(max_cities)

        # Create mapping from city name to country
        for _, row in df_sorted.iterrows():
            city_name = row.get("Name", "")
            country_name = row.get("Country name EN", "")

            # Handle potential float/NaN values
            if pd.isna(city_name) or pd.isna(country_name):
                continue

            city_name = str(city_name).strip()
            country_name = str(country_name).strip()

            if city_name and country_name:
                # Only store if not already present (larger cities take precedence)
                city_lower = city_name.lower()
                if city_lower not in city_to_country:
                    city_to_country[city_lower] = country_name

                # Also store alternate names if available (limit to first 5 to avoid explosion)
                alt_names = row.get("Alternate Names", "")
                if alt_names and isinstance(alt_names, str) and not pd.isna(alt_names):
                    for i, alt_name in enumerate(alt_names.split(",")):
                        if i >= 5:  # Limit alternate names per city
                            break
                        alt_name = alt_name.strip()
                        if alt_name and alt_name.lower() not in city_to_country:
                            city_to_country[alt_name.lower()] = country_name
    except Exception as e:
        print(f"Error loading city data: {e}")

    return city_to_country


def load_dmg_religions() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load religions from DMG-religion-list.csv and create mappings.
    
    Returns:
        - religion_to_main: Maps substrain -> main religion
        - religion_to_category: Maps substrain -> main category
    """
    lexicons_dir = _get_lexicons_dir()
    religion_to_main = {}
    religion_to_category = {}

    csv_path = lexicons_dir / "DMG-religion-list.csv"
    try:
        df = pd.read_csv(csv_path, dtype=str)

        for _, row in df.iterrows():
            category = row.get("Main Category", "")
            main_religion = row.get("Main Religion", "")
            substrain = row.get("Substrain/Denomination", "") or row.get(
                "Substrain", ""
            )

            # Handle potential float/NaN values
            category = str(category).strip() if pd.notna(category) else ""
            main_religion = (
                str(main_religion).strip() if pd.notna(main_religion) else ""
            )
            substrain = str(substrain).strip() if pd.notna(substrain) else ""

            if substrain:
                substrain_lower = substrain.lower()
                # Map substrain to main religion
                if main_religion:
                    religion_to_main[substrain_lower] = main_religion
                # Map substrain to category
                if category:
                    religion_to_category[substrain_lower] = category

            # Also add main religion as a key pointing to itself
            if main_religion:
                main_lower = main_religion.lower()
                religion_to_main[main_lower] = main_religion
                if category:
                    religion_to_category[main_lower] = category

    except Exception as e:
        print(f"Error loading religion data: {e}")

    return religion_to_main, religion_to_category


def load_dmg_occupations() -> Dict[str, str]:
    """Load occupations from Excel file and create direct match -> SOC title mapping."""
    lexicons_dir = _get_lexicons_dir()
    occupation_to_soc = {}

    xlsx_path = lexicons_dir / "DMG-soc_2018_direct_match_title_file.xlsx"
    try:
        df = pd.read_excel(xlsx_path, dtype=str)

        # Find columns (they might have different names)
        direct_match_col = None
        soc_title_col = None

        for col in df.columns:
            if "Direct Match Title" in col:
                direct_match_col = col
            elif "SOC Title" in col or "Occupation" in col:
                soc_title_col = col

        if direct_match_col and soc_title_col:
            for _, row in df.iterrows():
                direct_match = row.get(direct_match_col, "").strip()
                soc_title = row.get(soc_title_col, "").strip()

                if direct_match and soc_title:
                    occupation_to_soc[direct_match.lower()] = soc_title

    except Exception as e:
        print(f"Error loading occupation data: {e}")

    return occupation_to_soc

