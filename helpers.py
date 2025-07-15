#!/usr/bin/env python3
"""
Shared helper functions for Reddit and TUSC data processing.
Contains functions for self-identification detection, linguistic feature computation,
flattening results, and simple I/O utilities.
"""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Set

import nltk
import pandas as pd
from nltk.corpus import stopwords
from presidio_analyzer import AnalyzerEngine
from collections import Counter


def print_banner() -> None:
    """Print the ABCDE startup banner."""
    banner = """
 █████╗ ██████╗  ██████╗██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝
███████║██████╔╝██║     ██║  ██║█████╗  
██╔══██║██╔══██╗██║     ██║  ██║██╔══╝  
██║  ██║██████╔╝╚██████╗██████╔╝███████╗
╚═╝  ╚═╝╚═════╝  ╚═════╝╚═════╝ ╚══════╝

"""
    print(banner)


# Download stopwords if not already available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# -------------------- #
# Date Parsing Helpers
# -------------------- #


def parse_tusc_created_at_year(created_at: str) -> Optional[int]:
    """Parse TUSC createdAt field and extract year from either format:
    - "Wed Apr 01 14:35:59 +0000 2020"
    - "2015-04-10T02:47:38.000Z"

    Returns None if parsing fails.
    """
    if not isinstance(created_at, str) or not created_at.strip():
        return None

    created_at = created_at.strip()

    # Try format 1: "Wed Apr 01 14:35:59 +0000 2020"
    try:
        dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        return dt.year
    except (ValueError, TypeError):
        pass

    # Try format 2: "2015-04-10T02:47:38.000Z"
    try:
        # Handle both with and without milliseconds
        if created_at.endswith("Z"):
            if "." in created_at:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        else:
            # Handle ISO format without Z
            if "." in created_at:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S")
        return dt.year
    except (ValueError, TypeError):
        pass

    return None


# -------------------- #
# Self-Identification Detection
# -------------------- #


class SelfIdentificationDetector:
    """Detect self-identification statements (age, etc.) inside free text.

    The detector is data-source agnostic – it expects plain strings and can be
    reused for Reddit, Twitter, blogs, or any other textual resource.
    """

    def __init__(self) -> None:
        # Use NLTK stopwords for English
        self.stopwords = set(stopwords.words("english"))

        # Store valid cities set for validation
        self.valid_cities = set()
        self.valid_cities_lower = set()

        # Build term groups from loaded data
        # For patterns that capture the term directly, use capturing=True
        OCCUPATIONS_TERMS_REGEX = build_term_group(dmg_occupations, capturing=True)
        GENDERS_TERMS_REGEX = build_term_group(dmg_genders, capturing=True)
        COUNTRIES_TERMS_REGEX = build_term_group(dmg_countries, capturing=True)
        NATIONALITIES_TERMS_REGEX = build_term_group(dmg_nationalities, capturing=True)

        # Filter cities to remove common words and countries
        filtered_cities = self._filter_city_list(dmg_cities, dmg_countries)
        # Store valid cities for validation
        self.valid_cities = set(filtered_cities)
        self.valid_cities_lower = {city.lower() for city in filtered_cities}
        CITIES_TERMS_REGEX = build_term_group(filtered_cities, capturing=True)

        RELIGIONS_ADHERENTS_TERMS_REGEX = build_term_group(
            dmg_religion_adherents, capturing=True
        )
        RELIGIONS_NAMES_TERMS_REGEX = build_term_group(dmg_religions, capturing=True)

        self.patterns: Dict[str, List[Pattern[str]]] = {
            "age": [
                # Pattern 1: "I am/I'm X years old" (explicit age statement)
                re.compile(r"\bI(?:\s+am|'m)\s+(\d{1,2})\s+years?\s+old\b", re.I),
                # Pattern 2: "I am/I'm X" followed by end of string, punctuation, or age-related conjunctions
                re.compile(
                    r"\bI(?:\s+am|'m)\s+(\d{1,2})"
                    r"(?=\s*(?:$|[,.!?]|(?:and|but|so|yet)\s))",
                    re.I,
                ),
                # Pattern 3: "I was/am born in YYYY" (four-digit year)
                re.compile(
                    r"\bI(?:\s+was|\s+am|'m)\s+born\s+in\s+"
                    r"(19\d{2}|20(?:0\d|1\d|2[0-4]))\b",
                    re.I,
                ),
                # Pattern 4: "I was/am born in 'YY" (two-digit year with apostrophe)
                re.compile(r"\bI(?:\s+was|\s+am|'m)\s+born\s+in\s+'(\d{2})\b", re.I),
                # Pattern 5: "I was born on DD Month YYYY" (full date format)
                re.compile(
                    r"\bI\s+was\s+born\s+on\s+"
                    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
                    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+"
                    r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?"
                    r"(19\d{2}|20(?:0\d|1\d|2[0-4]))\b",
                    re.I,
                ),
                # Pattern 6: "I was born on MM/DD/YYYY" or similar date formats
                re.compile(
                    r"\bI\s+was\s+born\s+on\s+"
                    r"\d{1,2}[/\-]\d{1,2}[/\-](19\d{2}|20(?:0\d|1\d|2[0-4]))\b",
                    re.I,
                ),
            ],
            "occupation": [
                # Pattern 1 (from L1): "I am/I'm (a/an/the) [Occupation] (at [Company]) (followed by punctuation/conjunction)"
                # Includes negative lookaheads and specific following context.
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+looking\s+for)\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"(?:\s+at\s+[\w\s.-]+)?(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|while|when|because)\b))",
                    re.I,
                ),
                # Pattern 2 (Variant of L1P1, inspired by L2P1's simpler \b ending):
                # Catches cases where L1P1's lookahead is too restrictive (e.g., "I'm a teacher who...").
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+looking\s+for)\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"(?:\s+at\s+[\w\s.-]+)?\b",  # Simpler \b termination
                    re.I,
                ),
                # Pattern 3 (from L1, covers L2P2): "I work as (a/an/the) [Occupation] (at [Company])"
                # More comprehensive than L2P2 (includes "the", optional company).
                re.compile(
                    r"\bI\s+work\s+as\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"(?:\s+at\s+[\w\s.-]+)?\b",
                    re.I,
                ),
                # Pattern 4 (from L1, covers L2P3): "My job/occupation/profession/role is (to be) (a/an/as) [Occupation]"
                # More comprehensive (includes "role", "to be", "as").
                re.compile(
                    r"\bMy\s+(?:job|occupation|profession|role)\s+is\s+(?:to\s+be\s+)?(?:a|an|as)?\s*"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 5 (from L1): "I'm (currently) employed as (a/an/the) [Occupation]"
                re.compile(
                    r"\bI'm\s+(?:currently\s+)?employed\s+as\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "gender": [
                # Pattern 1 (from L1): "I am/I'm (a/an) [Gender] (followed by punctuation/conjunction)"
                # Includes negative lookahead and specific following context.
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an)?\s+"
                    + GENDERS_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|because)\b))",
                    re.I,
                ),
                # Pattern 2 (Variant of L1P1, inspired by L2P1's simpler \b ending):
                # Catches cases where L1P1's lookahead is too restrictive. Covers L2P1.
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an)?\s+"
                    + GENDERS_TERMS_REGEX
                    + r"\b",  # Simpler \b termination
                    re.I,
                ),
                # Pattern 3 (from L1, covers L2P3): "I identify as (a/an) [Gender]"
                # More comprehensive with optional (a/an).
                re.compile(
                    r"\bI\s+identify\s+as\s+(?:a|an)?\s*" + GENDERS_TERMS_REGEX + r"\b",
                    re.I,
                ),
                # Pattern 4 (from L1, covers L2P2): "My gender (identity) is [Gender]"
                # Identical to L2P2 given re.I.
                re.compile(
                    r"\bMy\s+gender(?:\s+identity)?\s+is\s+"
                    + GENDERS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 5 (from L1): "I'm/I am (a) (transgender/trans) [Gender like man/woman]"
                re.compile(
                    r"\b(?:I'm|I\s+am)\s+(?:a\s+)?(?:transgender|trans)\s+"
                    + GENDERS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "country": [
                # Pattern 1 (from L1): "I am/I'm from (the) [Country]" (current primary location focus)
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+just\s+visiting|\s+originally)\s+from\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 2 (Enhanced L2P1 - simpler version of L1P1 without negative lookaheads but with (the)):
                re.compile(
                    r"\bI(?:\s+am|'m)\s+from\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 3 (from L1, `an?` corrected): "I am/I'm (a/an/the) [Nationality] (followed by punctuation/conjunction)"
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an|the)?\s*"
                    + NATIONALITIES_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|who)\b))",
                    re.I,
                ),
                # Pattern 4 (Variant of L1P3 with `\b` end, covers L2P4 for nationalities):
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an|the)?\s*"
                    + NATIONALITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 5 (Combined L1P3 & L2P3, adding 'at' and '(the)'): "I live in/at (the) [Country]"
                re.compile(
                    r"\bI\s+live\s+(?:in|at)\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 6 (from L2P2, enhanced with `(?:the\s+)?`): "I come from (the) [Country]"
                re.compile(
                    r"\bI\s+come\s+from\s+(?:the\s+)?" + COUNTRIES_TERMS_REGEX + r"\b",
                    re.I,
                ),
                # Pattern 7 (from L1): "My nationality/citizenship is [Nationality]"
                re.compile(
                    r"\bMy\s+(?:nationality|citizenship)\s+is\s+"
                    + NATIONALITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 8 (from L1): "I was/am born and raised in (the) [Country]"
                re.compile(
                    r"\bI(?:\s+was|'m)\s+born\s+and\s+(?:raised|grew\s+up)\s+in\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 9 (from L1): "I am/I'm originally from (the) [Country]" (origin focus)
                re.compile(
                    r"\bI(?:\s+am|'m)\s+originally\s+from\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "city": [
                # Pattern 1 (from L1): "I am/I'm from [City] (followed by punctuation/conjunction)"
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+just\s+visiting|\s+originally)\s+from\s+"
                    + CITIES_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet)\b))",
                    re.I,
                ),
                # Pattern 2 (Variant of L1P1, inspired by L2P1's simpler \b ending, covers L2P1):
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+just\s+visiting|\s+originally)\s+from\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 3 (Combined L1P2 & L2P2, adding 'at' and explicit \b before FP lookahead): "I live in/at [City]"
                re.compile(
                    r"\bI\s+live\s+(?:in|at)\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b(?!\s+(?:fear|hope|a\s+state\s+of|sin|poverty|luxury))",
                    re.I,
                ),
                # Pattern 4 (from L1, covers L2P4): "I'm/I am (currently) residing/based in [City]"
                re.compile(
                    r"\b(?:I'm\s+|I\s+am\s+)?(?:currently\s+)?(?:residing|based)\s+in\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 5 (from L1): "My (current/home) city/town is [City]"
                re.compile(
                    r"\bMy\s+(?:current\s+|home\s+)?(?:city|town)\s+is\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",  # Corrected 'home '
                    re.I,
                ),
                # Pattern 6 (from L1, covers L2P3): "I (grew up / was raised) in [City]"
                re.compile(
                    r"\bI\s+(?:grew\s+up|was\s+raised)\s+in\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "religion": [
                # Pattern 1 (from L1): "I am/I'm (a/an/a devout/a practicing) [Religion Adherent] (followed by context)"
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+sure\s+if\s+I'm)\s+(?:a|an|a\s+devout|a\s+practicing)?\s+"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|who)\b))",
                    re.I,
                ),
                # Pattern 2 (Variant of L1P1 with `\b` end, covers L2P1 for adherents):
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+sure\s+if\s+I'm)\s+(?:a|an|a\s+devout|a\s+practicing)?\s+"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 3 (Combined L1P2 & L2P2, adding 'faith'): "My religion/faith is [Religion Name]"
                re.compile(
                    r"\bMy\s+(?:religion|faith)\s+is\s+"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 4 (from L1, covers L2P3 for names): "I (actively) practice [Religion Name]"
                re.compile(
                    r"\bI\s+(?:actively\s+)?practice\s+"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 5 (from L1): "I am/I'm a follower of [Religion Name]"
                re.compile(
                    r"\bI(?:\s+am|'m)\s+a\s+follower\s+of\s+"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 6 (from L1): "I converted to [Religion Name]"
                re.compile(
                    r"\bI\s+converted\s+to\s+" + RELIGIONS_NAMES_TERMS_REGEX + r"\b",
                    re.I,
                ),
                # Pattern 7 (Combined L1P6 & L2P4, `a(n?)` corrected, adds 'born'): "I was raised/born (as a/an) [Religion Adherent]"
                re.compile(
                    r"\bI\s+was\s+(?:raised|born)\s+(?:as\s+(?:a|an))?\s*"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                # Pattern 8 (from L1, covers L2P5 for adherents): "I identify as [Adherent] / identify with [Religion Name]"
                re.compile(
                    r"\bI\s+identify\s+(?:as\s+"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"|with\s+(?:the\s+)?"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r")\b",
                    re.I,
                ),
            ],
        }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Return category → unique matched strings for self-identification."""
        if not isinstance(text, str) or not text:
            return {}
        text = text.strip()
        matches: Dict[str, List[str]] = {}
        for category, regs in self.patterns.items():
            cat_matches: List[str] = []
            for reg in regs:
                for m in reg.finditer(text):
                    cat_matches.append(m.group(1) if m.groups() else m.group(0))
            if cat_matches:
                uniq: List[str] = []
                for cm in cat_matches:
                    if cm not in uniq:
                        uniq.append(cm)
                matches[category] = uniq
        return matches

    def _filter_city_list(self, cities: List[str], countries: List[str]) -> List[str]:
        """Filter city list to remove common words and country names.

        Args:
            cities: List of city names
            countries: List of country names

        Returns:
            Filtered list of cities
        """
        filtered = []
        countries_lower = {c.lower() for c in countries}

        for city in cities:
            city_lower = city.lower()

            # Skip if it's a stopword
            if city_lower in self.stopwords:
                continue

            # Skip if it's a country name
            if city_lower in countries_lower:
                continue

            # Skip very short "cities" (less than 4 characters)
            if len(city_lower) <= 4:
                continue

            filtered.append(city)

        return filtered

    def detect_with_mappings(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Return demographic detections with both raw extractions and mapped values.

        Returns a dict with structure:
        {
            "city": {
                "raw": ["austin", "new york"],
                "country_mapped": ["United States", "United States"]
            },
            "religion": {
                "raw": ["catholic", "buddhist"],
                "main_religion_mapped": ["Christianity", "Buddhism"],
                "category_mapped": ["Abrahamic Religions", "Eastern Religions"]
            },
            "occupation": {
                "raw": ["software engineer", "teacher"],
                "soc_mapped": ["Software Developers", "Elementary School Teachers"]
            }
        }
        """
        raw_matches = self.detect(text)
        result = {}

        for category, matches in raw_matches.items():
            # Lowercase all raw matches before further processing
            lower_matches = [match.lower() for match in matches if match is not None]
            result[category] = {"raw": lower_matches}

            if category == "city":
                # Map cities to countries - validate against known cities
                country_mapped = []
                validated_cities = []
                # Use lower_matches for validation
                for city in lower_matches:
                    # Only include if it's a valid city in our geonames data
                    if city in self.valid_cities_lower and city in dmg_city_to_country:
                        validated_cities.append(city)
                        country_mapped.append(dmg_city_to_country[city])
                # Update matches to only include validated, lowercased cities
                result[category]["raw"] = validated_cities
                result[category]["country_mapped"] = country_mapped

            elif category == "religion":
                # Map religions to main religion and category
                main_religion_mapped = []
                category_mapped = []
                for religion in lower_matches:
                    # religion is already lowercased

                    # Try direct mapping first
                    if religion in dmg_religion_to_main:
                        main_religion_mapped.append(dmg_religion_to_main[religion])
                    # Try adding 'ism' for adherent forms (e.g., Catholic -> Catholicism)
                    elif religion + "ism" in dmg_religion_to_main:
                        main_religion_mapped.append(
                            dmg_religion_to_main[religion + "ism"]
                        )
                    # Try adding 'ity' for some forms (e.g., Christian -> Christianity)
                    elif religion + "ity" in dmg_religion_to_main:
                        main_religion_mapped.append(
                            dmg_religion_to_main[religion + "ity"]
                        )
                    # Special case for atheist/agnostic (map to atheism/agnosticism)
                    elif religion == "atheist" and "atheism" in dmg_religion_to_main:
                        main_religion_mapped.append(dmg_religion_to_main["atheism"])
                    elif (
                        religion == "agnostic" and "agnosticism" in dmg_religion_to_main
                    ):
                        main_religion_mapped.append(dmg_religion_to_main["agnosticism"])
                    else:
                        main_religion_mapped.append(None)

                    # Same logic for category mapping
                    if religion in dmg_religion_to_category:
                        category_mapped.append(dmg_religion_to_category[religion])
                    elif religion + "ism" in dmg_religion_to_category:
                        category_mapped.append(
                            dmg_religion_to_category[religion + "ism"]
                        )
                    elif religion + "ity" in dmg_religion_to_category:
                        category_mapped.append(
                            dmg_religion_to_category[religion + "ity"]
                        )
                    # Special case for atheist/agnostic (map to atheism/agnosticism)
                    elif (
                        religion == "atheist" and "atheism" in dmg_religion_to_category
                    ):
                        category_mapped.append(dmg_religion_to_category["atheism"])
                    elif (
                        religion == "agnostic"
                        and "agnosticism" in dmg_religion_to_category
                    ):
                        category_mapped.append(dmg_religion_to_category["agnosticism"])
                    else:
                        category_mapped.append(None)

                result[category]["main_religion_mapped"] = main_religion_mapped
                result[category]["category_mapped"] = category_mapped

            elif category == "occupation":
                # Map occupations to SOC titles
                soc_mapped = []
                for occupation in lower_matches:
                    if occupation in dmg_occupation_to_soc:
                        soc_mapped.append(dmg_occupation_to_soc[occupation])
                    else:
                        soc_mapped.append(None)
                result[category]["soc_mapped"] = soc_mapped

        return result

    def resolve_multiple_ages(
        self,
        age_matches: List[str],
        current_year: Optional[int] = None,
    ) -> Optional[Tuple[int, float]]:
        """Resolve multiple age extractions using clustering and confidence scoring."""
        if not age_matches:
            return None

        birth_year_candidates: List[Tuple[int, float]] = []
        for age_str in age_matches:
            try:
                # Handle two-digit birth years like '85
                if age_str.startswith("'") and len(age_str) == 3 and age_str[1:].isdigit():
                    year_val = int(age_str[1:])
                    birth_year = 1900 + year_val if year_val > (current_year % 100) else 2000 + year_val
                    weight = 1.0  # High confidence for explicit birth years
                else:
                    age_val = int(age_str)
                    if 1900 <= age_val <= current_year:
                        birth_year, weight = age_val, 1.0
                    elif 13 <= age_val <= 99:
                        birth_year, weight = current_year - age_val, 0.8
                    else:
                        continue
                birth_year_candidates.append((birth_year, weight))
            except (ValueError, TypeError):
                continue

        if not birth_year_candidates:
            return None

        clusters: Dict[int, List[Tuple[int, float]]] = {}
        for by, wt in birth_year_candidates:
            key = next((k for k in clusters if abs(by - k) <= 2), None)
            if key is None:
                key = by
                clusters[key] = []
            clusters[key].append((by, wt))

        best_cluster = None
        best_score = 0.0
        for center, members in clusters.items():
            total_weight = sum(w for _, w in members)
            score = total_weight + len(members) * 0.1
            if score > best_score:
                best_score = score
                best_cluster = members

        if not best_cluster:
            return None

        total_weight = sum(w for _, w in best_cluster)
        weighted_year = sum(by * w for by, w in best_cluster) / total_weight
        resolved_age = current_year - int(round(weighted_year))
        confidence = min(1.0, best_score / (len(age_matches) * 1.0))

        if not (13 <= resolved_age <= 99):
            return None

        return resolved_age, confidence


def detect_self_identification_in_entry(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, List[str]]:
    """Detect self-identification in a Reddit-style entry (title+body)."""
    title = entry.get("title", "") or ""
    body = entry.get("selftext", "") or ""
    combined = f"{title} {body}".strip()
    return detector.detect(combined)


def detect_self_identification_with_mappings_in_entry(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, Dict[str, List[str]]]:
    """Detect self-identification with mappings in a Reddit-style entry (title+body).

    Returns both raw extractions and mapped values for city->country,
    religion->main/category, and occupation->SOC title.
    """
    title = entry.get("title", "") or ""
    body = entry.get("selftext", "") or ""
    combined = f"{title} {body}".strip()
    return detector.detect_with_mappings(combined)


def format_demographic_detections_for_output(
    detections: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Format demographic detections with user-specified field names.

    Converts the nested detection structure to flat fields with specific names:
    - DMGRawExtractedCity, DMGCountryMappedFromExtractedCity
    - DMGRawExtractedReligion, DMGMainReligionMappedFromExtractedReligion
    - DMGRawExtractedOccupation, DMGSOCTitleMappedFromExtractedOccupation
    """
    output = {}

    # City fields
    if "city" in detections:
        city_data = detections["city"]
        output["DMGRawExtractedCity"] = city_data.get("raw", [])
        output["DMGCountryMappedFromExtractedCity"] = city_data.get(
            "country_mapped", []
        )

    # Religion fields
    if "religion" in detections:
        religion_data = detections["religion"]
        output["DMGRawExtractedReligion"] = religion_data.get("raw", [])
        output["DMGMainReligionMappedFromExtractedReligion"] = religion_data.get(
            "main_religion_mapped", []
        )
        output["DMGMainCategoryMappedFromExtractedReligion"] = religion_data.get(
            "category_mapped", []
        )

    # Occupation fields
    if "occupation" in detections:
        occupation_data = detections["occupation"]
        output["DMGRawExtractedOccupation"] = occupation_data.get("raw", [])
        output["DMGSOCTitleMappedFromExtractedOccupation"] = occupation_data.get(
            "soc_mapped", []
        )

    # Other raw fields (age, gender, country)
    if "age" in detections:
        output["DMGRawExtractedAge"] = detections["age"].get("raw", [])

    if "gender" in detections:
        output["DMGRawExtractedGender"] = detections["gender"].get("raw", [])

    if "country" in detections:
        output["DMGRawExtractedCountry"] = detections["country"].get("raw", [])

    # City/Country conflict resolution
    # If both city and country are extracted, validate consistency
    if "city" in detections and "country" in detections:
        city_countries = output.get("DMGCountryMappedFromExtractedCity", [])
        raw_countries = output.get("DMGRawExtractedCountry", [])

        # If we have mapped countries from cities, prioritize those
        if city_countries and any(c is not None for c in city_countries):
            # Get the first non-None country from city mapping
            resolved_country = next((c for c in city_countries if c is not None), None)
            if resolved_country:
                # Check if raw country matches
                if raw_countries and resolved_country.lower() not in [
                    c.lower() for c in raw_countries if c
                ]:
                    # There's a conflict - prioritize city-based mapping
                    # Keep the city mapping as is, but note the conflict in raw country
                    pass

    # Normalize all DMG feature values to lowercase
    for k, v in output.items():
        if isinstance(v, list):
            output[k] = [s.lower() if isinstance(s, str) else s for s in v]
        elif isinstance(v, str):
            output[k] = v.lower()

    return output


def detect_self_identification_with_resolved_age(
    entry: Dict[str, Any], detector: "SelfIdentificationDetector"
) -> Dict[str, Any]:
    """Detect self identification with age resolution for multiple age extractions.

    Parameters
    ----------
    entry : Dict[str, Any]
        Reddit-style entry with title and body
    detector : SelfIdentificationDetector
        Detector instance

    Returns
    -------
    Dict[str, Any]
        Dictionary with original matches plus resolved_age info:
        {
            "age": ["25", "1998"],  # original extractions
            "resolved_age": {"age": 27, "confidence": 0.9, "raw_matches": ["25", "1998"]}
        }
    """
    matches = detect_self_identification_in_entry(entry, detector)

    # If age matches found, resolve them
    if "age" in matches:
        # Extract post year for age resolution
        if "Year" in entry:  # TUSC data with explicit Year field
            try:
                ref_year = int(entry.get("Year", ""))
            except (TypeError, ValueError):
                ref_year = None
        elif "createdAt" in entry:  # TUSC data with createdAt field
            ref_year = parse_tusc_created_at_year(entry.get("createdAt", ""))
        else:  # Reddit data
            try:
                ts = entry.get("created_utc") or entry.get("post", {}).get(
                    "created_utc"
                )
                ref_year = datetime.utcfromtimestamp(int(ts)).year
            except Exception:
                ref_year = None

        age_resolution = detector.resolve_multiple_ages(
            matches["age"], current_year=ref_year
        )

        if age_resolution is not None:
            resolved_age, confidence = age_resolution
            matches["resolved_age"] = {
                "age": resolved_age,
                "confidence": confidence,
                "raw_matches": matches["age"].copy(),
            }

    return matches


# -------------------- #
# Lexicon Loading and Feature Computation
# -------------------- #

_DATA_DIR = Path("data")


def _safe_read(path: Path) -> List[str]:
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
    lines = _safe_read(_DATA_DIR / filename)
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


def _load_nrc_vad_lexicon() -> Dict[str, Dict[str, float]]:
    vad_dict: Dict[str, Dict[str, float]] = {}
    for line in _safe_read(_DATA_DIR / "NRC-VAD-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        w, v, a, d = line.split("\t")
        vad_dict[w.lower()] = {
            "valence": float(v),
            "arousal": float(a),
            "dominance": float(d),
        }
    return vad_dict


def _load_nrc_emotion_lexicon() -> Dict[str, set]:
    from collections import defaultdict

    em: Dict[str, set] = defaultdict(set)
    for line in _safe_read(_DATA_DIR / "NRC-Emotion-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        w, emo, flag = line.split("\t")
        if int(flag) == 1:
            em[w.lower()].add(emo)
    return em


def _load_nrc_worrywords_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-WorryWords-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_eng_tenses_lexicon() -> Dict[str, List[str]]:
    return _load_lexicon(
        "TIME-eng-word-tenses.txt",
        key_col=1,
        value_col=2,
        value_type="list",
        accumulate=True,
    )


def _load_nrc_moraltrust_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-MoralTrustworthy-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_nrc_socialwarmth_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-SocialWarmth-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_nrc_warmth_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-CombinedWarmth-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_cog_lexicon() -> Dict[str, Set[str]]:
    import json
    path = _DATA_DIR / "COG-thinking-words-categorized.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cog_dict: Dict[str, Set[str]] = {}
    for item in data:
        cat = item["category"].replace(" ", "").replace("&", "").replace("/", "").replace(",", "")
        terms = {term.lower() for term in item["terms"]}
        cog_dict[cat] = terms
    return cog_dict


# -------------------- #
# Demographic Data Loading
# -------------------- #


def _load_dmg_countries() -> List[str]:
    """Load country names from DMG-country-list.txt."""
    countries = []
    for line in _safe_read(_DATA_DIR / "DMG-country-list.txt"):
        if line:
            countries.append(line)
    return countries


def _load_dmg_genders() -> List[str]:
    """Load gender terms from DMG-gender-list.txt."""
    genders = []
    for line in _safe_read(_DATA_DIR / "DMG-gender-list.txt"):
        if line:
            genders.append(line)
    return genders


def _load_dmg_cities() -> Dict[str, str]:
    """Load cities from DMG-geonames CSV and create city->country mapping."""
    city_to_country = {}

    csv_path = _DATA_DIR / "DMG-geonames-all-cities-with-a-population-1000.csv"
    try:
        # Read CSV with semicolon separator
        df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False)

        # Sort by population (descending) to prioritize larger cities
        df["Population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(0)
        df_sorted = df.sort_values("Population", ascending=False)

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

                # Also store alternate names if available
                alt_names = row.get("Alternate Names", "")
                if alt_names and isinstance(alt_names, str) and not pd.isna(alt_names):
                    for alt_name in alt_names.split(","):
                        alt_name = alt_name.strip()
                        if alt_name and alt_name.lower() not in city_to_country:
                            city_to_country[alt_name.lower()] = country_name
    except Exception as e:
        print(f"Error loading city data: {e}")

    return city_to_country


def _load_dmg_religions() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load religions from DMG-religion-list.csv and create mappings.
    Returns:
        - religion_to_main: Maps substrain -> main religion
        - religion_to_category: Maps substrain -> main category
    """
    religion_to_main = {}
    religion_to_category = {}

    csv_path = _DATA_DIR / "DMG-religion-list.csv"
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


def _load_dmg_occupations() -> Dict[str, str]:
    """Load occupations from Excel file and create direct match -> SOC title mapping."""
    occupation_to_soc = {}

    xlsx_path = _DATA_DIR / "DMG-soc_2018_direct_match_title_file.xlsx"
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


def build_term_group(terms: List[str], capturing: bool = False) -> str:
    """Build a regex group from a list of terms, escaping special characters.

    Args:
        terms: List of terms to build into a regex group
        capturing: If True, returns a capturing group (...), else non-capturing (?:...)
    """
    # Sort by length (longest first) to avoid partial matches
    sorted_terms = sorted(terms, key=len, reverse=True)

    # Escape special regex characters and join with |
    escaped_terms = []
    for term in sorted_terms:
        # Escape special regex characters
        escaped = re.escape(term)
        # Replace spaces with \s+ to match multiple spaces
        escaped = escaped.replace(r"\ ", r"\s+")
        escaped_terms.append(escaped)

    if capturing:
        return r"(" + "|".join(escaped_terms) + ")"
    else:
        return r"(?:" + "|".join(escaped_terms) + ")"


def _build_nationalities_from_countries(countries: List[str]) -> List[str]:
    """Generate nationality terms from country names using common patterns."""
    nationalities = []

    # Common country to nationality mappings
    special_mappings = {
        "united states": "american",
        "united states of america": "american",
        "usa": "american",
        "u.s.a.": "american",
        "u.s.": "american",
        "united kingdom": "british",
        "uk": "british",
        "great britain": "british",
        "england": "english",
        "scotland": "scottish",
        "wales": "welsh",
        "ireland": "irish",
        "netherlands": "dutch",
        "france": "french",
        "spain": "spanish",
        "portugal": "portuguese",
        "germany": "german",
        "switzerland": "swiss",
        "greece": "greek",
        "turkey": "turkish",
        "denmark": "danish",
        "sweden": "swedish",
        "norway": "norwegian",
        "finland": "finnish",
        "poland": "polish",
        "czech republic": "czech",
        "slovakia": "slovak",
        "philippines": "filipino",
        "china": "chinese",
        "japan": "japanese",
        "vietnam": "vietnamese",
        "thailand": "thai",
        "bangladesh": "bangladeshi",
        "pakistan": "pakistani",
        "afghanistan": "afghan",
        "iraq": "iraqi",
        "iran": "iranian",
        "israel": "israeli",
        "new zealand": "new zealander",
    }

    for country in countries:
        country_lower = country.lower()

        # Check special mappings first
        if country_lower in special_mappings:
            nationalities.append(special_mappings[country_lower])
        # Common patterns
        elif country_lower.endswith("ia"):
            # Countries ending in 'ia' typically add 'n' (e.g., India -> Indian)
            nationalities.append(country + "n")
        elif country_lower.endswith("a") and not country_lower.endswith("ia"):
            # Countries ending in 'a' typically add 'n' (e.g., Canada -> Canadian)
            nationalities.append(country + "n")
        elif country_lower.endswith("land"):
            # Countries ending in 'land' typically add 'er' or 'ish'
            base = country[:-4]
            nationalities.append(base + "er")
            nationalities.append(base + "ish")
        elif country_lower.endswith("y"):
            # Countries ending in 'y' often change to 'ian' (e.g., Italy -> Italian)
            nationalities.append(country[:-1] + "ian")
        else:
            # Default: add 'ian' or 'ese'
            nationalities.append(country + "ian")
            nationalities.append(country + "ese")

    return nationalities


def _build_religion_adherents(religions: List[str]) -> List[str]:
    """Generate adherent terms from religion names."""
    adherents = []

    # Special mappings for religion adherents
    special_mappings = {
        "christianity": "christian",
        "islam": "muslim",
        "judaism": "jewish",
        "buddhism": "buddhist",
        "hinduism": "hindu",
        "sikhism": "sikh",
        "jainism": "jain",
        "zoroastrianism": "zoroastrian",
        "catholicism": "catholic",
        "protestantism": "protestant",
        "orthodoxy": "orthodox",
        "eastern orthodoxy": "eastern orthodox",
        "shia islam": "shia",
        "sunni islam": "sunni",
        "atheism": "atheist",
        "agnosticism": "agnostic",
    }

    for religion in religions:
        religion_lower = religion.lower()

        # Check special mappings first
        if religion_lower in special_mappings:
            adherents.append(special_mappings[religion_lower])
        # Common patterns
        elif religion_lower.endswith("ism"):
            # Remove 'ism' and add 'ist'
            adherents.append(religion[:-3] + "ist")
        elif religion_lower.endswith("ity"):
            # Remove 'ity' (e.g., Christianity -> Christian)
            adherents.append(religion[:-3])
        else:
            # Default: add as is and with 'ist' suffix
            adherents.append(religion)
            adherents.append(religion + "ist")

    return adherents


# Load demographic data
dmg_countries = _load_dmg_countries()
dmg_genders = _load_dmg_genders()
dmg_city_to_country = _load_dmg_cities()
dmg_religion_to_main, dmg_religion_to_category = _load_dmg_religions()
dmg_occupation_to_soc = _load_dmg_occupations()

# Build term groups for regex patterns
dmg_nationalities = _build_nationalities_from_countries(dmg_countries)
dmg_cities = list(dmg_city_to_country.keys())
dmg_religions = list(
    set(list(dmg_religion_to_main.keys()) + list(dmg_religion_to_main.values()))
)
dmg_religion_adherents = _build_religion_adherents(dmg_religions)
dmg_occupations = list(dmg_occupation_to_soc.keys())


vad_dict = _load_nrc_vad_lexicon()
emotion_dict = _load_nrc_emotion_lexicon()
worry_dict = _load_nrc_worrywords_lexicon()
moraltrust_dict = _load_nrc_moraltrust_lexicon()
socialwarmth_dict = _load_nrc_socialwarmth_lexicon()
warmth_dict = _load_nrc_warmth_lexicon()
tense_dict = _load_eng_tenses_lexicon()
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
cog_dict = _load_cog_lexicon()


def compute_vad_and_emotions(
    text: str,
    vad_dict: Dict[str, Dict[str, float]],
    emotion_dict: Dict[str, set],
    emotions: List[str],
    worry_dict: Dict[str, int],
    moraltrust_dict: Dict[str, int],
    socialwarmth_dict: Dict[str, int],
    warmth_dict: Dict[str, int],
) -> Dict[str, Any]:
    words = text.lower().split() if isinstance(text, str) else []
    vad_scores = {dim: [] for dim in ("valence", "arousal", "dominance")}
    high_flags = {dim: 0 for dim in vad_scores}
    low_flags = {dim: 0 for dim in vad_scores}
    high_counts = {dim: 0 for dim in vad_scores}
    low_counts = {dim: 0 for dim in vad_scores}
    emotion_counts = {emo: 0 for emo in emotions}
    emotion_flags = {emo: 0 for emo in emotions}
    sum_anx = count_anx = sum_calm = count_calm = 0
    has_anx = has_calm = has_hi_anx = has_hi_calm = 0
    hi_anx_count = hi_calm_count = 0
    sum_moral = count_moral = has_hi_moral = has_lo_moral = 0
    hi_moral_count = lo_moral_count = 0
    sum_soc = count_soc = has_hi_soc = has_lo_soc = 0
    hi_soc_count = lo_soc_count = 0
    sum_warm = count_warm = has_hi_warm = has_lo_warm = 0
    hi_warm_count = lo_warm_count = 0
    for w in words:
        if w in vad_dict:
            for dim in vad_scores:
                sc = vad_dict[w][dim]
                vad_scores[dim].append(sc)
                if sc > 0.67:
                    high_flags[dim] = 1
                    high_counts[dim] += 1
                if sc < 0.33:
                    low_flags[dim] = 1
                    low_counts[dim] += 1
        if w in emotion_dict:
            for emo in emotion_dict[w]:
                emotion_flags[emo] = 1
                emotion_counts[emo] += 1
        if worry_dict and w in worry_dict:
            oc = worry_dict[w]
            if oc > 0:
                has_anx = 1
                sum_anx += oc
                count_anx += 1
                if oc == 3:
                    has_hi_anx = 1
                    hi_anx_count += 1
            elif oc < 0:
                has_calm = 1
                sum_calm += oc
                count_calm += 1
                if oc == -3:
                    has_hi_calm = 1
                    hi_calm_count += 1
        if moraltrust_dict and w in moraltrust_dict:
            oc = moraltrust_dict[w]
            sum_moral += oc
            count_moral += 1
            if oc == 3:
                has_hi_moral = 1
                hi_moral_count += 1
            if oc == -3:
                has_lo_moral = 1
                lo_moral_count += 1
        if socialwarmth_dict and w in socialwarmth_dict:
            oc = socialwarmth_dict[w]
            sum_soc += oc
            count_soc += 1
            if oc == 3:
                has_hi_soc = 1
                hi_soc_count += 1
            if oc == -3:
                has_lo_soc = 1
                lo_soc_count += 1
        if warmth_dict and w in warmth_dict:
            oc = warmth_dict[w]
            sum_warm += oc
            count_warm += 1
            if oc == 3:
                has_hi_warm = 1
                hi_warm_count += 1
            if oc == -3:
                has_lo_warm = 1
                lo_warm_count += 1
    avg_vad = {
        f"NRCAvg{dim.capitalize()}": sum(vals) / len(vals) if vals else 0
        for dim, vals in vad_scores.items()
    }
    emo_cols = {
        f"NRCHas{emo.capitalize()}Word": flag for emo, flag in emotion_flags.items()
    }
    emo_cnt_cols = {
        f"NRCCount{emo.capitalize()}Words": cnt for emo, cnt in emotion_counts.items()
    }
    vad_thr = {
        f"NRCHasHigh{dim.capitalize()}Word": high_flags[dim] for dim in high_flags
    }
    vad_thr.update(
        {f"NRCHasLow{dim.capitalize()}Word": low_flags[dim] for dim in low_flags}
    )
    vad_cnt = {
        f"NRCCountHigh{dim.capitalize()}Words": high_counts[dim] for dim in high_counts
    }
    vad_cnt.update(
        {f"NRCCountLow{dim.capitalize()}Words": low_counts[dim] for dim in low_counts}
    )
    word_count = {"WordCount": len(words)}
    avg_anx = sum_anx / count_anx if count_anx else 0
    avg_calm = sum_calm / count_calm if count_calm else 0
    worry_cols = {
        "NRCHasAnxietyWord": has_anx,
        "NRCHasCalmnessWord": has_calm,
        "NRCAvgAnxiety": avg_anx,
        "NRCAvgCalmness": avg_calm,
        "NRCHasHighAnxietyWord": has_hi_anx,
        "NRCCountHighAnxietyWords": hi_anx_count,
        "NRCHasHighCalmnessWord": has_hi_calm,
        "NRCCountHighCalmnessWords": hi_calm_count,
    }
    avg_moral = sum_moral / count_moral if count_moral else 0
    moral_cols = {
        "NRCHasHighMoralTrustWord": has_hi_moral,
        "NRCCountHighMoralTrustWord": hi_moral_count,
        "NRCHasLowMoralTrustWord": has_lo_moral,
        "NRCCountLowMoralTrustWord": lo_moral_count,
        "NRCAvgMoralTrustWord": avg_moral,
    }
    avg_soc = sum_soc / count_soc if count_soc else 0
    soc_cols = {
        "NRCHasHighSocialWarmthWord": has_hi_soc,
        "NRCCountHighSocialWarmthWord": hi_soc_count,
        "NRCHasLowSocialWarmthWord": has_lo_soc,
        "NRCCountLowSocialWarmthWord": lo_soc_count,
        "NRCAvgSocialWarmthWord": avg_soc,
    }
    avg_warmth = sum_warm / count_warm if count_warm else 0
    warm_cols = {
        "NRCHasHighWarmthWord": has_hi_warm,
        "NRCCountHighWarmthWord": hi_warm_count,
        "NRCHasLowWarmthWord": has_lo_warm,
        "NRCCountLowWarmthWord": lo_warm_count,
        "NRCAvgWarmthWord": avg_warmth,
    }
    return {
        **avg_vad,
        **vad_thr,
        **emo_cols,
        **emo_cnt_cols,
        **vad_cnt,
        **word_count,
        **worry_cols,
        **moral_cols,
        **soc_cols,
        **warm_cols,
    }


def load_body_parts(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip()]


BODY_PARTS = load_body_parts(_DATA_DIR / "BPM-bodywords-full.txt")


def compute_prefixed_body_part_mentions(
    text: str, body_parts: List[str]
) -> Dict[str, Any]:
    lower = text.lower()
    prefixes = [
        ("my ", "MyBPM"),
        ("your ", "YourBPM"),
        ("her ", "HerBPM"),
        ("his ", "HisBPM"),
        ("their ", "TheirBPM"),
    ]
    res: Dict[str, Any] = {}
    for pref, label in prefixes:
        res[label] = ", ".join(
            p for p in (pref + bp for bp in body_parts) if p in lower
        )
    res["HasBPM"] = any(bp in lower for bp in body_parts)
    return res


def compute_individual_pronouns(text: str) -> Dict[str, int]:
    pronouns = {
        "PRNHasI": ["i"],
        "PRNHasMe": ["me"],
        "PRNHasMy": ["my"],
        "PRNHasMine": ["mine"],
        "PRNHasWe": ["we"],
        "PRNHasOur": ["our"],
        "PRNHasOurs": ["ours"],
        "PRNHasYou": ["you"],
        "PRNHasYour": ["your"],
        "PRNHasYours": ["yours"],
        "PRNHasShe": ["she"],
        "PRNHasHer": ["her"],
        "PRNHasHers": ["hers"],
        "PRNHasHe": ["he"],
        "PRNHasHim": ["him"],
        "PRNHasHis": ["his"],
        "PRNHasThey": ["they"],
        "PRNHasThem": ["them"],
        "PRNHasTheir": ["their"],
        "PRNHasTheirs": ["theirs"],
    }
    words = set(text.lower().split()) if isinstance(text, str) else set()
    return {col: int(any(w in words for w in lst)) for col, lst in pronouns.items()}


def compute_tense_features(
    text: str, tense_dict: Dict[str, List[str]]
) -> Dict[str, int]:
    """Compute tense-related features from text."""
    words = text.lower().split() if isinstance(text, str) else []

    # Tense-related features
    past_count = 0
    present_count = 0
    future_modals = {"will", "shall", "should", "going to"}
    future_modal_count = 0

    for w in words:
        # Check future modals directly
        if w in future_modals:
            future_modal_count += 1

        # Check tense_dict if available
        if tense_dict and w in tense_dict:
            for tag in tense_dict[w]:
                # Check for past tense
                if "PST" in tag:
                    past_count += 1
                # Check for present tense
                elif "PRS" in tag:
                    present_count += 1

    # Check for future time reference words
    future_time_words = {"tomorrow", "next day", "next year", "next month"}
    joined_text = " ".join(words)
    has_future_time_reference = any(
        phrase in joined_text for phrase in future_time_words
    )

    return {
        "TIMEHasPastVerb": 1 if past_count > 0 else 0,
        "TIMECountPastVerbs": past_count,
        "TIMEHasPresentVerb": 1 if present_count > 0 else 0,
        "TIMECountPresentVerbs": present_count,
        "TIMEHasFutureModal": 1 if future_modal_count > 0 else 0,
        "TIMECountFutureModals": future_modal_count,
        "TIMEHasPresentNoFuture": (
            1 if (present_count > 0 and future_modal_count == 0) else 0
        ),
        "TIMEHasFutureReference": 1 if has_future_time_reference else 0,
    }


def compute_cognitive_features(
    text: str, cog_dict: Dict[str, Set[str]]
) -> Dict[str, int]:
    if not isinstance(text, str) or not text.strip():
        return {f"COGHas{cat}Word": 0 for cat in cog_dict}
    words = set(text.lower().split())
    return {
        f"COGHas{cat}Word": int(bool(words & terms))
        for cat, terms in cog_dict.items()
    }


# -------------------- #
# TUSC-specific Helper
# -------------------- #


def detect_self_identification_in_tusc_entry(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, List[str]]:
    tweet = entry.get("Tweet", "") or ""
    # Create combined entry preserving original metadata
    combined_entry = entry.copy()
    combined_entry.update({"title": "", "selftext": tweet})
    return detect_self_identification_with_resolved_age(combined_entry, detector)


def detect_self_identification_in_tusc_entry_with_mappings(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect self-identification in TUSC entry with demographic mappings.

    Returns:
        Tuple of (age_resolved_matches, formatted_demographics)
    """
    tweet = entry.get("Tweet", "") or ""
    # Create combined entry preserving original metadata
    combined_entry = entry.copy()
    combined_entry.update({"title": "", "selftext": tweet})

    # Get age-resolved matches
    age_matches = detect_self_identification_with_resolved_age(combined_entry, detector)

    # Get full demographic detections with mappings
    demographic_detections = detector.detect_with_mappings(tweet)
    formatted_demographics = format_demographic_detections_for_output(
        demographic_detections
    )

    return age_matches, formatted_demographics


def apply_linguistic_features(
    text: str, include_features: bool = True
) -> Dict[str, Any]:
    if not include_features:
        return {}
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Text for linguistic feature extraction must be a non-empty string"
        )
    features = compute_vad_and_emotions(
        text,
        vad_dict,
        emotion_dict,
        emotions,
        worry_dict,
        moraltrust_dict,
        socialwarmth_dict,
        warmth_dict,
    )
    features.update(compute_individual_pronouns(text))
    features.update(compute_prefixed_body_part_mentions(text, BODY_PARTS))
    features.update(compute_tense_features(text, tense_dict))
    features.update(compute_cognitive_features(text, cog_dict))
    return features


# -------------------- #
# Flattening and I/O Utilities
# -------------------- #


def flatten_result_to_csv_row(
    result: Dict[str, Any],
    data_source: str,
    split: Optional[str] = None,
    stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    row: Dict[str, Any] = {}
    # Author column
    if data_source == "tusc":
        row["Author"] = result.get("UserID", "") or result.get("userID", "") or ""
    else:
        row["Author"] = result.get("author", "") or ""

    # Compute majority birthyear and raw birthyear extractions when available
    if "self_identification" in result:
        self_id = result["self_identification"]
        if data_source == "tusc":
            try:
                ref_year = int(result.get("Year", ""))
            except (TypeError, ValueError):
                ref_year = parse_tusc_created_at_year(result.get("createdAt", ""))
                if ref_year is None:
                    ref_year = datetime.now().year
        else:
            try:
                ts = result.get("post", {}).get("created_utc")
                ref_year = datetime.utcfromtimestamp(int(ts)).year
            except Exception:
                ref_year = datetime.now().year

        raw_matches: List[str] = []
        majority_birthyear: Optional[int] = None
        if "resolved_age" in self_id:
            raw_matches = list(self_id["resolved_age"].get("raw_matches", []))
            age_val = self_id["resolved_age"].get("age")
            if isinstance(age_val, int):
                # Stage 1: Check if age is within valid range (13-99)
                if stage == "users" and not (13 <= age_val <= 99):
                    return None
                majority_birthyear = ref_year - age_val
        else:
            raw_matches = list(self_id.get("age", []))
            if raw_matches:
                try:
                    val = int(raw_matches[0])
                except ValueError:
                    val = None
                if isinstance(val, int):
                    if (
                        (ref_year - 99) <= val <= (ref_year - 13)
                    ):  # Birth years for ages 13-99
                        majority_birthyear = val
                        # Stage 1: Check if age is within valid range (13-99)
                        if stage == "users":
                            age_at_ref = ref_year - val
                            if not (13 <= age_at_ref <= 99):
                                return None
                    elif 13 <= val <= 99:  # Direct age values within valid range
                        majority_birthyear = ref_year - val
                        # Stage 1: Check if age is within valid range (13-99)
                        if stage == "users" and not (13 <= val <= 99):
                            return None
        raw_birthyears: List[int] = []
        for m in raw_matches:
            try:
                v = int(m)
            except ValueError:
                continue
            if (ref_year - 99) <= v <= (ref_year - 13):  # Birth years for ages 13-99
                raw_birthyears.append(v)
            elif 13 <= v <= 99:  # Direct age values within valid range
                raw_birthyears.append(ref_year - v)
        row["DMGMajorityBirthyear"] = majority_birthyear or ""
        row["DMGRawBirthyearExtractions"] = "|".join(str(x) for x in raw_birthyears)

    # Include age at posting if available (stage2)
    if "DMGAgeAtPost" in result:
        age_at_post = result.get("DMGAgeAtPost", "")
        # Stage 2: Check if age at posting is within valid range (13-99)
        if stage == "posts" and age_at_post != "":
            try:
                age_val = int(age_at_post)
                if not (13 <= age_val <= 99):
                    return None
            except (ValueError, TypeError):
                return None
        row["DMGAgeAtPost"] = age_at_post

    if data_source == "tusc":
        row["PostID"] = result.get("TweetID", "")
        row["PostText"] = result.get("Tweet", "")
        row["PostCreatedAt"] = result.get("createdAt", "")
        row["PostYear"] = result.get("Year", "")
        row["PostMonth"] = result.get("Month", "")
        if split == "city":
            row["PostCity"] = result.get("City", "")
            loc_fields = ["PostCity"]
        else:
            row["PostCountry"] = result.get("Country", "")
            row["PostMyCountry"] = result.get("MyCountry", "")
            loc_fields = ["PostCountry", "PostMyCountry"]
        row["PostPlace"] = result.get("Place", "")
        row["PostPlaceID"] = result.get("PlaceID", "")
        row["PostPlaceType"] = result.get("PlaceType", "")
        static_keys = {
            "TweetID",
            "Tweet",
            "createdAt",
            "Year",
            "Month",
            "City",
            "Country",
            "MyCountry",
            "Place",
            "PlaceID",
            "PlaceType",
            "UserID",
            "userID",
            "userName",
            "Author",
            "DMGAgeAtPost",
            "DMGMajorityBirthyear",
            "DMGRawBirthyearExtractions",
        }
        # Include demographic fields for both users and posts stages
        for key, val in result.items():
            if (
                key not in static_keys
                and key not in row
                and key not in {"File", "RowNum", "self_identification"}
                and key.startswith("DMG")  # Include all demographic fields
            ):
                row[key] = val
        # Include other dynamic fields only for posts stage
        if stage == "posts":
            for key, val in result.items():
                if (
                    key not in static_keys
                    and key not in row
                    and key not in {"File", "RowNum", "self_identification"}
                    and not key.startswith("DMG")  # Non-demographic fields
                ):
                    row[key] = val
    else:
        post = result.get("post", result)
        row["PostID"] = post.get("id", "")
        row["PostSubreddit"] = post.get("subreddit", "")
        row["PostTitle"] = clean_text_newlines(post.get("title", ""))
        row["PostSelftext"] = clean_text_newlines(post.get("selftext", ""))
        row["PostCreatedUtc"] = post.get("created_utc", "")
        row["PostScore"] = post.get("score", "")
        row["PostNumComments"] = post.get("num_comments", "")
        row["PostPermalink"] = post.get("permalink", "")
        row["PostUrl"] = post.get("url", "")
        row["PostMediaPath"] = post.get("media_path", "")
        static_keys = {
            "id",
            "subreddit",
            "title",
            "selftext",
            "created_utc",
            "score",
            "num_comments",
            "permalink",
            "url",
            "media_path",
            "author",
        }
        # Include demographic fields for both users and posts stages
        for key, val in result.items():
            if (
                key not in static_keys
                and key not in row
                and key not in {"File", "RowNum", "self_identification", "post"}
                and key.startswith("DMG")  # Include all demographic fields
            ):
                row[key] = val
        # Include other dynamic fields only for posts stage
        if stage == "posts":
            for key, val in post.items():
                if key not in static_keys and key not in {
                    "File",
                    "RowNum",
                    "self_identification",
                }:
                    row[key] = val
    return row


def get_csv_fieldnames(
    data_source: str,
    split: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[str]:
    """
    Get static CSV/TSV header fields based on data source and stage ('users' or 'posts').
    """
    # User-level headers: DMG birthyear info and all demographic fields; for posts only include age at post
    if stage == "users":
        user_cols = [
            "Author",
            "DMGMajorityBirthyear",
            "DMGRawBirthyearExtractions",
            "DMGRawExtractedAge",
            "DMGRawExtractedGender",
            "DMGRawExtractedCity",
            "DMGCountryMappedFromExtractedCity",
            "DMGRawExtractedCountry",
            "DMGRawExtractedReligion",
            "DMGMainReligionMappedFromExtractedReligion",
            "DMGMainCategoryMappedFromExtractedReligion",
            "DMGRawExtractedOccupation",
            "DMGSOCTitleMappedFromExtractedOccupation",
        ]
    else:
        user_cols = ["Author", "DMGAgeAtPost"]
    if data_source == "tusc":
        base = user_cols + [
            "PostID",
            "PostText",
            "PostCreatedAt",
            "PostYear",
            "PostMonth",
        ]
        if split == "city":
            loc = ["PostCity", "PostPlace", "PostPlaceID", "PostPlaceType"]
        else:
            loc = [
                "PostCountry",
                "PostMyCountry",
                "PostPlace",
                "PostPlaceID",
                "PostPlaceType",
            ]
        return base + loc
    # Reddit headers
    base = user_cols + [
        "PostID",
        "PostSubreddit",
        "PostTitle",
        "PostSelftext",
        "PostCreatedUtc",
        "PostScore",
        "PostNumComments",
        "PostPermalink",
        "PostUrl",
        "PostMediaPath",
    ]
    return base


def ensure_output_directory(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str,
    output_tsv: bool,
    data_source: str,
    split: Optional[str] = None,
) -> None:
    sep = "\t" if output_tsv else ","
    ext = "tsv" if output_tsv else "csv"
    out = output_file.replace(".csv", f".{ext}") if output_tsv else output_file
    ensure_output_directory(out)
    # Determine whether writing users or posts file based on filename
    fname = os.path.basename(out)
    stage = "posts" if "posts" in fname else "users"
    if results:
        rows = [
            flatten_result_to_csv_row(r, data_source, split, stage) for r in results
        ]
        # Filter out None results (rows that don't meet age criteria)
        rows = [row for row in rows if row is not None]

        if rows:  # Only write if we have valid rows after filtering
            # determine header including any feature columns
            static_fields = get_csv_fieldnames(data_source, split, stage)
            extra_fields = sorted(
                {k for row in rows for k in row.keys() if k not in static_fields}
            )
            fieldnames = static_fields + extra_fields
            with open(out, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        else:
            # Write empty file with headers only
            with open(out, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=get_csv_fieldnames(data_source, split, stage),
                    delimiter=sep,
                )
                writer.writeheader()
    else:
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=get_csv_fieldnames(data_source, split, stage),
                delimiter=sep,
            )
            writer.writeheader()


def append_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str,
    output_tsv: bool,
    data_source: str,
    split: Optional[str] = None,
) -> None:
    """Append rows to a CSV/TSV while creating the file with a header if needed."""
    sep = "\t" if output_tsv else ","
    ext = "tsv" if output_tsv else "csv"
    out = output_file.replace(".csv", f".{ext}") if output_tsv else output_file
    ensure_output_directory(out)
    fname = os.path.basename(out)
    stage = "posts" if "posts" in fname else "users"
    if not results:
        return
    rows = [flatten_result_to_csv_row(r, data_source, split, stage) for r in results]
    # Filter out None results (rows that don't meet age criteria)
    rows = [row for row in rows if row is not None]

    if not rows:  # No valid rows after filtering
        return

    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(sep)
        fieldnames = header
        write_header = False
    else:
        static_fields = get_csv_fieldnames(data_source, split, stage)
        extra_fields = sorted(
            {k for row in rows for k in row.keys() if k not in static_fields}
        )
        fieldnames = static_fields + extra_fields
        write_header = True
    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -------------------- #
# JSONL File Handling & Filtering
# -------------------- #


def get_all_jsonl_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for root, _, fnames in os.walk(path):
        for nm in fnames:
            if nm.startswith("RS_"):
                files.append(os.path.join(root, nm))
    # Sort by name to ensure consistent order
    files.sort()
    return files


def clean_text_newlines(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def filter_entry(
    entry: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
) -> bool:
    if entry.get("over_18", False):
        return False
    if (
        entry.get("promoted") is True
        or entry.get("whitelist_status") == "promo_specified"
    ):
        return False
    text = entry.get("selftext", "")
    if not text.strip():
        return False
    n = len(text.strip().split())
    if n < min_words or n > max_words:
        return False
    has_vid = bool(entry.get("is_video", False))
    url = entry.get("url", "") or ""
    has_img = any(
        url.lower().endswith(ext) for ext in (".jpg", ".png", ".jpeg", ".gif")
    )
    if split == "text" and (has_vid or has_img):
        return False
    if split == "multimodal" and not (has_vid or has_img):
        return False
    return True


def extract_columns(
    entry: Dict[str, Any], local_media_path: Optional[str]
) -> Dict[str, Any]:
    return {
        "id": entry.get("id"),
        "subreddit": entry.get("subreddit"),
        "title": entry.get("title", ""),
        "selftext": clean_text_newlines(entry.get("selftext", "")),
        "created_utc": entry.get("created_utc"),
        "score": entry.get("score"),
        "num_comments": entry.get("num_comments"),
        "author": entry.get("author"),
        "permalink": entry.get("permalink"),
        "url": entry.get("url"),
        "media_path": local_media_path,
    }


# -------------------- #
# PII Detection
# -------------------- #


class PIIDetector:
    """Detect personally identifiable information (PII) using Presidio."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        # Entity types to detect - only keeping relevant PII types
        self.entity_types = [
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "IP_ADDRESS",
            "MEDICAL_LICENSE",
            "PHONE_NUMBER",
            "CRYPTO",
        ]

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII entities in text and return them grouped by type."""
        if not text or not isinstance(text, str):
            return {}

        # Analyze text
        results = self.analyzer.analyze(
            text=text, entities=self.entity_types, language="en"
        )

        # Group results by entity type
        pii_by_type = {}
        for result in results:
            entity_type = result.entity_type
            entity_text = text[result.start : result.end]

            if entity_type not in pii_by_type:
                pii_by_type[entity_type] = []

            # Avoid duplicates
            if entity_text not in pii_by_type[entity_type]:
                pii_by_type[entity_type].append(entity_text)

        return pii_by_type

    def format_pii_for_output(self, pii_dict: Dict[str, List[str]]) -> Dict[str, str]:
        """Format PII detection results for TSV output."""
        formatted = {}

        # Create individual columns for each PII type
        for entity_type in self.entity_types:
            column_name = f"PII_{entity_type}"
            if entity_type in pii_dict:
                # Join multiple instances with semicolon
                formatted[column_name] = "; ".join(pii_dict[entity_type])
            else:
                formatted[column_name] = ""

        # Add summary columns
        formatted["PII_Types_Found"] = "; ".join(sorted(pii_dict.keys()))
        formatted["PII_Count"] = sum(len(entities) for entities in pii_dict.values())
        formatted["Has_PII"] = len(pii_dict) > 0

        return formatted


def detect_pii_in_post(
    entry: Dict[str, Any], detector: PIIDetector
) -> Optional[Dict[str, Any]]:
    """Detect PII in a Reddit post and return formatted results."""
    # Combine title and selftext for analysis
    title = entry.get("title", "")
    selftext = entry.get("selftext", "")
    combined_text = f"{title} {selftext}"

    # Detect PII
    pii_detected = detector.detect_pii(combined_text)

    # Only return posts that have PII
    if not pii_detected:
        return None

    # Get basic post information
    post_info = extract_columns(entry, None)

    # Add PII detection results
    pii_formatted = detector.format_pii_for_output(pii_detected)
    post_info.update(pii_formatted)

    return post_info


def aggregate_user_demographics(df: pd.DataFrame, data_source: str) -> pd.DataFrame:
    """
    Aggregate demographics per author using majority vote across multiple self-identification posts.
    Handles birthyear resolution by collecting all raw extractions and re-resolving with the latest post year.
    For other fields, uses majority vote (mode) for mapped fields and most common value for raw extractions.
    """
    detector = SelfIdentificationDetector()

    def get_post_year(group):
        if data_source == "tusc" and "PostYear" in group.columns:
            return group["PostYear"].max()
        elif data_source == "reddit" and "PostCreatedUtc" in group.columns:
            years = []
            for utc in group["PostCreatedUtc"].dropna():
                try:
                    years.append(datetime.utcfromtimestamp(int(utc)).year)
                except (ValueError, TypeError):
                    pass
            return max(years) if years else datetime.now().year
        else:
            return datetime.now().year

    def aggregate_group(group):
        agg_row = {"Author": group.name}

        # Special handling for birthyear: collect all raw extractions, resolve using max post year
        all_raw_birthyears = []
        for val in group.get("DMGRawBirthyearExtractions", pd.Series()).dropna():
            all_raw_birthyears.extend(str(val).split("|"))
        all_raw_birthyears = [r.strip() for r in all_raw_birthyears if r.strip()]

        if all_raw_birthyears:
            ref_year = get_post_year(group)
            resolution = detector.resolve_multiple_ages(all_raw_birthyears, current_year=ref_year)
            if resolution and resolution[1] >= 0.5:
                resolved_age, _ = resolution
                agg_row["DMGMajorityBirthyear"] = ref_year - resolved_age
            else:
                agg_row["DMGMajorityBirthyear"] = pd.NA
            agg_row["DMGRawBirthyearExtractions"] = "|".join(sorted(set(all_raw_birthyears)))
        else:
            agg_row["DMGMajorityBirthyear"] = pd.NA
            agg_row["DMGRawBirthyearExtractions"] = pd.NA

        # For other DMG fields
        for col in [c for c in group.columns if c.startswith("DMG") and c not in ["DMGMajorityBirthyear", "DMGRawBirthyearExtractions"]]:
            if "Raw" in col:
                # Most common raw value
                all_items = []
                for val in group[col].dropna():
                    all_items.extend(str(val).split("|"))
                all_items = [item.strip() for item in all_items if item.strip()]
                if all_items:
                    counts = Counter(all_items)
                    most_common = counts.most_common(1)
                    agg_row[col] = most_common[0][0] if most_common else pd.NA
                else:
                    agg_row[col] = pd.NA
            else:
                # Mode for mapped/non-raw fields
                series = group[col].dropna()
                if not series.empty:
                    mode = series.mode()
                    agg_row[col] = mode[0] if not mode.empty else series.iloc[0]
                else:
                    agg_row[col] = pd.NA

        return pd.Series(agg_row)

    # Group by Author and apply aggregation
    aggregated = df.groupby("Author").apply(aggregate_group).reset_index(drop=True)

    return aggregated
