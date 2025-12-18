"""Self-identification detection for demographic information."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple

import nltk
from nltk.corpus import stopwords

from abcde.utils.text import build_term_group
from abcde.utils.dates import parse_tusc_created_at_year

# Download stopwords if not already available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


# Lazy-loaded demographic data
_dmg_data_loaded = False
_dmg_countries: List[str] = []
_dmg_genders: List[str] = []
_dmg_city_to_country: Dict[str, str] = {}
_dmg_religion_to_main: Dict[str, str] = {}
_dmg_religion_to_category: Dict[str, str] = {}
_dmg_occupation_to_soc: Dict[str, str] = {}
_dmg_nationalities: List[str] = []
_dmg_cities: List[str] = []
_dmg_religions: List[str] = []
_dmg_religion_adherents: List[str] = []
_dmg_occupations: List[str] = []


def _ensure_dmg_data_loaded() -> None:
    """Ensure demographic data is loaded (lazy loading)."""
    global _dmg_data_loaded
    global _dmg_countries, _dmg_genders, _dmg_city_to_country
    global _dmg_religion_to_main, _dmg_religion_to_category, _dmg_occupation_to_soc
    global _dmg_nationalities, _dmg_cities, _dmg_religions, _dmg_religion_adherents
    global _dmg_occupations

    if _dmg_data_loaded:
        return

    from abcde.lexicons.loaders import (
        load_dmg_countries,
        load_dmg_genders,
        load_dmg_cities,
        load_dmg_religions,
        load_dmg_occupations,
    )

    _dmg_countries = load_dmg_countries()
    _dmg_genders = load_dmg_genders()
    _dmg_city_to_country = load_dmg_cities()
    _dmg_religion_to_main, _dmg_religion_to_category = load_dmg_religions()
    _dmg_occupation_to_soc = load_dmg_occupations()

    # Build derived data
    _dmg_nationalities = _build_nationalities_from_countries(_dmg_countries)
    _dmg_cities = list(_dmg_city_to_country.keys())
    _dmg_religions = list(
        set(list(_dmg_religion_to_main.keys()) + list(_dmg_religion_to_main.values()))
    )
    _dmg_religion_adherents = _build_religion_adherents(_dmg_religions)
    _dmg_occupations = list(_dmg_occupation_to_soc.keys())

    _dmg_data_loaded = True


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


class SelfIdentificationDetector:
    """Detect self-identification statements (age, etc.) inside free text.

    The detector is data-source agnostic – it expects plain strings and can be
    reused for Reddit, Twitter, blogs, or any other textual resource.
    """

    def __init__(self) -> None:
        _ensure_dmg_data_loaded()

        # Use NLTK stopwords for English
        self.stopwords = set(stopwords.words("english"))

        # Store valid cities set for validation
        self.valid_cities = set()
        self.valid_cities_lower = set()

        # Build term groups from loaded data
        # For patterns that capture the term directly, use capturing=True
        OCCUPATIONS_TERMS_REGEX = build_term_group(_dmg_occupations, capturing=True)
        GENDERS_TERMS_REGEX = build_term_group(_dmg_genders, capturing=True)
        COUNTRIES_TERMS_REGEX = build_term_group(_dmg_countries, capturing=True)
        NATIONALITIES_TERMS_REGEX = build_term_group(_dmg_nationalities, capturing=True)

        # Filter cities to remove common words and countries
        filtered_cities = self._filter_city_list(_dmg_cities, _dmg_countries)
        # Store valid cities for validation
        self.valid_cities = set(filtered_cities)
        self.valid_cities_lower = {city.lower() for city in filtered_cities}
        CITIES_TERMS_REGEX = build_term_group(filtered_cities, capturing=True)

        RELIGIONS_ADHERENTS_TERMS_REGEX = build_term_group(
            _dmg_religion_adherents, capturing=True
        )
        RELIGIONS_NAMES_TERMS_REGEX = build_term_group(_dmg_religions, capturing=True)

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
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+looking\s+for)\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"(?:\s+at\s+[\w\s.-]+)?(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|while|when|because)\b))",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+looking\s+for)\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"(?:\s+at\s+[\w\s.-]+)?\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+work\s+as\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"(?:\s+at\s+[\w\s.-]+)?\b",
                    re.I,
                ),
                re.compile(
                    r"\bMy\s+(?:job|occupation|profession|role)\s+is\s+(?:to\s+be\s+)?(?:a|an|as)?\s*"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI'm\s+(?:currently\s+)?employed\s+as\s+(?:a|an|the)?\s+"
                    + OCCUPATIONS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "gender": [
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an)?\s+"
                    + GENDERS_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|because)\b))",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an)?\s+"
                    + GENDERS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+identify\s+as\s+(?:a|an)?\s*" + GENDERS_TERMS_REGEX + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bMy\s+gender(?:\s+identity)?\s+is\s+"
                    + GENDERS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\b(?:I'm|I\s+am)\s+(?:a\s+)?(?:transgender|trans)\s+"
                    + GENDERS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "country": [
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+just\s+visiting|\s+originally)\s+from\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)\s+from\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an|the)?\s*"
                    + NATIONALITIES_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|who)\b))",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not)\s+(?:a|an|the)?\s*"
                    + NATIONALITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+live\s+(?:in|at)\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+come\s+from\s+(?:the\s+)?" + COUNTRIES_TERMS_REGEX + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bMy\s+(?:nationality|citizenship)\s+is\s+"
                    + NATIONALITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+was|'m)\s+born\s+and\s+(?:raised|grew\s+up)\s+in\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)\s+originally\s+from\s+(?:the\s+)?"
                    + COUNTRIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "city": [
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+just\s+visiting|\s+originally)\s+from\s+"
                    + CITIES_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet)\b))",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+just\s+visiting|\s+originally)\s+from\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+live\s+(?:in|at)\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b(?!\s+(?:fear|hope|a\s+state\s+of|sin|poverty|luxury))",
                    re.I,
                ),
                re.compile(
                    r"\b(?:I'm\s+|I\s+am\s+)?(?:currently\s+)?(?:residing|based)\s+in\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bMy\s+(?:current\s+|home\s+)?(?:city|town)\s+is\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+(?:grew\s+up|was\s+raised)\s+in\s+"
                    + CITIES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
            ],
            "religion": [
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+sure\s+if\s+I'm)\s+(?:a|an|a\s+devout|a\s+practicing)?\s+"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"(?=\s*(?:$|[,.!?]|\b(?:and|but|so|or|yet|who)\b))",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)(?!\s+not|\s+sure\s+if\s+I'm)\s+(?:a|an|a\s+devout|a\s+practicing)?\s+"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bMy\s+(?:religion|faith)\s+is\s+"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+(?:actively\s+)?practice\s+"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s+am|'m)\s+a\s+follower\s+of\s+"
                    + RELIGIONS_NAMES_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+converted\s+to\s+" + RELIGIONS_NAMES_TERMS_REGEX + r"\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+was\s+(?:raised|born)\s+(?:as\s+(?:a|an))?\s*"
                    + RELIGIONS_ADHERENTS_TERMS_REGEX
                    + r"\b",
                    re.I,
                ),
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
        """Filter city list to remove common words and country names."""
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
        """Return demographic detections with both raw extractions and mapped values."""
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
                for city in lower_matches:
                    if city in self.valid_cities_lower and city in _dmg_city_to_country:
                        validated_cities.append(city)
                        country_mapped.append(_dmg_city_to_country[city])
                result[category]["raw"] = validated_cities
                result[category]["country_mapped"] = country_mapped

            elif category == "religion":
                # Map religions to main religion and category
                main_religion_mapped = []
                category_mapped = []
                for religion in lower_matches:
                    # Try direct mapping first
                    if religion in _dmg_religion_to_main:
                        main_religion_mapped.append(_dmg_religion_to_main[religion])
                    elif religion + "ism" in _dmg_religion_to_main:
                        main_religion_mapped.append(
                            _dmg_religion_to_main[religion + "ism"]
                        )
                    elif religion + "ity" in _dmg_religion_to_main:
                        main_religion_mapped.append(
                            _dmg_religion_to_main[religion + "ity"]
                        )
                    elif religion == "atheist" and "atheism" in _dmg_religion_to_main:
                        main_religion_mapped.append(_dmg_religion_to_main["atheism"])
                    elif (
                        religion == "agnostic" and "agnosticism" in _dmg_religion_to_main
                    ):
                        main_religion_mapped.append(_dmg_religion_to_main["agnosticism"])
                    else:
                        main_religion_mapped.append(None)

                    # Category mapping
                    if religion in _dmg_religion_to_category:
                        category_mapped.append(_dmg_religion_to_category[religion])
                    elif religion + "ism" in _dmg_religion_to_category:
                        category_mapped.append(
                            _dmg_religion_to_category[religion + "ism"]
                        )
                    elif religion + "ity" in _dmg_religion_to_category:
                        category_mapped.append(
                            _dmg_religion_to_category[religion + "ity"]
                        )
                    elif (
                        religion == "atheist" and "atheism" in _dmg_religion_to_category
                    ):
                        category_mapped.append(_dmg_religion_to_category["atheism"])
                    elif (
                        religion == "agnostic"
                        and "agnosticism" in _dmg_religion_to_category
                    ):
                        category_mapped.append(_dmg_religion_to_category["agnosticism"])
                    else:
                        category_mapped.append(None)

                result[category]["main_religion_mapped"] = main_religion_mapped
                result[category]["category_mapped"] = category_mapped

            elif category == "occupation":
                # Map occupations to SOC titles
                soc_mapped = []
                for occupation in lower_matches:
                    if occupation in _dmg_occupation_to_soc:
                        soc_mapped.append(_dmg_occupation_to_soc[occupation])
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
                if (
                    age_str.startswith("'")
                    and len(age_str) == 3
                    and age_str[1:].isdigit()
                ):
                    year_val = int(age_str[1:])
                    birth_year = (
                        1900 + year_val
                        if year_val > (current_year % 100)
                        else 2000 + year_val
                    )
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
    """Detect self-identification with mappings in a Reddit-style entry (title+body)."""
    title = entry.get("title", "") or ""
    body = entry.get("selftext", "") or ""
    combined = f"{title} {body}".strip()
    return detector.detect_with_mappings(combined)


def detect_self_identification_with_resolved_age(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, Any]:
    """Detect self identification with age resolution for multiple age extractions."""
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


def detect_self_identification_in_tusc_entry(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, List[str]]:
    """Detect self-identification in a TUSC entry."""
    tweet = entry.get("Tweet", "") or ""
    combined_entry = entry.copy()
    combined_entry.update({"title": "", "selftext": tweet})
    return detect_self_identification_with_resolved_age(combined_entry, detector)


def detect_self_identification_in_tusc_entry_with_mappings(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect self-identification in TUSC entry with demographic mappings."""
    tweet = entry.get("Tweet", "") or ""
    combined_entry = entry.copy()
    combined_entry.update({"title": "", "selftext": tweet})

    age_matches = detect_self_identification_with_resolved_age(combined_entry, detector)
    demographic_detections = detector.detect_with_mappings(tweet)
    
    # Import here to avoid circular import
    from abcde.io.demographics import format_demographic_detections_for_output
    formatted_demographics = format_demographic_detections_for_output(
        demographic_detections
    )

    return age_matches, formatted_demographics

