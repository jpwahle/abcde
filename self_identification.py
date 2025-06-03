import re
from typing import Dict, List, Pattern, Any, Optional, Tuple
from collections import Counter
from datetime import datetime


class SelfIdentificationDetector:
    """Detect self-identification statements (age, gender, location, etc.) inside free text.

    The detector is **data-source agnostic** – it expects plain strings and can therefore be
    reused for Reddit, Twitter, blogs, or any other textual resource.
    """

    def __init__(self) -> None:
        # Build and compile the detection regexes only once during init.
        self.patterns: Dict[str, List[Pattern[str]]] = {
            # --- AGE --------------------------------------------------------
            "age": [
                # I am 24 years old / I'm 25 years old
                re.compile(r"\bI(?:\s+am|'m)\s+([1-9]\d?)\s+years?\s+old\b", re.I),

                # I am 24 / I'm 24 (followed by age-related word or sentence boundary, no symbols)
                re.compile(
                    r"""
                    \bI(?:\s+am|'m)\s+              # “I am” / “I'm”
                    ([1-9]\d?)                      # capture the age (1–99)

                    # ── look-ahead: what’s allowed *after* the number ──────────────
                    (?=                             # start of forward look-ahead
                        (?:\s+                      # optional whitespace then …
                            (?:years?(?:\s+old|-old)?|yo|yrs?)\b  # … one of the age words
                        )?                          # age words are optional
                        \s*                         # optional spaces
                        (?:[.!?;,]\s*)?             # optional sentence-ending punctuation
                        $                           # … and then end-of-string only
                    )
                    (?!\s*[%$°#@&*+=<>()[\]{}|\\~`^_])  # still forbid the symbol set
                    """,
                    re.I | re.VERBOSE
                ),

                # I was born in 1998 / I am born in 1998  (four-digit birth year ≤ 2025)
                re.compile(
                    r"\bI(?:\s+was|\s+am|'m)\s+born\s+in\s+(19\d{2}|20(?:0\d|1\d|2[0-4]|25))\b",
                    re.I
                ),

                # I was born on 14 July 1992  (birth year from full date, ≤ 2025)
                re.compile(
                    r"\bI\s+was\s+born\s+on\s+\d{1,2}\s+\w+\s+(19\d{2}|20(?:0\d|1\d|2[0-4]|25))\b",
                    re.I
                ),

                # I'm turning 25 / I turn 25 / I turned 25
                re.compile(
                    r"\bI(?:\s*'m\s*turning|\s+turn(?:ed)?)\s+([1-9]\d?)"
                    r"(?=\s*[.!?;,]|\s*$)(?!\s*[%$°#@&*+=<>()[\]{}|\\~`^_])",
                    re.I
                ),

            ],
            # # TODO: Extract Gender, Location, Profession, Religion, etc. using lists
            # # --- GENDER -----------------------------------------------------
            # "gender": [

            # ],
            # # --- LOCATION / NATIONALITY ------------------------------------
            # "location": [

            # ],
            # # --- PROFESSION -------------------------------------------------
            # "profession": [

            # ],
            # # --- RELIGION ---------------------------------------------------
            # "religion": [

            # ],
        }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Return a mapping from category name to **unique** matched strings.

        Parameters
        ----------
        text: str
            Free-form text that potentially contains self-identification statements.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary such as ``{"age": ["24"], "gender": ["female"]}``. Empty lists are
            omitted.
        """
        if not isinstance(text, str) or not text:
            return {}

        text = text.strip()
        matches: Dict[str, List[str]] = {}

        for category, regs in self.patterns.items():
            cat_matches: List[str] = []
            for reg in regs:
                for m in reg.finditer(text):
                    # Prefer captured group if available else full match.
                    if m.groups():
                        cat_matches.append(m.group(1).strip())
                    else:
                        cat_matches.append(m.group(0).strip())
            if cat_matches:
                # Deduplicate while preserving order
                uniq: List[str] = []
                for cm in cat_matches:
                    if cm not in uniq:
                        uniq.append(cm)
                matches[category] = uniq

        return matches

    def resolve_multiple_ages(self, age_matches: List[str], current_year: Optional[int] = None) -> Optional[Tuple[int, float]]:
        """Resolve multiple age extractions using birth year normalization and majority vote.
        
        Parameters
        ----------
        age_matches : List[str]
            List of extracted age strings (can be ages like "25" or birth years like "1998")
        current_year : Optional[int]
            Current year for birth year calculation. Defaults to current year.
            
        Returns
        -------
        Optional[Tuple[int, float]]
            Tuple of (resolved_age, confidence_score) or None if no valid ages found.
            Confidence ranges from 0.0 to 1.0.
        """
        if not age_matches:
            return None
            
        if current_year is None:
            current_year = datetime.now().year
            
        # Convert all age matches to estimated birth years with confidence weights
        birth_year_candidates = []
        
        for age_str in age_matches:
            try:
                age_val = int(age_str)
                
                # Determine if this is a birth year or current age
                if 1900 <= age_val <= current_year:
                    # This is likely a birth year
                    birth_year = age_val
                    weight = 1.0  # High confidence for explicit birth years
                elif 1 <= age_val <= 120:
                    # This is likely a current age
                    birth_year = current_year - age_val
                    weight = 0.8  # Slightly lower confidence for calculated birth years
                else:
                    # Invalid age range
                    continue
                    
                birth_year_candidates.append((birth_year, weight))
                
            except ValueError:
                continue
                
        if not birth_year_candidates:
            return None
            
        # Group birth years within ±2 years (clustering)
        clusters = {}
        for birth_year, weight in birth_year_candidates:
            # Find existing cluster within ±2 years
            cluster_key = None
            for existing_key in clusters.keys():
                if abs(birth_year - existing_key) <= 2:
                    cluster_key = existing_key
                    break
                    
            if cluster_key is None:
                cluster_key = birth_year
                clusters[cluster_key] = []
                
            clusters[cluster_key].append((birth_year, weight))
            
        # Select cluster with highest total weight (majority vote with confidence)
        best_cluster = None
        best_score = 0.0
        
        for cluster_center, cluster_members in clusters.items():
            # Calculate cluster score: sum of weights * cluster size bonus
            total_weight = sum(weight for _, weight in cluster_members)
            cluster_size_bonus = len(cluster_members) * 0.1  # Small bonus for larger clusters
            cluster_score = total_weight + cluster_size_bonus
            
            if cluster_score > best_score:
                best_score = cluster_score
                best_cluster = cluster_members
                
        if best_cluster is None:
            return None
            
        # Calculate weighted average birth year within the best cluster
        total_weight = sum(weight for _, weight in best_cluster)
        weighted_birth_year = sum(birth_year * weight for birth_year, weight in best_cluster) / total_weight
        
        # Convert back to current age
        resolved_age = current_year - int(round(weighted_birth_year))
        
        # Calculate confidence score (0.0 to 1.0)
        max_possible_weight = len(age_matches) * 1.0  # If all were birth years
        confidence = min(1.0, best_score / max_possible_weight)
        
        # Ensure reasonable age bounds
        if not (1 <= resolved_age <= 120):
            return None
            
        return (resolved_age, confidence)


# Convenience function that merges title and body.

def detect_self_identification_in_entry(entry: Dict[str, Any], detector: "SelfIdentificationDetector") -> Dict[str, List[str]]:
    """Detect self identification inside a Reddit-style entry.

    The function is intentionally generic: only relies on *title* and *body* keys
    that are expected to exist across multiple data sources.
    """
    title = entry.get("title", "") or ""
    body = entry.get("selftext", "") or ""
    combined = f"{title}\n{body}"
    return detector.detect(combined)


def detect_self_identification_with_resolved_age(entry: Dict[str, Any], detector: "SelfIdentificationDetector") -> Dict[str, Any]:
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
        age_resolution = detector.resolve_multiple_ages(matches["age"])
        if age_resolution is not None:
            resolved_age, confidence = age_resolution
            matches["resolved_age"] = {
                "age": resolved_age,
                "confidence": confidence,
                "raw_matches": matches["age"].copy()
            }
    
    return matches 