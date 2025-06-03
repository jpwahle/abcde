import re
from typing import Dict, List, Pattern, Any


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
                re.compile(r"\bI\s+(?:am|'m)\s+(\d{1,2})\s+years?\s+old\b", re.I),
                # I am 24 / I'm 24 (only followed by age-related words or sentence boundaries, no symbols)
                re.compile(r"\bI\s+(?:am|'m)\s+(\d{1,2})(?=\s+(?:years?(?:\s+old|-old)?|yo|yrs?)\b|\s*[.!?;,]|\s*$)(?!\s*[%$#@&*+=<>()[\]{}|\\~`^_])", re.I),
                # I was born in 1998 / I am born in 1998 (4-digit birth years)
                re.compile(r"\bI\s+(?:was|am|'m)\s+born\s+in\s+(19\d{2}|20\d{2})\b", re.I),
                # I was born on 14 July 1992 (birth year from date)
                re.compile(r"\bI\s+was\s+born\s+on\s+\d{1,2}\s+\w+\s+(19\d{2}|20\d{2})\b", re.I),
                # Additional age patterns
                # I'm turning 25 / I turn 25
                re.compile(r"\bI\s+(?:'m\s+turning|turn|turned)\s+(\d{1,2})(?=\s*[.!?;,]|\s*$)(?!\s*[%$#@&*+=<>()[\]{}|\\~`^_])", re.I),
                # At 24 / At age 24
                re.compile(r"\bat\s+(?:age\s+)?(\d{1,2})(?=\s*[.!?;,]|\s*$)(?!\s*[%$#@&*+=<>()[\]{}|\\~`^_])", re.I),
                # 24M / 24F / 24m / 24f (age with gender)
                re.compile(r"\b(\d{1,2})[MFmf]\b", re.I),
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