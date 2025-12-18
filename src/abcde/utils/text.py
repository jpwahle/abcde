"""Text processing utilities."""

import re
from typing import List


def clean_text_newlines(text: str) -> str:
    """Clean newlines and extra whitespace from text."""
    if not text:
        return text
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


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
