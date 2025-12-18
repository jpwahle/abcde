"""PII (Personally Identifiable Information) detection."""

from typing import Any, Dict, List, Optional

from presidio_analyzer import AnalyzerEngine


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
    # Import here to avoid circular imports
    from abcde.io.jsonl_utils import extract_columns

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
