"""
ABCDE - Age-Based Corpus of Demographic Expressions

A library for processing large textual datasets to extract demographic
self-identification and compute linguistic features.
"""

__version__ = "0.1.0"

from abcde.core.detector import SelfIdentificationDetector
from abcde.core.pii import PIIDetector
from abcde.core.features import apply_linguistic_features
from abcde.utils.banner import print_banner

__all__ = [
    "SelfIdentificationDetector",
    "PIIDetector",
    "apply_linguistic_features",
    "print_banner",
]

