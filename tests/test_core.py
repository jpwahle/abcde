"""Tests for core functionality."""

import pytest


class TestSelfIdentificationDetector:
    """Tests for SelfIdentificationDetector."""

    def test_import(self):
        """Test that the detector can be imported."""
        from abcde import SelfIdentificationDetector

        detector = SelfIdentificationDetector()
        assert detector is not None

    def test_detect_age(self):
        """Test age detection."""
        from abcde import SelfIdentificationDetector

        detector = SelfIdentificationDetector()

        text = "I am 25 years old"
        matches = detector.detect(text)
        assert "age" in matches
        assert "25" in matches["age"]

    def test_detect_city(self):
        """Test city detection."""
        from abcde import SelfIdentificationDetector

        detector = SelfIdentificationDetector()

        text = "I live in London"
        matches = detector.detect(text)
        assert "city" in matches

    def test_detect_with_mappings(self):
        """Test detection with mappings."""
        from abcde import SelfIdentificationDetector

        detector = SelfIdentificationDetector()

        text = "I live in London"
        detailed = detector.detect_with_mappings(text)
        assert "city" in detailed
        assert "country_mapped" in detailed["city"]


class TestFeatures:
    """Tests for feature extraction."""

    def test_apply_linguistic_features(self):
        """Test linguistic feature extraction."""
        from abcde import apply_linguistic_features

        text = "I am happy and excited about this wonderful day!"
        features = apply_linguistic_features(text)

        assert "WordCount" in features
        assert features["WordCount"] > 0
        assert "NRCAvgValence" in features

    def test_empty_text_raises(self):
        """Test that empty text raises an error."""
        from abcde import apply_linguistic_features

        with pytest.raises(ValueError):
            apply_linguistic_features("")


class TestBackwardCompatibility:
    """Tests for backward-compatible helpers module."""

    def test_helpers_imports(self):
        """Test that helpers module imports work."""
        from helpers import (
            SelfIdentificationDetector,
            apply_linguistic_features,
            print_banner,
        )

        assert SelfIdentificationDetector is not None
        assert apply_linguistic_features is not None
        assert print_banner is not None
