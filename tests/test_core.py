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


class TestBodyPartMentions:
    """Tests for body part mention (BPM) features."""

    def test_whole_word_match(self):
        """Body parts as standalone words are detected."""
        from abcde.core import compute_prefixed_body_part_mentions

        res = compute_prefixed_body_part_mentions("I broke my foot yesterday")
        assert res["HasBPM"] is True
        assert "my foot" in res["MyBPM"]

    def test_multiword_body_part(self):
        """Multi-word lexicon entries are detected."""
        from abcde.core import compute_prefixed_body_part_mentions

        res = compute_prefixed_body_part_mentions("her belly button piercing")
        assert res["HasBPM"] is True
        assert "her belly button" in res["HerBPM"]

    def test_no_partial_word_match(self):
        """Body parts embedded in longer words do not count (issue
        reported for HasBPM: 'Lightfoot' -> 'foot', 'Liverpool' ->
        'liver', 'lmaolippi' -> 'lip', 'Columbus' -> 'lumbus')."""
        from abcde.core import compute_prefixed_body_part_mentions

        false_positives = [
            "@ThatEricAlper Carefree highway Gordon Lightfoot",
            "City losing to South Hampton after they crucified Liverpool\U0001F62A .",
            "@lmaolippi do you think she cares ? lol",
            "@runner6565 @CatharineMc @CTVNews i guess that’s why there’s "
            "was a statue in Baltimore- home of the Star Spangled Banner "
            "\U0001F1FA\U0001F1F8 and Columbus’s last stand",
        ]
        for text in false_positives:
            res = compute_prefixed_body_part_mentions(text)
            assert res["HasBPM"] is False, text

    def test_prefix_requires_word_boundary(self):
        """A possessive prefix inside another word ('jimmy foot') does
        not count as 'my foot', but the standalone body part still
        sets HasBPM."""
        from abcde.core import compute_prefixed_body_part_mentions

        res = compute_prefixed_body_part_mentions("jimmy foot race")
        assert res["HasBPM"] is True
        assert res["MyBPM"] == ""

    def test_punctuation_adjacent_match(self):
        """Punctuation next to a body part does not block the match."""
        from abcde.core import compute_prefixed_body_part_mentions

        res = compute_prefixed_body_part_mentions("My arm, it hurts!")
        assert res["HasBPM"] is True
        assert "my arm" in res["MyBPM"]


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
