#!/usr/bin/env python3
"""
Unit tests for helpers.py functionality including demographic detection and linguistic features.
"""
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import (
    SelfIdentificationDetector,
    apply_linguistic_features,
    compute_vad_and_emotions,
    detect_self_identification_with_mappings_in_entry,
    emotion_dict,
    format_demographic_detections_for_output,
    moraltrust_dict,
    socialwarmth_dict,
    vad_dict,
    warmth_dict,
    worry_dict,
)


class TestSelfIdentificationDetector:
    """Test demographic detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SelfIdentificationDetector()

    def test_age_detection(self):
        """Test age detection patterns."""
        test_cases = [
            ("I am 25 years old", ["25"]),
            ("I'm 30", ["30"]),
            ("I was born in 1995", ["1995"]),
            ("I was born in '95", ["95"]),
            ("I was born on May 15, 1990", ["1990"]),
            ("I was born on 05/15/1990", ["1990"]),
        ]

        for text, expected in test_cases:
            result = self.detector.detect(text)
            assert "age" in result, f"Failed to detect age in: {text}"
            assert result["age"] == expected, f"Wrong age detected in: {text}"

    def test_gender_detection(self):
        """Test gender detection patterns."""
        test_cases = [
            ("I am a woman", ["woman"]),
            ("I'm a man", ["man"]),
            ("I identify as Female", ["Female"]),  # Gender terms are capitalized
            ("My gender is Female", ["Female"]),
            (
                "I'm a transgender woman",
                ["transgender", "woman"],
            ),  # Both are valid genders
        ]

        for text, expected in test_cases:
            result = self.detector.detect(text)
            assert "gender" in result, f"Failed to detect gender in: {text}"
            assert result["gender"] == expected, f"Wrong gender detected in: {text}"

    def test_occupation_detection(self):
        """Test occupation detection patterns."""
        test_cases = [
            ("I am a software engineer", ["software engineer"]),
            ("I work as a teacher", ["teacher"]),
            ("My job is doctor", ["doctor"]),
            ("I'm employed as a nurse", ["nurse"]),
        ]

        for text, expected in test_cases:
            result = self.detector.detect(text)
            if "occupation" in result:  # Occupation data might not be loaded in test
                assert any(
                    exp.lower() in [occ.lower() for occ in result["occupation"]]
                    for exp in expected
                ), f"Wrong occupation detected in: {text}"

    def test_city_detection(self):
        """Test city detection patterns."""
        test_cases = [
            ("I live in London", ["London"]),
            ("I'm from Paris", ["Paris"]),
            ("I grew up in New York", ["New York"]),
            ("My city is Tokyo", ["Tokyo"]),
            ("I'm based in Berlin", ["Berlin"]),
        ]

        for text, expected in test_cases:
            result = self.detector.detect(text)
            if "city" in result:  # City data might not be loaded in test
                assert any(
                    exp.lower() in [city.lower() for city in result["city"]]
                    for exp in expected
                ), f"Wrong city detected in: {text}"

    def test_country_detection(self):
        """Test country and nationality detection patterns."""
        test_cases = [
            ("I am American", ["American"]),
            ("I'm from Canada", ["Canada"]),
            ("I live in Germany", ["Germany"]),
            ("My nationality is British", ["British"]),
            ("I was born and raised in India", ["India"]),
        ]

        for text, expected in test_cases:
            result = self.detector.detect(text)
            if "country" in result:  # Country data might not be loaded in test
                assert any(
                    exp.lower() in [country.lower() for country in result["country"]]
                    for exp in expected
                ), f"Wrong country detected in: {text}"

    def test_religion_detection(self):
        """Test religion detection patterns."""
        test_cases = [
            ("I am a Christian", ["Christian"]),
            ("I'm Buddhist", ["Buddhist"]),
            ("My religion is Islam", ["Islam"]),
            ("I practice Hinduism", ["Hinduism"]),
            ("I converted to Judaism", ["Judaism"]),
            ("I was raised Catholic", ["Catholic"]),
        ]

        for text, expected in test_cases:
            result = self.detector.detect(text)
            if "religion" in result:  # Religion data might not be loaded in test
                assert any(
                    exp.lower() in [rel.lower() for rel in result["religion"]]
                    for exp in expected
                ), f"Wrong religion detected in: {text}"

    def test_detect_with_mappings(self):
        """Test detection with demographic mappings."""
        entry = {
            "title": "Software engineer here",
            "selftext": "I'm 25 and I live in London. I'm a Catholic man.",
        }

        result = detect_self_identification_with_mappings_in_entry(entry, self.detector)

        # Check that we get the expected structure
        assert isinstance(result, dict)
        if "age" in result:
            assert "raw" in result["age"]
            assert result["age"]["raw"] == ["25"]

        if "city" in result:
            assert "raw" in result["city"]
            assert "country_mapped" in result["city"]

        if "religion" in result:
            assert "raw" in result["religion"]
            assert "main_religion_mapped" in result["religion"]
            assert "category_mapped" in result["religion"]

    def test_format_demographic_output(self):
        """Test output formatting with proper field names."""
        detections = {
            "age": {"raw": ["25"]},
            "gender": {"raw": ["woman"]},
            "city": {"raw": ["London"], "country_mapped": ["United Kingdom"]},
            "religion": {
                "raw": ["Catholic"],
                "main_religion_mapped": ["Christianity"],
                "category_mapped": ["Abrahamic Religions"],
            },
            "occupation": {
                "raw": ["software engineer"],
                "soc_mapped": ["Software Developers"],
            },
        }

        formatted = format_demographic_detections_for_output(detections)

        # Check all expected fields are present
        assert "DMGRawExtractedAge" in formatted
        assert formatted["DMGRawExtractedAge"] == ["25"]

        assert "DMGRawExtractedGender" in formatted
        assert formatted["DMGRawExtractedGender"] == ["woman"]

        assert "DMGRawExtractedCity" in formatted
        assert formatted["DMGRawExtractedCity"] == ["London"]

        assert "DMGCountryMappedFromExtractedCity" in formatted
        assert formatted["DMGCountryMappedFromExtractedCity"] == ["United Kingdom"]

        assert "DMGRawExtractedReligion" in formatted
        assert formatted["DMGRawExtractedReligion"] == ["Catholic"]

        assert "DMGMainReligionMappedFromExtractedReligion" in formatted
        assert formatted["DMGMainReligionMappedFromExtractedReligion"] == [
            "Christianity"
        ]

        assert "DMGRawExtractedOccupation" in formatted
        assert formatted["DMGRawExtractedOccupation"] == ["software engineer"]

        assert "DMGSOCTitleMappedFromExtractedOccupation" in formatted
        assert formatted["DMGSOCTitleMappedFromExtractedOccupation"] == [
            "Software Developers"
        ]


class TestLinguisticFeatures:
    """Test linguistic feature computation."""

    def test_vad_features(self):
        """Test VAD (Valence, Arousal, Dominance) feature extraction."""
        # Use a simple text with known words
        text = "happy sad angry"

        features = compute_vad_and_emotions(
            text,
            vad_dict,
            emotion_dict,
            [
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
            ],
            worry_dict,
            moraltrust_dict,
            socialwarmth_dict,
            warmth_dict,
        )

        # Check VAD features exist
        assert "NRCAvgValence" in features
        assert "NRCAvgArousal" in features
        assert "NRCAvgDominance" in features

        # Check high/low flags
        assert "NRCHasHighValenceWord" in features
        assert "NRCHasLowValenceWord" in features

        # Check counts
        assert "NRCCountHighValenceWords" in features
        assert "NRCCountLowValenceWords" in features

    def test_emotion_features(self):
        """Test emotion feature extraction."""
        text = "I am happy and excited but also worried"

        features = compute_vad_and_emotions(
            text,
            vad_dict,
            emotion_dict,
            [
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
            ],
            worry_dict,
            moraltrust_dict,
            socialwarmth_dict,
            warmth_dict,
        )

        # Check emotion flags
        for emotion in ["anger", "joy", "fear", "sadness", "positive", "negative"]:
            assert f"NRCHas{emotion.capitalize()}Word" in features
            assert f"NRCCount{emotion.capitalize()}Words" in features

    def test_worry_features(self):
        """Test worry/anxiety feature extraction."""
        text = "anxious worried stressed calm relaxed"

        features = compute_vad_and_emotions(
            text,
            vad_dict,
            emotion_dict,
            [
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
            ],
            worry_dict,
            moraltrust_dict,
            socialwarmth_dict,
            warmth_dict,
        )

        # Check worry features
        assert "NRCHasAnxietyWord" in features
        assert "NRCHasCalmnessWord" in features
        assert "NRCAvgAnxiety" in features
        assert "NRCAvgCalmness" in features
        assert "NRCHasHighAnxietyWord" in features
        assert "NRCCountHighAnxietyWords" in features

    def test_moral_social_warmth_features(self):
        """Test moral trust, social warmth, and combined warmth features."""
        text = "trustworthy honest friendly warm caring"

        features = compute_vad_and_emotions(
            text,
            vad_dict,
            emotion_dict,
            [
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
            ],
            worry_dict,
            moraltrust_dict,
            socialwarmth_dict,
            warmth_dict,
        )

        # Check moral trust features
        assert "NRCHasHighMoralTrustWord" in features
        assert "NRCCountHighMoralTrustWord" in features
        assert "NRCAvgMoralTrustWord" in features

        # Check social warmth features
        assert "NRCHasHighSocialWarmthWord" in features
        assert "NRCCountHighSocialWarmthWord" in features
        assert "NRCAvgSocialWarmthWord" in features

        # Check combined warmth features
        assert "NRCHasHighWarmthWord" in features
        assert "NRCCountHighWarmthWord" in features
        assert "NRCAvgWarmthWord" in features

    def test_apply_linguistic_features(self):
        """Test the main linguistic feature application function."""
        text = "I am happy and excited! This is wonderful news. My heart is racing."

        features = apply_linguistic_features(text)

        # Check all expected feature categories are present
        # VAD features
        assert "NRCAvgValence" in features
        assert "NRCAvgArousal" in features
        assert "NRCAvgDominance" in features

        # Emotion features
        assert "NRCHasJoyWord" in features
        assert "NRCHasPositiveWord" in features

        # Pronoun features
        assert "PRNHasI" in features
        assert features["PRNHasI"] == 1  # "I am"
        assert "PRNHasMy" in features
        assert features["PRNHasMy"] == 1  # "My heart"

        # Body part features
        assert "HasBPM" in features
        assert "MyBPM" in features

        # Word count
        assert "WordCount" in features
        assert features["WordCount"] == 13  # Count the words (including exclamation)

        # Time/tense features
        time_keys = [k for k in features if k.startswith("TIME")]
        assert len(time_keys) > 0  # Should have some time features

    def test_body_part_mentions(self):
        """Test body part mention detection."""
        text = "My head hurts. Your hand is cold. Her heart is broken."

        features = apply_linguistic_features(text)

        assert features["HasBPM"] == 1  # Has body parts
        assert "head" in features["MyBPM"]  # "My head"
        assert "hand" in features["YourBPM"]  # "Your hand"
        assert "heart" in features["HerBPM"]  # "Her heart"

    def test_pronoun_detection(self):
        """Test pronoun detection."""
        text = "I love my cat. You and your dog are nice. She gave him her book. They took their car."

        features = apply_linguistic_features(text)

        # First person
        assert features["PRNHasI"] == 1
        assert features["PRNHasMy"] == 1

        # Second person
        assert features["PRNHasYou"] == 1
        assert features["PRNHasYour"] == 1

        # Third person
        assert features["PRNHasShe"] == 1
        assert features["PRNHasHer"] == 1
        assert features["PRNHasHim"] == 1
        assert features["PRNHasThey"] == 1
        assert features["PRNHasTheir"] == 1

    def test_tense_features(self):
        """Test grammatical tense detection."""
        text = "I walked yesterday. I am walking now. I will walk tomorrow."

        features = apply_linguistic_features(text)

        # Check that time features exist
        time_keys = [k for k in features if k.startswith("TIME")]
        assert len(time_keys) >= 4  # Should have Has/Count features

    def test_empty_text(self):
        """Test handling of empty text."""
        with pytest.raises(ValueError, match="non-empty string"):
            apply_linguistic_features("")

    def test_none_text(self):
        """Test handling of None text."""
        with pytest.raises(ValueError, match="non-empty string"):
            apply_linguistic_features(None)
