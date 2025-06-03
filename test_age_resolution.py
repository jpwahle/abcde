#!/usr/bin/env python3
"""Test script for the multiple age resolution functionality."""

from self_identification import SelfIdentificationDetector, detect_self_identification_with_resolved_age


def test_age_resolution():
    """Test various age resolution scenarios."""
    detector = SelfIdentificationDetector()
    
    test_cases = [
        {
            "name": "Single age",
            "text": "I am 25 years old",
            "expected_age": 25,
            "expected_confidence_range": (0.8, 1.0)
        },
        {
            "name": "Multiple consistent ages",
            "text": "I'm 30 and was born in 1995",  # Should resolve to ~30 (2025-1995=30)
            "expected_age": 30,
            "expected_confidence_range": (0.8, 1.0)
        },
        {
            "name": "Multiple inconsistent ages",
            "text": "I'm 25 but was born in 1990",  # 25 vs 35, should pick majority
            "expected_age": None,  # Could be either, depends on clustering
            "expected_confidence_range": (0.5, 1.0)
        },
        {
            "name": "Birth year only",
            "text": "I was born in 1992",
            "expected_age": 33,  # 2025 - 1992
            "expected_confidence_range": (0.9, 1.0)
        },
        {
            "name": "Age with gender marker",
            "text": "24M looking for advice",
            "expected_age": 24,
            "expected_confidence_range": (0.8, 1.0)
        }
    ]
    
    print("Testing Age Resolution")
    print("=" * 40)
    
    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"Text: '{case['text']}'")
        
        # Create mock entry
        entry = {"title": case['text'], "selftext": ""}
        
        # Test detection
        matches = detect_self_identification_with_resolved_age(entry, detector)
        
        print(f"Raw matches: {matches.get('age', [])}")
        
        if "resolved_age" in matches:
            resolved = matches["resolved_age"]
            print(f"Resolved age: {resolved['age']}")
            print(f"Confidence: {resolved['confidence']:.3f}")
            print(f"Raw matches: {resolved['raw_matches']}")
            
            # Check expectations
            if case["expected_age"] is not None:
                if resolved["age"] == case["expected_age"]:
                    print("✓ Age matches expected")
                else:
                    print(f"✗ Age mismatch: expected {case['expected_age']}, got {resolved['age']}")
            
            conf_min, conf_max = case["expected_confidence_range"]
            if conf_min <= resolved["confidence"] <= conf_max:
                print("✓ Confidence in expected range")
            else:
                print(f"✗ Confidence out of range: expected {conf_min}-{conf_max}, got {resolved['confidence']}")
        else:
            print("No resolved age found")
        
        print("-" * 30)


def test_edge_cases():
    """Test edge cases for age resolution."""
    detector = SelfIdentificationDetector()
    
    print("\n\nTesting Edge Cases")
    print("=" * 40)
    
    edge_cases = [
        # Test direct method calls
        (["25", "26", "27"], "Close ages should cluster"),
        (["25", "1998"], "Age + birth year should be consistent"),
        (["20", "30", "40"], "Scattered ages should pick best cluster"),
        (["1995", "1996", "1997"], "Multiple birth years should cluster"),
        (["abc", "25"], "Invalid values should be filtered"),
        ([], "Empty list should return None"),
        (["999"], "Invalid age should return None"),
    ]
    
    for ages, description in edge_cases:
        print(f"\nTest: {description}")
        print(f"Input: {ages}")
        
        result = detector.resolve_multiple_ages(ages, current_year=2025)
        
        if result:
            age, confidence = result
            print(f"Resolved: age={age}, confidence={confidence:.3f}")
        else:
            print("Resolved: None")


if __name__ == "__main__":
    test_age_resolution()
    test_edge_cases()