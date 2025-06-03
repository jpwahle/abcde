#!/usr/bin/env python3
"""Test script for the MyBPM, YourBPM extraction functionality."""

from compute_features import compute_all_features, load_body_parts

def test_bpm_extraction():
    """Test the body part mention extraction."""
    
    # Test text with various pronouns and body parts
    test_text = "I hurt my hand yesterday. Your face looks good. Her leg is broken. His head aches. Their eyes are beautiful. The dog has a tail."
    
    # Load body parts
    body_parts = load_body_parts('data/bodywords-full.txt')
    
    # Compute features
    features = compute_all_features(test_text, body_parts=body_parts)
    
    # Print BPM features
    print("Body Part Mention Features:")
    print(f"My BPM: '{features.get('My BPM', '')}'")
    print(f"Your BPM: '{features.get('Your BPM', '')}'") 
    print(f"Her BPM: '{features.get('Her BPM', '')}'")
    print(f"His BPM: '{features.get('His BPM', '')}'")
    print(f"Their BPM: '{features.get('Their BPM', '')}'")
    print(f"HasBPM: {features.get('HasBPM', 0)}")
    
    print("\nPronoun Features:")
    pronoun_keys = [k for k in features.keys() if k.startswith('PRNHas')]
    for key in sorted(pronoun_keys):
        if features[key] == 1:
            print(f"{key}: {features[key]}")

if __name__ == "__main__":
    test_bpm_extraction()