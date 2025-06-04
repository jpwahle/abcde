"""
Core data processing functions shared between Reddit and TUSC pipelines.
Contains common functionality for self-identification detection and feature extraction.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from self_identification import SelfIdentificationDetector, detect_self_identification_with_resolved_age
from compute_features import compute_all_features

logger = logging.getLogger("core.data_processing")


def detect_self_identification_in_tusc_entry(entry: Dict[str, Any], detector: SelfIdentificationDetector) -> Dict[str, Any]:
    """Detect self identification inside a TUSC entry.
    
    TUSC entries have 'Tweet' field instead of 'title' and 'selftext'.
    """
    tweet_text = entry.get("Tweet", "") or ""
    
    # Create a minimal entry structure compatible with detector
    adapted_entry = {
        "title": "",  # TUSC doesn't have titles
        "selftext": tweet_text  # Use tweet text as body
    }
    
    return detect_self_identification_with_resolved_age(adapted_entry, detector)


def flatten_result_to_csv_row(result: Dict[str, Any], data_source: str = "reddit") -> Dict[str, Any]:
    """Flatten nested result structure for CSV output with CapitalCase headers."""
    row = {}
    
    # Author information - adapt to data source
    if data_source == "tusc":
        # TUSC uses different user ID fields depending on split
        row["Author"] = result.get("UserID") or result.get("userID", "")
        row["AuthorName"] = result.get("UserName") or result.get("userName", "")
    else:  # reddit
        row["Author"] = result.get("author", "")
    
    # Self-identification with resolved age
    self_id = result.get("self_identification", {})
    
    # Use resolved age if available, otherwise fallback to first raw age
    if "resolved_age" in self_id:
        resolved = self_id["resolved_age"]
        row["SelfIdentificationAgeMajorityVote"] = resolved.get("age", "")
        row["SelfIdentificationRawAges"] = "|".join(map(str, resolved.get("raw_matches", [])))
    else:
        # Fallback to first age from raw matches
        age_matches = self_id.get("age", [])
        row["SelfIdentificationAgeMajorityVote"] = age_matches[0] if age_matches else ""
        row["SelfIdentificationRawAges"] = "|".join(map(str, age_matches))
    
    # Post information with Post prefix - adapt to data source
    if data_source == "tusc":
        row["PostID"] = result.get("TweetID", "")
        row["PostText"] = result.get("Tweet", "")
        row["PostCreatedAt"] = result.get("createdAt", "")
        row["PostYear"] = result.get("Year", "")
        row["PostMonth"] = result.get("Month", "")
        
        # Location info varies by TUSC split
        if "City" in result:  # city split
            row["PostCity"] = result.get("City", "")
        if "Country" in result:  # country split
            row["PostCountry"] = result.get("Country", "")
            row["PostMyCountry"] = result.get("MyCountry", "")
        row["PostPlace"] = result.get("Place", "")
        row["PostPlaceID"] = result.get("PlaceID", "")
        row["PostPlaceType"] = result.get("PlaceType", "")
    else:  # reddit
        post = result.get("post", {})
        row["PostID"] = post.get("id", "")
        row["PostSubreddit"] = post.get("subreddit", "")
        row["PostTitle"] = post.get("title", "")
        row["PostSelftext"] = post.get("selftext", "")
        row["PostCreatedUtc"] = post.get("created_utc", "")
        row["PostScore"] = post.get("score", "")
        row["PostNumComments"] = post.get("num_comments", "")
        row["PostPermalink"] = post.get("permalink", "")
        row["PostUrl"] = post.get("url", "")
        row["PostMediaPath"] = post.get("media_path", "")
    
    return row


def get_csv_fieldnames(data_source: str, split: str = None) -> List[str]:
    """Get appropriate CSV fieldnames based on data source and split."""
    if data_source == "tusc":
        base_fieldnames = [
            "Author", "AuthorName", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
            "PostID", "PostText", "PostCreatedAt", "PostYear", "PostMonth"
        ]
        
        if split == "city":
            location_fieldnames = ["PostCity", "PostPlace", "PostPlaceID", "PostPlaceType"]
        else:  # country
            location_fieldnames = ["PostCountry", "PostMyCountry", "PostPlace", "PostPlaceID", "PostPlaceType"]
        
        return base_fieldnames + location_fieldnames
    else:  # reddit
        return [
            "Author", "SelfIdentificationAgeMajorityVote", "SelfIdentificationRawAges", 
            "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
            "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath"
        ]


def apply_linguistic_features(text: str, include_features: bool = True) -> Dict[str, Any]:
    """Apply linguistic feature computation to text if enabled."""
    if not include_features or not text:
        return {}
    
    try:
        from compute_features import (
            vad_dict, emotion_dict, emotions, worry_dict, tense_dict,
            moraltrust_dict, socialwarmth_dict, warmth_dict, compute_all_features
        )
        
        return compute_all_features(
            text, vad_dict, emotion_dict, emotions, worry_dict, tense_dict,
            moraltrust_dict, socialwarmth_dict, warmth_dict
        )
    except ImportError:
        logger.warning("Feature computation modules not available, skipping linguistic features")
        return {}