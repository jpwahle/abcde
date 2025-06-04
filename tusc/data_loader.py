"""
TUSC-specific data loading and processing functions.
Handles parquet files with simpler single-machine processing.
"""
import logging
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime

from self_identification import SelfIdentificationDetector
from core.data_processing import detect_self_identification_in_tusc_entry, apply_linguistic_features

logger = logging.getLogger("tusc.data_loader")


def determine_tusc_split(input_file: str) -> str:
    """Auto-determine TUSC split type from filename."""
    filename = Path(input_file).name.lower()
    if 'city' in filename:
        return "city"
    elif 'country' in filename:
        return "country"
    else:
        logger.warning(f"Could not determine split type from filename '{filename}', defaulting to 'country'")
        return "country"


def process_tusc_batch_for_self_identification(
    df_batch: pd.DataFrame,
    detector: SelfIdentificationDetector,
    split: str,
    min_words: int,
    max_words: int,
) -> List[Dict[str, Any]]:
    """Process a batch of TUSC data and return entries that contain self-identification."""
    results: List[Dict[str, Any]] = []

    for _, row in df_batch.iterrows():
        # Convert row to dict
        entry = row.to_dict()
        
        # Basic filtering
        tweet_text = entry.get("Tweet", "")
        if not tweet_text or pd.isna(tweet_text):
            continue
            
        # Word count filtering
        word_count = len(tweet_text.split())
        if word_count < min_words or word_count > max_words:
            continue

        # Detect self-identification
        matches = detect_self_identification_in_tusc_entry(entry, detector)
        if not matches:
            continue  # no self-identification found

        # Skip entries with missing user information
        user_id = entry.get("UserID" if split == "country" else "userID", "")
        if not user_id or pd.isna(user_id):
            continue

        # Create result combining entry data with self-identification
        result = entry.copy()
        result["self_identification"] = matches
        results.append(result)

    return results


def process_tusc_batch_for_user_posts(
    df_batch: pd.DataFrame,
    target_user_ids: Set[str],
    split: str,
    include_features: bool = True
) -> pd.DataFrame:
    """Process TUSC batch to filter by users and optionally add linguistic features."""
    if df_batch.empty:
        return df_batch
    
    # Determine the user ID column based on split
    user_id_col = "UserID" if split == "country" else "userID"
    user_name_col = "UserName" if split == "country" else "userName"
    
    # Create a mask for rows where either user ID or user name matches
    id_mask = df_batch[user_id_col].astype(str).isin(target_user_ids)
    
    # Also check user name if available
    if user_name_col in df_batch.columns:
        name_mask = df_batch[user_name_col].astype(str).isin(target_user_ids)
        mask = id_mask | name_mask
    else:
        mask = id_mask
    
    filtered_df = df_batch[mask].copy()
    
    if filtered_df.empty:
        return filtered_df
    
    # Add datetime-derived columns based on split type
    if split == "city":
        def date_fn(x):
            if pd.isna(x):
                return None
            try:
                return datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y')
            except (ValueError, TypeError):
                return None
    else:  # country
        def parse_iso_date(x):
            if pd.isna(x):
                return None
            try:
                x_clean = x.replace('Z', '+00:00') if x.endswith('Z') else x
                return datetime.fromisoformat(x_clean)
            except (ValueError, TypeError, AttributeError):
                return None
        date_fn = parse_iso_date

    # Add additional datetime columns
    filtered_df['Day'] = filtered_df['createdAt'].apply(lambda x: date_fn(x).day if date_fn(x) is not None else None)
    filtered_df['Hour'] = filtered_df['createdAt'].apply(lambda x: date_fn(x).hour if date_fn(x) is not None else None)
    filtered_df['Weekday'] = filtered_df['createdAt'].apply(lambda x: date_fn(x).strftime('%A') if date_fn(x) is not None else None)

    # Add linguistic features if requested
    if include_features:
        # Apply linguistic features to tweet text
        tweet_column = 'Tweet'
        features_list = filtered_df[tweet_column].apply(
            lambda tw: apply_linguistic_features(tw, include_features=True)
        )
        
        # Convert to DataFrame and merge
        if not features_list.empty:
            features_df = pd.DataFrame(list(features_list))
            filtered_df = pd.concat([filtered_df, features_df], axis=1)
    
    return filtered_df


def load_tusc_file(
    input_file: str,
    detector: SelfIdentificationDetector = None,
    target_user_ids: Set[str] = None,
    split: str = None,
    min_words: int = 5,
    max_words: int = 1000,
    mode: str = "self_identification",  # or "user_posts"
    test_mode: bool = False,
    test_samples: int = 10000,
    include_features: bool = True
) -> pd.DataFrame:
    """Load TUSC parquet file for processing.
    
    Args:
        input_file: Path to parquet file
        detector: Self-identification detector (for self_identification mode)
        target_user_ids: Set of user IDs to filter (for user_posts mode)
        split: Data split type (auto-determined if None)
        min_words: Minimum word count (for self_identification mode)
        max_words: Maximum word count (for self_identification mode)
        mode: Processing mode ("self_identification" or "user_posts")
        test_mode: Whether to use test mode with limited samples
        test_samples: Number of samples in test mode
        include_features: Whether to include linguistic features (for user_posts mode)
        
    Returns:
        Processed DataFrame
    """
    if split is None:
        split = determine_tusc_split(input_file)
    
    logger.info(f"Loading TUSC file: {input_file} (split: {split}, mode: {mode})")
    
    # Load data
    if test_mode:
        df = pd.read_parquet(input_file, engine='pyarrow').head(test_samples)
        logger.info(f"Test mode: Processing {len(df)} samples")
    else:
        df = pd.read_parquet(input_file, engine='pyarrow')
        logger.info(f"Processing {len(df)} rows")
    
    if mode == "self_identification":
        if detector is None:
            raise ValueError("Detector required for self_identification mode")
            
        # Process in chunks to avoid memory issues
        chunk_size = 50000
        all_results = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_results = process_tusc_batch_for_self_identification(
                chunk, detector, split, min_words, max_words
            )
            all_results.extend(chunk_results)
            
        logger.info(f"Found {len(all_results)} self-identification entries")
        return pd.DataFrame(all_results)
        
    elif mode == "user_posts":
        if target_user_ids is None:
            raise ValueError("Target user IDs required for user_posts mode")
            
        result_df = process_tusc_batch_for_user_posts(
            df, target_user_ids, split, include_features
        )
        logger.info(f"Filtered to {len(result_df)} posts from target users")
        return result_df
        
    else:
        raise ValueError(f"Unknown mode: {mode}")