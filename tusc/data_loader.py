"""
TUSC-specific data loading and processing functions.
Handles parquet files with distributed parallel processing using Dask.
"""
import logging
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd
import dask.bag as db
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime
from dask.diagnostics import ProgressBar

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
    import gc
    
    if df_batch.empty:
        return []
    
    results: List[Dict[str, Any]] = []
    
    # Pre-filter on word count and non-empty tweets using vectorized operations
    tweet_series = df_batch["Tweet"].fillna("")
    word_counts = tweet_series.str.split().str.len().fillna(0)
    
    # Create mask for valid tweets
    valid_mask = (
        (tweet_series != "") & 
        (word_counts >= min_words) & 
        (word_counts <= max_words)
    )
    
    # Filter dataframe early
    filtered_df = df_batch[valid_mask].copy()
    
    # Clear memory
    del df_batch, tweet_series, word_counts, valid_mask
    gc.collect()
    
    if filtered_df.empty:
        return []

    # Process remaining rows (already filtered)
    user_id_col = "UserID" if split == "country" else "userID"
    
    for _, row in filtered_df.iterrows():
        # Convert row to dict
        entry = row.to_dict()
        
        # Skip entries with missing user information
        user_id = entry.get(user_id_col, "")
        if not user_id or pd.isna(user_id):
            continue

        # Detect self-identification
        matches = detect_self_identification_in_tusc_entry(entry, detector)
        if not matches:
            continue  # no self-identification found

        # Create result combining entry data with self-identification
        result = entry.copy()
        result["self_identification"] = matches
        results.append(result)

    # Clean up
    del filtered_df
    gc.collect()
    
    return results


def process_tusc_batch_for_user_posts(
    df_batch: pd.DataFrame,
    target_user_ids: Set[str],
    split: str,
    include_features: bool = True
) -> pd.DataFrame:
    """Process TUSC batch to filter by users and optionally add linguistic features."""
    import gc
    
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
    
    # Early exit if no matches to save memory
    if filtered_df.empty:
        del df_batch, mask, id_mask
        gc.collect()
        return filtered_df
    
    # Clear original dataframe to free memory
    del df_batch, mask, id_mask
    if 'name_mask' in locals():
        del name_mask
    
    # Add datetime-derived columns based on split type using vectorized operations
    if split == "city":
        # Vectorized datetime parsing for city format
        try:
            filtered_df['parsed_datetime'] = pd.to_datetime(
                filtered_df['createdAt'], 
                format='%a %b %d %H:%M:%S %z %Y', 
                errors='coerce'
            )
        except:
            # Fallback to individual parsing
            filtered_df['parsed_datetime'] = filtered_df['createdAt'].apply(
                lambda x: pd.to_datetime(x, format='%a %b %d %H:%M:%S %z %Y', errors='coerce') if not pd.isna(x) else None
            )
    else:  # country
        # Vectorized datetime parsing for country format (ISO)
        filtered_df['createdAt_clean'] = filtered_df['createdAt'].str.replace('Z', '+00:00')
        filtered_df['parsed_datetime'] = pd.to_datetime(filtered_df['createdAt_clean'], errors='coerce')
        filtered_df.drop('createdAt_clean', axis=1, inplace=True)

    # Extract date components vectorized
    filtered_df['Day'] = filtered_df['parsed_datetime'].dt.day
    filtered_df['Hour'] = filtered_df['parsed_datetime'].dt.hour  
    filtered_df['Weekday'] = filtered_df['parsed_datetime'].dt.day_name()
    filtered_df.drop('parsed_datetime', axis=1, inplace=True)

    # Add linguistic features if requested
    if include_features and len(filtered_df) > 0:
        # Process in smaller batches to control memory
        batch_size = 1000
        tweet_column = 'Tweet'
        all_features = []
        
        for i in range(0, len(filtered_df), batch_size):
            batch_tweets = filtered_df[tweet_column].iloc[i:i+batch_size]
            batch_features = batch_tweets.apply(
                lambda tw: apply_linguistic_features(tw, include_features=True)
            )
            all_features.extend(list(batch_features))
            
            # Force garbage collection between batches
            del batch_tweets, batch_features
            gc.collect()
        
        # Convert to DataFrame and merge
        if all_features:
            features_df = pd.DataFrame(all_features)
            filtered_df = pd.concat([filtered_df, features_df], axis=1)
            del features_df, all_features
            gc.collect()
    
    return filtered_df


def create_parquet_chunks(
    input_file: str,
    chunk_size: int = 100000
) -> List[Dict[str, Any]]:
    """Create chunks from parquet file for parallel processing.
    
    Returns list of chunk metadata with row ranges for each partition.
    """
    # Get total number of rows
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    
    chunks = []
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunks.append({
            "file_path": input_file,
            "start_row": start,
            "end_row": end,
            "num_rows": end - start
        })
    
    logger.info(f"Created {len(chunks)} chunks from parquet file with {total_rows} total rows")
    return chunks


def _read_parquet_chunk_safe(file_path: str, start_row: int, end_row: int) -> pd.DataFrame:
    """Safely read a specific chunk of rows from parquet file."""
    try:
        # Try to use pyarrow streaming for memory efficiency
        with pq.ParquetFile(file_path) as pf:
            # Calculate which row groups we need
            row_groups_to_read = []
            current_row = 0
            
            for i in range(pf.num_row_groups):
                rg_metadata = pf.metadata.row_group(i)
                rg_end = current_row + rg_metadata.num_rows
                
                # Check if this row group overlaps with our target range
                if current_row < end_row and rg_end > start_row:
                    row_groups_to_read.append(i)
                
                current_row = rg_end
                if current_row >= end_row:
                    break
            
            if not row_groups_to_read:
                return pd.DataFrame()
            
            # Read only the necessary row groups
            table = pf.read_row_groups(row_groups_to_read)
            df = table.to_pandas()
            
            # Now slice to exact range if needed
            if len(df) > 0:
                actual_start = max(0, start_row - sum(pf.metadata.row_group(i).num_rows for i in range(row_groups_to_read[0])))
                actual_end = min(len(df), end_row - start_row + actual_start)
                df = df.iloc[actual_start:actual_end]
            
            return df
            
    except Exception as e:
        logger.warning(f"Failed optimized parquet reading, falling back to full read: {e}")
        # Fallback to reading full file and slicing (less memory efficient but reliable)
        df_full = pd.read_parquet(file_path, engine='pyarrow')
        return df_full.iloc[start_row:end_row]


def process_tusc_chunk_for_self_identification_worker(
    chunk_info: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
) -> List[Dict[str, Any]]:
    """Process a single parquet chunk for self-identification detection.
    
    Creates detector locally to avoid large object transfer to workers.
    """
    # Create detector locally on worker to avoid serialization overhead
    from self_identification import SelfIdentificationDetector
    detector = SelfIdentificationDetector()
    
    df_chunk = _read_parquet_chunk_safe(
        chunk_info["file_path"],
        chunk_info["start_row"],
        chunk_info["end_row"]
    )
    
    return process_tusc_batch_for_self_identification(
        df_chunk, detector, split, min_words, max_words
    )


def process_tusc_chunk_for_user_posts(
    chunk_info: Dict[str, Any],
    target_user_ids: Set[str],
    split: str,
    include_features: bool = True
) -> pd.DataFrame:
    """Process a single parquet chunk for user post collection."""
    df_chunk = _read_parquet_chunk_safe(
        chunk_info["file_path"],
        chunk_info["start_row"],
        chunk_info["end_row"]
    )
    
    return process_tusc_batch_for_user_posts(
        df_chunk, target_user_ids, split, include_features
    )


def load_tusc_files_for_self_identification(
    input_file: str,
    detector: SelfIdentificationDetector,
    split: str = "country",
    min_words: int = 5,
    max_words: int = 1000,
    chunk_size: int = 100000,
    client=None
) -> List[Dict[str, Any]]:
    """Load and process TUSC parquet file for self-identification using Dask."""
    chunks = create_parquet_chunks(input_file, chunk_size)
    if not chunks:
        raise ValueError(f"No data found in parquet file: {input_file}")
    
    logger.info(f"Processing {len(chunks)} chunks for self-identification detection.")

    bag = db.from_sequence(chunks, npartitions=len(chunks))
    processed_bag = bag.map(
        lambda chunk: process_tusc_chunk_for_self_identification_worker(
            chunk, split=split, min_words=min_words, max_words=max_words
        )
    ).flatten()

    with ProgressBar():
        results: List[Dict[str, Any]] = processed_bag.compute()

    return results


def load_tusc_files_for_user_posts(
    input_file: str,
    target_user_ids: Set[str],
    split: str = "country", 
    include_features: bool = True,
    chunk_size: int = 100000,
    client=None
) -> List[Dict[str, Any]]:
    """Load and process TUSC parquet file for user posts using Dask."""
    chunks = create_parquet_chunks(input_file, chunk_size)
    if not chunks:
        raise ValueError(f"No data found in parquet file: {input_file}")
        
    logger.info(f"Processing {len(chunks)} chunks for user post collection.")

    bag = db.from_sequence(chunks, npartitions=len(chunks))
    processed_bag = bag.map(
        lambda chunk: process_tusc_chunk_for_user_posts(
            chunk, target_user_ids=target_user_ids, split=split, include_features=include_features
        )
    )

    with ProgressBar():
        results_dfs = processed_bag.compute()

    # Combine all DataFrames
    non_empty_dfs = [df for df in results_dfs if not df.empty]
    if non_empty_dfs:
        combined_df = pd.concat(non_empty_dfs, ignore_index=True)
        return combined_df.to_dict('records')
    else:
        return []


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
    include_features: bool = True,
    chunk_size: int = 100000,
    client=None
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
        chunk_size: Chunk size for parallel processing
        client: Dask client for parallel processing
        
    Returns:
        Processed DataFrame
    """
    if split is None:
        split = determine_tusc_split(input_file)
    
    logger.info(f"Loading TUSC file: {input_file} (split: {split}, mode: {mode})")
    
    # Use parallel processing for large datasets, single-machine for test mode
    if test_mode:
        # Load data in test mode (small sample)
        df = pd.read_parquet(input_file, engine='pyarrow').head(test_samples)
        logger.info(f"Test mode: Processing {len(df)} samples")
        
        if mode == "self_identification":
            if detector is None:
                raise ValueError("Detector required for self_identification mode")
                
            # Process in chunks to avoid memory issues
            local_chunk_size = 50000
            all_results = []
            
            for i in range(0, len(df), local_chunk_size):
                chunk = df.iloc[i:i+local_chunk_size]
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
    
    else:
        # Use parallel processing for large datasets
        if mode == "self_identification":
            if detector is None:
                raise ValueError("Detector required for self_identification mode")
                
            results = load_tusc_files_for_self_identification(
                input_file=input_file,
                detector=detector,
                split=split,
                min_words=min_words,
                max_words=max_words,
                chunk_size=chunk_size,
                client=client
            )
            logger.info(f"Found {len(results)} self-identification entries")
            return pd.DataFrame(results)
            
        elif mode == "user_posts":
            if target_user_ids is None:
                raise ValueError("Target user IDs required for user_posts mode")
                
            results = load_tusc_files_for_user_posts(
                input_file=input_file,
                target_user_ids=target_user_ids,
                split=split,
                include_features=include_features,
                chunk_size=chunk_size,
                client=client
            )
            logger.info(f"Filtered to {len(results)} posts from target users")
            return pd.DataFrame(results)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")