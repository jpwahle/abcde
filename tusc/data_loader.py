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


def create_parquet_batch_specs(
    input_file: str,
    batch_size: int = 100000
) -> List[Dict[str, Any]]:
    """Create batch specifications for parallel processing of parquet file.
    
    Returns list of batch metadata for distributed processing.
    """
    # Get total number of rows
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    total_batches = total_rows // batch_size + 1
    
    batch_specs = []
    for batch_idx in range(total_batches):
        start_row = batch_idx * batch_size
        end_row = min(start_row + batch_size, total_rows)
        
        if start_row < total_rows:  # Only add if there are rows to process
            batch_specs.append({
                "file_path": input_file,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "start_row": start_row,
                "end_row": end_row,
                "num_rows": end_row - start_row
            })
    
    logger.info(f"Created {len(batch_specs)} batch specs from parquet file with {total_rows} total rows")
    return batch_specs


def process_parquet_in_batches(
    input_file: str,
    batch_size: int,
    process_batch_fn,
    **process_kwargs
) -> List[Any]:
    """Process parquet file batch-by-batch using streaming approach."""
    import gc
    from tqdm import tqdm
    
    results = []
    
    with pq.ParquetFile(input_file) as parquet_file:
        total_batches = parquet_file.metadata.num_rows // batch_size + 1
        
        logger.info(f"Processing {parquet_file.metadata.num_rows} rows in {total_batches} batches of {batch_size}")
        
        for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), total=total_batches, desc="Processing batches"):
            # Convert Arrow batch to pandas DataFrame
            df_batch = batch.to_pandas()
            
            if df_batch.empty:
                continue
            
            # Process the batch
            batch_results = process_batch_fn(df_batch, **process_kwargs)
            
            if batch_results:
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                elif isinstance(batch_results, pd.DataFrame):
                    if not batch_results.empty:
                        results.append(batch_results)
                else:
                    results.append(batch_results)
            
            # Force garbage collection after each batch
            del df_batch, batch_results
            gc.collect()
    
    return results


def process_tusc_batch_for_self_identification_worker(
    batch_spec: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
    output_dir: str,
    max_entries_in_memory: int = 10000
) -> str:
    """Process a single parquet batch for self-identification detection using streaming.
    
    Writes results directly to disk chunks to avoid large memory usage and transfers.
    Returns the path to the written chunk file.
    """
    import gc
    import os
    import pandas as pd
    from pathlib import Path
    
    # Create detector locally on worker to avoid serialization overhead
    from self_identification import SelfIdentificationDetector
    detector = SelfIdentificationDetector()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique chunk filename
    worker_id = os.getpid()  # Use process ID as worker identifier
    batch_idx = batch_spec["batch_idx"]
    chunk_filename = f"chunk_worker{worker_id}_batch{batch_idx}.csv"
    chunk_path = os.path.join(output_dir, chunk_filename)
    
    # Process data in smaller sub-batches to control memory
    all_results = []
    
    with pq.ParquetFile(batch_spec["file_path"]) as parquet_file:
        # Skip to the correct batch using iter_batches with offset
        batch_size = batch_spec["batch_size"]
        
        # Iterate through batches until we reach our target batch
        for current_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            if current_idx == batch_spec["batch_idx"]:
                df_batch = batch.to_pandas()
                
                # Process in sub-batches to control memory usage
                for start_idx in range(0, len(df_batch), max_entries_in_memory):
                    end_idx = min(start_idx + max_entries_in_memory, len(df_batch))
                    df_sub_batch = df_batch.iloc[start_idx:end_idx]
                    
                    sub_batch_results = process_tusc_batch_for_self_identification(
                        df_sub_batch, detector, split, min_words, max_words
                    )
                    
                    all_results.extend(sub_batch_results)
                    
                    # Write to disk if we have enough results to avoid memory buildup
                    if len(all_results) >= max_entries_in_memory:
                        # Convert to DataFrame and append to file
                        if all_results:
                            results_df = pd.DataFrame(all_results)
                            # Write header only on first write
                            write_header = not os.path.exists(chunk_path)
                            results_df.to_csv(chunk_path, mode='a', index=False, header=write_header)
                            
                        # Clear results from memory
                        all_results = []
                        del results_df
                        gc.collect()
                    
                    # Clean up sub-batch
                    del df_sub_batch, sub_batch_results
                    gc.collect()
                
                # Clean up main batch
                del df_batch
                gc.collect()
                break
            elif current_idx > batch_spec["batch_idx"]:
                break  # We've gone past our target batch
    
    # Write any remaining results
    if all_results:
        results_df = pd.DataFrame(all_results)
        write_header = not os.path.exists(chunk_path)
        results_df.to_csv(chunk_path, mode='a', index=False, header=write_header)
        del results_df, all_results
        gc.collect()
    
    # Return the chunk file path (empty file if no results)
    if not os.path.exists(chunk_path):
        # Create empty file to indicate this worker completed
        Path(chunk_path).touch()
    
    return chunk_path


def process_tusc_batch_for_user_posts_worker(
    batch_spec: Dict[str, Any],
    target_user_ids: Set[str],
    split: str,
    include_features: bool = True,
    output_dir: str = None,
    max_entries_in_memory: int = 10000
) -> str:
    """Process a single parquet batch for user post collection using streaming.
    
    Writes results directly to disk chunks to avoid large memory usage and transfers.
    Returns the path to the written chunk file.
    """
    import gc
    import os
    import pandas as pd
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate unique chunk filename
        worker_id = os.getpid()  # Use process ID as worker identifier
        batch_idx = batch_spec["batch_idx"]
        chunk_filename = f"chunk_worker{worker_id}_batch{batch_idx}.csv"
        chunk_path = os.path.join(output_dir, chunk_filename)
    else:
        chunk_path = None
    
    # Process data in smaller sub-batches to control memory
    all_results = []
    
    with pq.ParquetFile(batch_spec["file_path"]) as parquet_file:
        # Skip to the correct batch using iter_batches with offset
        batch_size = batch_spec["batch_size"]
        
        # Iterate through batches until we reach our target batch
        for current_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            if current_idx == batch_spec["batch_idx"]:
                df_batch = batch.to_pandas()
                
                # Process in sub-batches to control memory usage
                for start_idx in range(0, len(df_batch), max_entries_in_memory):
                    end_idx = min(start_idx + max_entries_in_memory, len(df_batch))
                    df_sub_batch = df_batch.iloc[start_idx:end_idx]
                    
                    sub_batch_result = process_tusc_batch_for_user_posts(
                        df_sub_batch, target_user_ids, split, include_features
                    )
                    
                    if not sub_batch_result.empty:
                        if chunk_path:
                            # Write to disk chunk
                            write_header = not os.path.exists(chunk_path)
                            sub_batch_result.to_csv(chunk_path, mode='a', index=False, header=write_header)
                        else:
                            # Accumulate in memory (for backward compatibility)
                            all_results.append(sub_batch_result)
                    
                    # Clean up sub-batch
                    del df_sub_batch, sub_batch_result
                    gc.collect()
                
                # Clean up main batch
                del df_batch
                gc.collect()
                break
            elif current_idx > batch_spec["batch_idx"]:
                break  # We've gone past our target batch
    
    if chunk_path:
        # Return the chunk file path (create empty file if no results)
        if not os.path.exists(chunk_path):
            Path(chunk_path).touch()
        return chunk_path
    else:
        # Return combined DataFrame for backward compatibility
        if all_results:
            result_df = pd.concat(all_results, ignore_index=True)
            del all_results
            gc.collect()
            return result_df
        else:
            return pd.DataFrame()


def load_tusc_files_for_self_identification(
    input_file: str,
    detector: SelfIdentificationDetector,
    split: str = "country",
    min_words: int = 5,
    max_words: int = 1000,
    chunk_size: int = 100000,
    client=None,
    output_dir: str = None,
    max_entries_in_memory: int = 10000
) -> List[Dict[str, Any]]:
    """Load and process TUSC parquet file for self-identification using Dask with streaming.
    
    If output_dir is provided, workers write chunks directly to disk and results are
    concatenated from disk. Otherwise, uses in-memory processing.
    """
    import os
    import glob
    import tempfile
    
    batch_specs = create_parquet_batch_specs(input_file, chunk_size)
    if not batch_specs:
        raise ValueError(f"No data found in parquet file: {input_file}")
    
    logger.info(f"Processing {len(batch_specs)} batches for self-identification detection.")

    # Create temporary output directory if not provided
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="tusc_chunks_")
        output_dir = temp_dir
        cleanup_temp = True
    else:
        cleanup_temp = False
    
    logger.info(f"Writing worker chunks to: {output_dir}")

    bag = db.from_sequence(batch_specs, npartitions=len(batch_specs))
    processed_bag = bag.map(
        lambda batch_spec: process_tusc_batch_for_self_identification_worker(
            batch_spec, 
            split=split, 
            min_words=min_words, 
            max_words=max_words,
            output_dir=output_dir,
            max_entries_in_memory=max_entries_in_memory
        )
    )

    with ProgressBar():
        chunk_paths = processed_bag.compute()

    # Concatenate results from disk chunks
    logger.info(f"Concatenating {len(chunk_paths)} chunk files from disk...")
    all_results = []
    
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
            try:
                chunk_df = pd.read_csv(chunk_path)
                if not chunk_df.empty:
                    chunk_results = chunk_df.to_dict('records')
                    all_results.extend(chunk_results)
                    del chunk_df, chunk_results
            except Exception as e:
                logger.warning(f"Failed to read chunk {chunk_path}: {e}")
    
    # Clean up temporary files
    if cleanup_temp:
        import shutil
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {output_dir}: {e}")
    
    logger.info(f"Total results collected: {len(all_results)}")
    return all_results


def load_tusc_files_for_user_posts(
    input_file: str,
    target_user_ids: Set[str],
    split: str = "country", 
    include_features: bool = True,
    chunk_size: int = 100000,
    client=None,
    output_dir: str = None,
    max_entries_in_memory: int = 10000
) -> List[Dict[str, Any]]:
    """Load and process TUSC parquet file for user posts using Dask with streaming.
    
    If output_dir is provided, workers write chunks directly to disk and results are
    concatenated from disk. Otherwise, uses in-memory processing.
    """
    import os
    import tempfile
    
    batch_specs = create_parquet_batch_specs(input_file, chunk_size)
    if not batch_specs:
        raise ValueError(f"No data found in parquet file: {input_file}")
        
    logger.info(f"Processing {len(batch_specs)} batches for user post collection.")

    # Create temporary output directory if not provided
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="tusc_user_chunks_")
        output_dir = temp_dir
        cleanup_temp = True
        use_disk_chunks = True
    else:
        cleanup_temp = False
        use_disk_chunks = True

    if use_disk_chunks:
        logger.info(f"Writing worker chunks to: {output_dir}")

    bag = db.from_sequence(batch_specs, npartitions=len(batch_specs))
    processed_bag = bag.map(
        lambda batch_spec: process_tusc_batch_for_user_posts_worker(
            batch_spec, 
            target_user_ids=target_user_ids, 
            split=split, 
            include_features=include_features,
            output_dir=output_dir if use_disk_chunks else None,
            max_entries_in_memory=max_entries_in_memory
        )
    )

    with ProgressBar():
        results = processed_bag.compute()

    if use_disk_chunks:
        # Concatenate results from disk chunks
        logger.info(f"Concatenating {len(results)} chunk files from disk...")
        all_results = []
        
        for chunk_path in results:
            if isinstance(chunk_path, str) and os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                try:
                    chunk_df = pd.read_csv(chunk_path)
                    if not chunk_df.empty:
                        chunk_results = chunk_df.to_dict('records')
                        all_results.extend(chunk_results)
                        del chunk_df, chunk_results
                except Exception as e:
                    logger.warning(f"Failed to read chunk {chunk_path}: {e}")
        
        # Clean up temporary files
        if cleanup_temp:
            import shutil
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {output_dir}: {e}")
        
        logger.info(f"Total results collected: {len(all_results)}")
        return all_results
    else:
        # Combine all DataFrames (backward compatibility)
        non_empty_dfs = [df for df in results if isinstance(df, pd.DataFrame) and not df.empty]
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
    """Load TUSC parquet file for processing using streaming batch approach.
    
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
        chunk_size: Batch size for streaming processing
        client: Dask client for parallel processing
        
    Returns:
        Processed DataFrame
    """
    if split is None:
        split = determine_tusc_split(input_file)
    
    logger.info(f"Loading TUSC file: {input_file} (split: {split}, mode: {mode})")
    
    # Use streaming batch processing for both test and full modes
    if test_mode:
        # For test mode, use smaller batch size and early exit
        test_batch_size = min(chunk_size, test_samples)
        logger.info(f"Test mode: Processing up to {test_samples} samples with batch size {test_batch_size}")
        
        if mode == "self_identification":
            if detector is None:
                raise ValueError("Detector required for self_identification mode")
            
            # Process with early exit for test mode
            results = []
            processed_rows = 0
            
            with pq.ParquetFile(input_file) as parquet_file:
                for batch in parquet_file.iter_batches(batch_size=test_batch_size):
                    df_batch = batch.to_pandas()
                    
                    batch_results = process_tusc_batch_for_self_identification(
                        df_batch, detector, split, min_words, max_words
                    )
                    results.extend(batch_results)
                    
                    processed_rows += len(df_batch)
                    if processed_rows >= test_samples:
                        break
                
            logger.info(f"Found {len(results)} self-identification entries")
            return pd.DataFrame(results)
            
        elif mode == "user_posts":
            if target_user_ids is None:
                raise ValueError("Target user IDs required for user_posts mode")
            
            # Process with early exit for test mode
            result_dfs = []
            processed_rows = 0
            
            with pq.ParquetFile(input_file) as parquet_file:
                for batch in parquet_file.iter_batches(batch_size=test_batch_size):
                    df_batch = batch.to_pandas()
                    
                    batch_result = process_tusc_batch_for_user_posts(
                        df_batch, target_user_ids, split, include_features
                    )
                    if not batch_result.empty:
                        result_dfs.append(batch_result)
                    
                    processed_rows += len(df_batch)
                    if processed_rows >= test_samples:
                        break
            
            # Combine results
            if result_dfs:
                result_df = pd.concat(result_dfs, ignore_index=True)
            else:
                result_df = pd.DataFrame()
                
            logger.info(f"Filtered to {len(result_df)} posts from target users")
            return result_df
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    else:
        # Use streaming batch processing for full datasets
        if mode == "self_identification":
            if detector is None:
                raise ValueError("Detector required for self_identification mode")
                
            # Use parallel processing if client available, otherwise streaming
            if client is not None:
                results = load_tusc_files_for_self_identification(
                    input_file=input_file,
                    detector=detector,
                    split=split,
                    min_words=min_words,
                    max_words=max_words,
                    chunk_size=chunk_size,
                    client=client
                )
            else:
                # Single-machine streaming processing
                results = process_parquet_in_batches(
                    input_file=input_file,
                    batch_size=chunk_size,
                    process_batch_fn=process_tusc_batch_for_self_identification,
                    detector=detector,
                    split=split,
                    min_words=min_words,
                    max_words=max_words
                )
                
            logger.info(f"Found {len(results)} self-identification entries")
            return pd.DataFrame(results)
            
        elif mode == "user_posts":
            if target_user_ids is None:
                raise ValueError("Target user IDs required for user_posts mode")
                
            # Use parallel processing if client available, otherwise streaming
            if client is not None:
                results = load_tusc_files_for_user_posts(
                    input_file=input_file,
                    target_user_ids=target_user_ids,
                    split=split,
                    include_features=include_features,
                    chunk_size=chunk_size,
                    client=client
                )
                result_df = pd.DataFrame(results)
            else:
                # Single-machine streaming processing
                result_dfs = process_parquet_in_batches(
                    input_file=input_file,
                    batch_size=chunk_size,
                    process_batch_fn=process_tusc_batch_for_user_posts,
                    target_user_ids=target_user_ids,
                    split=split,
                    include_features=include_features
                )
                
                # Combine DataFrames
                if result_dfs and any(not df.empty for df in result_dfs):
                    result_df = pd.concat([df for df in result_dfs if not df.empty], ignore_index=True)
                else:
                    result_df = pd.DataFrame()
                
            logger.info(f"Filtered to {len(result_df)} posts from target users")
            return result_df
            
        else:
            raise ValueError(f"Unknown mode: {mode}")