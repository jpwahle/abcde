import argparse
import logging
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set
import csv

# Add Dask imports
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from process_tusc_data import process_tusc_batch

logger = logging.getLogger("collect_user_posts_tusc")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_tusc_user_ids(self_id_csv: str) -> Set[str]:
    """Load the set of user IDs that have been self-identified.
    
    Args:
        self_id_csv: Path to the CSV file from identify_self_users.py (TUSC mode)
        
    Returns:
        Set of user IDs (strings) that should be collected
    """
    delimiter = "\t" if self_id_csv.lower().endswith(".tsv") else ","
    user_ids = set()
    
    with open(self_id_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            # Get both possible user ID fields
            author_id = record.get("Author", "").strip()
            author_name = record.get("AuthorName", "").strip()
            
            if author_id:
                user_ids.add(author_id)
            if author_name:
                user_ids.add(author_name)
    
    logger.info(f"Loaded {len(user_ids)} unique user IDs from {self_id_csv}")
    return user_ids


def filter_tusc_batch_by_users(
    df_batch: pd.DataFrame,
    target_user_ids: Set[str],
    split: str
) -> pd.DataFrame:
    """Filter TUSC batch to only include posts from target users."""
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
    
    filtered_df = df_batch[mask]
    logger.debug(f"Filtered batch from {len(df_batch)} to {len(filtered_df)} rows")
    
    return filtered_df


def process_tusc_user_posts(
    input_file: str,
    self_id_csv: str,
    output_csv: str,
    split: str = None,  # Will be auto-determined from filename
    chunk_size: int = 100000,
    test_mode: bool = False,
    test_samples: int = 10000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
    output_tsv: bool = False,
):
    """Process TUSC parquet file to collect posts from self-identified users and compute linguistic features."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Auto-determine split type from filename if not provided
    if split is None:
        filename = Path(input_file).name.lower()
        if 'city' in filename:
            split = "city"
        elif 'country' in filename:
            split = "country"
        else:
            # Default to country if unclear
            split = "country"
            logger.warning(f"Could not determine split type from filename '{filename}', defaulting to 'country'")
    
    logger.info(f"Using split type: {split}")

    # Load target user IDs
    target_user_ids = load_tusc_user_ids(self_id_csv)
    if not target_user_ids:
        logger.warning("No target user IDs found. Exiting.")
        return

    # Configure Dask cluster
    if use_slurm:
        # Ensure log directory exists for worker logs
        os.makedirs("logs", exist_ok=True)
        logger.info(
            f"Using SLURM cluster with {n_workers} workers, memory {memory_per_worker} – dask logs in logs/"
        )
        cluster = SLURMCluster(
            cores=1,
            processes=1,
            memory=memory_per_worker,
            walltime="1-00:00:00",
            job_extra=[
                "--cpus-per-task=1",
                "-o",
                "logs/dask-%j.out",
                "-e",
                "logs/dask-%j.err",
            ],
        )
        cluster.scale(n_workers)
        client = Client(cluster)
    else:
        logger.info(f"Starting local Dask cluster with {n_workers} workers")
        client = Client(n_workers=n_workers, threads_per_worker=1, memory_limit=memory_per_worker)

    logger.info(f"Dashboard: {client.dashboard_link}")

    try:
        # Read parquet file with Dask DataFrame
        logger.info(f"Reading {input_file} with Dask DataFrame")
        
        if test_mode:
            # In test mode, read only a subset
            df = pd.read_parquet(input_file, engine='pyarrow')
            df = df.head(test_samples)
            ddf = dd.from_pandas(df, npartitions=min(n_workers, len(df) // chunk_size + 1))
            logger.info(f"Test mode: Processing {len(df)} samples")
        else:
            # Read with Dask and partition appropriately
            ddf = dd.read_parquet(input_file, engine='pyarrow')
            # Repartition to have reasonable chunk sizes
            ddf = ddf.repartition(partition_size=f"{chunk_size} rows")
            logger.info(f"Processing {len(ddf)} rows with {ddf.npartitions} partitions")

        # Apply filtering and processing to each partition
        def process_partition(df_partition):
            """Process a single partition: filter by users and compute features."""
            if df_partition.empty:
                return df_partition
                
            # First filter to only include target users
            filtered_df = filter_tusc_batch_by_users(df_partition, target_user_ids, split)
            
            if filtered_df.empty:
                return filtered_df
            
            # Then compute linguistic features using existing function
            from compute_features import load_body_parts
            body_parts = load_body_parts('data/bodywords-full.txt')
            processed_df = process_tusc_batch(filtered_df, body_parts, split)
            
            return processed_df

        # Apply processing function to each partition
        logger.info("Starting parallel processing...")
        with ProgressBar():
            # Create meta dataframe to describe expected output schema
            # Use the same schema as process_tusc_data.py
            meta_dict = {}
            
            # Get a sample to understand input schema
            sample_df = ddf.head(1)
            for col in sample_df.columns:
                meta_dict[col] = sample_df[col].dtype
            
            # Add the columns that will be created (same as in process_tusc_data.py)
            # DateTime columns
            meta_dict['Day'] = 'Int64'
            meta_dict['Hour'] = 'Int64'  
            meta_dict['Weekday'] = 'object'
            
            # TokenizedTweet
            meta_dict['TokenizedTweet'] = 'object'
            
            # All the linguistic features (copy from process_tusc_data.py)
            # VAD averages
            meta_dict['NRCAvgValence'] = 'float64'
            meta_dict['NRCAvgArousal'] = 'float64'
            meta_dict['NRCAvgDominance'] = 'float64'
            
            # VAD threshold flags
            meta_dict['NRCHasHighValenceWord'] = 'int64'
            meta_dict['NRCHasHighArousalWord'] = 'int64'
            meta_dict['NRCHasHighDominanceWord'] = 'int64'
            meta_dict['NRCHasLowValenceWord'] = 'int64'
            meta_dict['NRCHasLowArousalWord'] = 'int64'
            meta_dict['NRCHasLowDominanceWord'] = 'int64'
            
            # Emotion flags
            emotion_names = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Negative', 'Positive', 'Sadness', 'Surprise', 'Trust']
            for emotion in emotion_names:
                meta_dict[f'NRCHas{emotion}Word'] = 'int64'
            
            # Emotion counts  
            for emotion in emotion_names:
                meta_dict[f'NRCCount{emotion}Words'] = 'int64'
                
            # VAD counts
            meta_dict['NRCCountHighValenceWords'] = 'int64'
            meta_dict['NRCCountHighArousalWords'] = 'int64'
            meta_dict['NRCCountHighDominanceWords'] = 'int64'
            meta_dict['NRCCountLowValenceWords'] = 'int64'
            meta_dict['NRCCountLowArousalWords'] = 'int64'
            meta_dict['NRCCountLowDominanceWords'] = 'int64'
            
            # WordCount
            meta_dict['WordCount'] = 'int64'
            
            # Worry features
            meta_dict['NRCHasAnxietyWord'] = 'int64'
            meta_dict['NRCHasCalmnessWord'] = 'int64'
            meta_dict['NRCAvgAnxiety'] = 'float64'
            meta_dict['NRCAvgCalmness'] = 'float64'
            meta_dict['NRCHasHighAnxietyWord'] = 'int64'
            meta_dict['NRCCountHighAnxietyWords'] = 'int64'
            meta_dict['NRCHasHighCalmnessWord'] = 'int64'
            meta_dict['NRCCountHighCalmnessWords'] = 'int64'
            
            # MoralTrust features
            meta_dict['NRCHasHighMoralTrustWord'] = 'int64'
            meta_dict['NRCCountHighMoralTrustWord'] = 'int64'
            meta_dict['NRCHasLowMoralTrustWord'] = 'int64'
            meta_dict['NRCCountLowMoralTrustWord'] = 'int64'
            meta_dict['NRCAvgMoralTrustWord'] = 'float64'
            
            # SocialWarmth features
            meta_dict['NRCHasHighSocialWarmthWord'] = 'int64'
            meta_dict['NRCCountHighSocialWarmthWord'] = 'int64'
            meta_dict['NRCHasLowSocialWarmthWord'] = 'int64'
            meta_dict['NRCCountLowSocialWarmthWord'] = 'int64'
            meta_dict['NRCAvgSocialWarmthWord'] = 'float64'
            
            # Warmth features
            meta_dict['NRCHasHighWarmthWord'] = 'int64'
            meta_dict['NRCCountHighWarmthWord'] = 'int64'
            meta_dict['NRCHasLowWarmthWord'] = 'int64'
            meta_dict['NRCCountLowWarmthWord'] = 'int64'
            meta_dict['NRCAvgWarmthWord'] = 'float64'
            
            # Body part mentions
            meta_dict['MyBPM'] = 'object'
            meta_dict['YourBPM'] = 'object'
            meta_dict['HerBPM'] = 'object'
            meta_dict['HisBPM'] = 'object'
            meta_dict['TheirBPM'] = 'object'
            meta_dict['HasBPM'] = 'int64'
            
            # Individual pronouns
            pronoun_cols = ['PRNHasI', 'PRNHasMe', 'PRNHasMy', 'PRNHasMine', 'PRNHasWe', 'PRNHasOur', 'PRNHasOurs',
                           'PRNHasYou', 'PRNHasYour', 'PRNHasYours', 'PRNHasShe', 'PRNHasHer', 'PRNHasHers', 
                           'PRNHasHe', 'PRNHasHim', 'PRNHasHis', 'PRNHasThey', 'PRNHasThem', 'PRNHasTheir', 'PRNHasTheirs']
            for col in pronoun_cols:
                meta_dict[col] = 'int64'
            
            # TIME tense features
            meta_dict['TIMEHasPastVerb'] = 'int64'
            meta_dict['TIMECountPastVerbs'] = 'int64'
            meta_dict['TIMEHasPresentVerb'] = 'int64'
            meta_dict['TIMECountPresentVerbs'] = 'int64'
            meta_dict['TIMEHasFutureModal'] = 'int64'
            meta_dict['TIMECountFutureModals'] = 'int64'
            meta_dict['TIMEHasPresentNoFuture'] = 'int64'
            meta_dict['TIMEHasFutureReference'] = 'int64'
            
            # Create meta dataframe
            meta_df = pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in meta_dict.items()})
            
            processed_ddf = ddf.map_partitions(process_partition, meta=meta_df)
            
            # Compute the results and convert to pandas for output
            result_df = processed_ddf.compute()

        logger.info(f"Processed {len(result_df)} rows from target users")

        # Write output
        separator = '\t' if output_tsv else ','
        file_extension = 'tsv' if output_tsv else 'csv'
        output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
        
        result_df.to_csv(output_file, index=False, sep=separator)
        
        logger.info(f"Output written to {output_file}")

    finally:
        # Clean up cluster
        client.close()
        if use_slurm:
            cluster.close()


def main():
    parser = argparse.ArgumentParser(description="Collect posts from self-identified TUSC users and compute linguistic features")
    parser.add_argument("--input_file", required=True, help="Path to input parquet file")
    parser.add_argument("--self_identified_csv", required=True, help="Output CSV from identify_self_users.py (TUSC mode)")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for user posts with linguistic features")
    parser.add_argument("--split", choices=["city", "country"], default=None, help="Data split type (auto-determined from filename if not specified)")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for Dask DataFrame partitions")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with limited samples")
    parser.add_argument("--test_samples", type=int, default=10000, help="Number of samples in test mode")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")
    
    args = parser.parse_args()
    
    process_tusc_user_posts(
        input_file=args.input_file,
        self_id_csv=args.self_identified_csv,
        output_csv=args.output_csv,
        split=args.split,
        chunk_size=args.chunk_size,
        test_mode=args.test_mode,
        test_samples=args.test_samples,
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm,
        output_tsv=args.output_tsv,
    )


if __name__ == "__main__":
    main()