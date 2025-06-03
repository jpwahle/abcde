import argparse
import logging
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from pathlib import Path
from flashtext import KeywordProcessor
from tqdm.auto import tqdm
import spacy
from typing import Dict, Any, List

# Add Dask imports
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from compute_features import (
    compute_all_features,
    load_body_parts,
    vad_dict,
    emotion_dict,
    emotions,
    worry_dict,
    tense_dict,
    moraltrust_dict,
    socialwarmth_dict,
    warmth_dict,
)

logger = logging.getLogger("process_tusc_data")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Load the SpaCy model for tokenization
nlp = spacy.blank("en")  # Blank model for faster performance


def extract_keywords_from_tweet(tweet, keyword_processor=None):
    """Extract keywords from tweet using KeywordProcessor."""
    if pd.isna(tweet):
        return []
    return keyword_processor.extract_keywords(tweet)


def process_tusc_batch(
    df_batch: pd.DataFrame,
    body_parts: List[str],
    split: str = "country"
) -> pd.DataFrame:
    """Process a batch of TUSC data with all linguistic features."""
    
    # Add datetime-derived columns based on split type
    if split == "city":
        def date_fn(x):
            if pd.isna(x):
                return None
            try:
                return datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y')
            except (ValueError, TypeError):
                return None
        # City files already have Year and Month columns, but let's ensure they're derived from createdAt too
        df_batch['Year'] = df_batch['createdAt'].apply(lambda x: date_fn(x).year if date_fn(x) is not None else None)
        df_batch['Month'] = df_batch['createdAt'].apply(lambda x: date_fn(x).month if date_fn(x) is not None else None)
    else:  # country
        # Handle ISO format with 'Z' timezone suffix
        def parse_iso_date(x):
            if pd.isna(x):
                return None
            try:
                # Replace 'Z' with '+00:00' for proper timezone parsing
                x_clean = x.replace('Z', '+00:00') if x.endswith('Z') else x
                return datetime.fromisoformat(x_clean)
            except (ValueError, TypeError, AttributeError):
                return None
        
        date_fn = parse_iso_date
        # Country files already have Year and Month columns, but let's ensure they're derived from createdAt too
        df_batch['Year'] = df_batch['createdAt'].apply(lambda x: date_fn(x).year if date_fn(x) is not None else None)
        df_batch['Month'] = df_batch['createdAt'].apply(lambda x: date_fn(x).month if date_fn(x) is not None else None)

    # Add additional datetime columns
    df_batch['Day'] = df_batch['createdAt'].apply(lambda x: date_fn(x).day if date_fn(x) is not None else None)
    df_batch['Hour'] = df_batch['createdAt'].apply(lambda x: date_fn(x).hour if date_fn(x) is not None else None)
    df_batch['Weekday'] = df_batch['createdAt'].apply(lambda x: date_fn(x).strftime('%A') if date_fn(x) is not None else None)

    # Get the correct tweet column name (both city and country use 'Tweet')
    tweet_column = 'Tweet'
    
    # Add tokenized tweet using SpaCy
    df_batch['TokenizedTweet'] = df_batch[tweet_column].apply(
        lambda tweet: [token.text for token in nlp(tweet)] if pd.notna(tweet) else []
    )

    # Compute all linguistic features (this includes body parts, pronouns, NRC, TIME, etc.)
    all_features = df_batch[tweet_column].apply(
        lambda tw: compute_all_features(
            tw,
            vad_dict,
            emotion_dict,
            emotions,
            worry_dict,
            tense_dict,
            moraltrust_dict,
            socialwarmth_dict,
            warmth_dict,
            body_parts
        )
    )
    all_features_df = pd.DataFrame(list(all_features))
    df_batch = pd.concat([df_batch, all_features_df], axis=1)

    return df_batch


def track_pronoun_body_part_occurrence(
    input_file: str,
    output_csv: str,
    chunk_size: int = 100000,
    split: str = None,  # Will be auto-determined from filename
    test_mode: bool = False,
    test_samples: int = 10000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False
):
    """
    Process TUSC parquet file and compute linguistic features using Dask for parallelization.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        output_csv: Path to output TSV file
        chunk_size: Chunk size for Dask DataFrame partitions
        split: Type of data ("city" or "country") - auto-determined from filename if None
        test_mode: If True, only process first test_samples rows
        test_samples: Number of samples to process in test mode
        n_workers: Number of Dask workers
        memory_per_worker: Memory per worker for SLURMCluster
        use_slurm: Whether to use SLURM cluster
    """
    
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
    
    # Load body parts
    body_parts = load_body_parts('data/bodywords-full.txt')
    
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
        
        # Read the parquet file and set up partitions
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

        # Apply processing function to each partition
        def process_partition(df_partition):
            """Process a single partition of the DataFrame."""
            if df_partition.empty:
                return df_partition
            return process_tusc_batch(df_partition, body_parts, split)

        # Process all partitions in parallel
        logger.info("Starting parallel processing...")
        with ProgressBar():
            # Create meta dataframe to describe expected output schema
            # Start with the input schema and add all the new columns
            meta_dict = {}
            
            # Get a sample to understand input schema
            sample_df = ddf.head(1)
            for col in sample_df.columns:
                meta_dict[col] = sample_df[col].dtype
            
            # Add the columns that will be created in the exact order they appear
            # 1. DateTime columns added manually
            meta_dict['Day'] = 'Int64'  # Nullable integer
            meta_dict['Hour'] = 'Int64'  # Nullable integer  
            meta_dict['Weekday'] = 'object'  # String type
            
            # 2. TokenizedTweet added manually
            meta_dict['TokenizedTweet'] = 'object'  # List of strings
            
            # 3. Features from compute_all_features (in the order they're returned)
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
            
            # Worry (anxiety/calmness) features
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
            
            # Body part mentions (from compute_prefixed_body_part_mentions)
            meta_dict['MyBPM'] = 'object'  # String
            meta_dict['YourBPM'] = 'object'  # String
            meta_dict['HerBPM'] = 'object'  # String
            meta_dict['HisBPM'] = 'object'  # String
            meta_dict['TheirBPM'] = 'object'  # String
            meta_dict['HasBPM'] = 'int64'
            
            # Individual pronouns (from compute_individual_pronouns)
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
            
            # Column order for output (will be filtered based on split type)
            column_order = [
                # Identification and Temporal Data - Country
                "MyCountry", "TweetID", "UserID", "createdAt", "Country", "PlaceID", "PlaceType", "Place",
                # Identification and Temporal Data - City  
                "City", "userID", "userName",
                # Common temporal data
                "Year", "Month", "Day", "Weekday", "Hour",

                # Pronoun-Body Part Mentions
                "MyBPM", "YourBPM", "HerBPM", "HisBPM", "TheirBPM", "HasBPM",

                # Individual Pronoun Presence (PRN prefix)
                "PRNHasI", "PRNHasMe", "PRNHasMy", "PRNHasMine",
                "PRNHasWe", "PRNHasOur", "PRNHasOurs",
                "PRNHasYou", "PRNHasYour", "PRNHasYours",
                "PRNHasShe", "PRNHasHer", "PRNHasHers",
                "PRNHasHe", "PRNHasHim", "PRNHasHis",
                "PRNHasThey", "PRNHasThem", "PRNHasTheir", "PRNHasTheirs",

                # Tokenization and Word Count
                "TokenizedTweet", "WordCount",

                # VAD Averages
                "NRCAvgValence", "NRCAvgArousal", "NRCAvgDominance",

                # VAD Presence Flags
                "NRCHasHighValenceWord", "NRCHasLowValenceWord",
                "NRCHasHighArousalWord", "NRCHasLowArousalWord",
                "NRCHasHighDominanceWord", "NRCHasLowDominanceWord",

                # VAD Counts
                "NRCCountHighValenceWords", "NRCCountLowValenceWords",
                "NRCCountHighArousalWords", "NRCCountLowArousalWords",
                "NRCCountHighDominanceWords", "NRCCountLowDominanceWords",

                # Emotion Presence Flags
                "NRCHasAngerWord", "NRCHasAnticipationWord", "NRCHasDisgustWord",
                "NRCHasFearWord", "NRCHasJoyWord", "NRCHasNegativeWord",
                "NRCHasPositiveWord", "NRCHasSadnessWord", "NRCHasSurpriseWord", "NRCHasTrustWord",

                # Emotion Counts
                "NRCCountAngerWords", "NRCCountAnticipationWords", "NRCCountDisgustWords",
                "NRCCountFearWords", "NRCCountJoyWords", "NRCCountNegativeWords",
                "NRCCountPositiveWords", "NRCCountSadnessWords", "NRCCountSurpriseWords", "NRCCountTrustWords",

                # Anxiety/Calmness Presence & Averages
                "NRCHasAnxietyWord", "NRCHasCalmnessWord", "NRCAvgAnxiety", "NRCAvgCalmness",
                "NRCHasHighAnxietyWord", "NRCCountHighAnxietyWords",
                "NRCHasHighCalmnessWord", "NRCCountHighCalmnessWords",

                # Moral Trust Features
                "NRCHasHighMoralTrustWord", "NRCCountHighMoralTrustWord",
                "NRCHasLowMoralTrustWord", "NRCCountLowMoralTrustWord",
                "NRCAvgMoralTrustWord",

                # Social Warmth Features
                "NRCHasHighSocialWarmthWord", "NRCCountHighSocialWarmthWord",
                "NRCHasLowSocialWarmthWord", "NRCCountLowSocialWarmthWord",
                "NRCAvgSocialWarmthWord",

                # Warmth Features
                "NRCHasHighWarmthWord", "NRCCountHighWarmthWord",
                "NRCHasLowWarmthWord", "NRCCountLowWarmthWord",
                "NRCAvgWarmthWord",

                # TIME (tense) Features
                "TIMEHasPastVerb", "TIMECountPastVerbs",
                "TIMEHasPresentVerb", "TIMECountPresentVerbs",
                "TIMEHasFutureModal", "TIMECountFutureModals",
                "TIMEHasPresentNoFuture", "TIMEHasFutureReference"
            ]
            
            # Reorder columns function
            def reorder_columns(df_partition):
                """Reorder columns in the partition."""
                if df_partition.empty:
                    return df_partition
                # Reorder columns, placing any extras at the end
                existing_cols = [col for col in column_order if col in df_partition.columns]
                remaining_cols = [col for col in df_partition.columns if col not in existing_cols]
                return df_partition[existing_cols + remaining_cols]
            
            # Apply column reordering
            processed_ddf = processed_ddf.map_partitions(reorder_columns)
            
            # Compute the results and convert to pandas for output
            result_df = processed_ddf.compute()

        logger.info(f"Processed {len(result_df)} rows total")

        # Write TSV
        result_df.to_csv(output_csv, index=False, sep='\t')
        
        logger.info(f"Output written successfully")

    finally:
        # Clean up cluster
        client.close()
        if use_slurm:
            cluster.close()


def main():
    parser = argparse.ArgumentParser(description="Process TUSC data with linguistic features using Dask parallelization")
    parser.add_argument("--input_file", required=True, help="Path to input parquet file")
    parser.add_argument("--output_dir", required=True, help="Directory for output files")
    parser.add_argument("--split", choices=["city", "country"], default=None, help="Data split type (auto-determined from filename if not specified)")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for Dask DataFrame partitions")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with limited samples")
    parser.add_argument("--test_samples", type=int, default=10000, help="Number of samples in test mode")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames based on input filename
    input_name = Path(args.input_file).stem
    output_tsv = os.path.join(args.output_dir, f"{input_name}_processed.tsv")
    
    if args.test_mode:
        output_tsv = os.path.join(args.output_dir, f"{input_name}_test.tsv")
    
    track_pronoun_body_part_occurrence(
        input_file=args.input_file,
        output_csv=output_tsv,
        chunk_size=args.chunk_size,
        split=args.split,
        test_mode=args.test_mode,
        test_samples=args.test_samples,
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm
    )


if __name__ == "__main__":
    main()