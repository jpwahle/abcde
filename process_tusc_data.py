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
        date_fn = lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y')
        # City files already have Year and Month columns, but let's ensure they're derived from createdAt too
        df_batch['Year'] = df_batch['createdAt'].apply(lambda x: date_fn(x).year if pd.notna(x) else None)
        df_batch['Month'] = df_batch['createdAt'].apply(lambda x: date_fn(x).month if pd.notna(x) else None)
    else:  # country
        date_fn = lambda x: datetime.fromisoformat(x)
        # Country files already have Year and Month columns, but let's ensure they're derived from createdAt too
        df_batch['Year'] = df_batch['createdAt'].apply(lambda x: date_fn(x).year if pd.notna(x) else None)
        df_batch['Month'] = df_batch['createdAt'].apply(lambda x: date_fn(x).month if pd.notna(x) else None)

    # Add additional datetime columns
    df_batch['Day'] = df_batch['createdAt'].apply(lambda x: date_fn(x).day if pd.notna(x) else None)
    df_batch['Hour'] = df_batch['createdAt'].apply(lambda x: date_fn(x).hour if pd.notna(x) else None)
    df_batch['Weekday'] = df_batch['createdAt'].apply(lambda x: date_fn(x).strftime('%A') if pd.notna(x) else None)

    # Define prefixes and their labels for body parts
    prefixes_and_labels = [
        ("my ", "My BPM"),
        ("your ", "Your BPM"),
        ("her ", "Her BPM"),
        ("his ", "His BPM"),
        ("their ", "Their BPM"),
    ]

    # Create prefixed body parts dictionaries
    prefixed_body_parts = {}
    for prefix, label in prefixes_and_labels:
        prefixed_body_parts[label] = [f"{prefix}{body_part}" for body_part in body_parts]

    # Get the correct tweet column name (both city and country use 'Tweet')
    tweet_column = 'Tweet'
    
    # Process prefixed body part mentions
    for label, body_parts_list in prefixed_body_parts.items():
        keyword_processor = KeywordProcessor(case_sensitive=False)
        keyword_processor.add_keywords_from_list(body_parts_list)
        df_batch[label] = df_batch[tweet_column].apply(
            lambda t: ", ".join(extract_keywords_from_tweet(t, keyword_processor)) if pd.notna(t) else ""
        )

    # Process non-prefixed body parts for HasBPM
    keyword_processor_has_bpm = KeywordProcessor(case_sensitive=False)
    keyword_processor_has_bpm.add_keywords_from_list(body_parts)
    df_batch['HasBPM'] = df_batch[tweet_column].apply(
        lambda t: int(bool(extract_keywords_from_tweet(t, keyword_processor_has_bpm))) if pd.notna(t) else 0
    )

    # Define individual pronoun sets with PRNHas* column names
    individual_pronoun_sets = {
        "PRNHasI": ["i"],
        "PRNHasMe": ["me"],
        "PRNHasMy": ["my"],
        "PRNHasMine": ["mine"],
        "PRNHasWe": ["we"],
        "PRNHasOur": ["our"],
        "PRNHasOurs": ["ours"],
        "PRNHasYou": ["you"],
        "PRNHasYour": ["your"],
        "PRNHasYours": ["yours"],
        "PRNHasShe": ["she"],
        "PRNHasHer": ["her"],
        "PRNHasHers": ["hers"],
        "PRNHasHe": ["he"],
        "PRNHasHim": ["him"],
        "PRNHasHis": ["his"],
        "PRNHasThey": ["they"],
        "PRNHasThem": ["them"],
        "PRNHasTheir": ["their"],
        "PRNHasTheirs": ["theirs"],
    }

    # Process individual pronouns
    for col_name, pronoun_list in individual_pronoun_sets.items():
        keyword_processor = KeywordProcessor(case_sensitive=False)
        keyword_processor.add_keywords_from_list(pronoun_list)
        df_batch[col_name] = df_batch[tweet_column].apply(
            lambda t: int(bool(extract_keywords_from_tweet(t, keyword_processor))) if pd.notna(t) else 0
        )

    # Ensure schema consistency for body part mention columns
    for label in ["My BPM", "Your BPM", "Her BPM", "His BPM", "Their BPM"]:
        if label not in df_batch.columns:
            df_batch[label] = ""
        df_batch[label] = df_batch[label].astype(str)

    # Add tokenized tweet using SpaCy
    df_batch['TokenizedTweet'] = df_batch[tweet_column].apply(
        lambda tweet: [token.text for token in nlp(tweet)] if pd.notna(tweet) else []
    )

    # Compute all linguistic features
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

    # Drop unnecessary columns if they exist
    for col in ['File', 'RowNum']:
        if col in df_batch.columns:
            df_batch.drop(columns=[col], inplace=True)

    return df_batch


def track_pronoun_body_part_occurrence(
    input_file: str,
    output_file: str,
    output_csv: str,
    batch_size: int = 10000,
    split: str = "country",
    test_mode: bool = False,
    test_samples: int = 10000
):
    """
    Process TUSC parquet file and compute linguistic features.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        output_csv: Path to output TSV file
        batch_size: Batch size for processing
        split: Type of data ("city" or "country")
        test_mode: If True, only process first test_samples rows
        test_samples: Number of samples to process in test mode
    """
    
    # Load body parts
    body_parts = load_body_parts('data/bodywords-full.txt')

    # Column order for output (will be filtered based on split type)
    column_order = [
        # Identification and Temporal Data - Country
        "File", "RowNum", "MyCountry", "TweetID", "UserID", "createdAt", "Country", "PlaceID", "PlaceType", "Place",
        # Identification and Temporal Data - City  
        "City", "userID", "userName",
        # Common temporal data
        "Year", "Month", "Day", "Weekday", "Hour",

        # Pronoun-Body Part Mentions
        "My BPM", "Your BPM", "Her BPM", "His BPM", "Their BPM", "HasBPM",

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

    parquet_file = pq.ParquetFile(input_file)
    writer = None
    csv_initialized = False
    
    total_rows = parquet_file.metadata.num_rows
    if test_mode:
        total_rows = min(total_rows, test_samples)
        logger.info(f"Test mode: Processing only {total_rows} samples")
    
    total_batches = total_rows // batch_size + 1
    processed_rows = 0

    logger.info(f"Processing {input_file} with {total_rows} rows in {total_batches} batches")

    for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), total=total_batches):
        df_batch = batch.to_pandas()
        
        # In test mode, limit the number of rows processed
        if test_mode and processed_rows + len(df_batch) > test_samples:
            df_batch = df_batch.head(test_samples - processed_rows)
        
        if df_batch.empty:
            break

        # Process the batch
        df_batch = process_tusc_batch(df_batch, body_parts, split)

        # Reorder columns, placing any extras at the end
        existing_cols = [col for col in column_order if col in df_batch.columns]
        remaining_cols = [col for col in df_batch.columns if col not in existing_cols]
        df_batch = df_batch[existing_cols + remaining_cols]

        # Convert to PyArrow table
        table = pa.Table.from_pandas(df_batch)

        # Write Parquet
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)

        # Write TSV (tab-separated)
        if not csv_initialized:
            df_batch.to_csv(output_csv, mode='w', index=False, header=True, sep='\t')
            csv_initialized = True
        else:
            df_batch.to_csv(output_csv, mode='a', index=False, header=False, sep='\t')

        processed_rows += len(df_batch)
        
        # In test mode, break if we've processed enough samples
        if test_mode and processed_rows >= test_samples:
            break

    if writer:
        writer.close()

    logger.info(f"Processed {processed_rows} rows total")
    logger.info(f"Output written to: {output_file} (parquet) and {output_csv} (TSV)")


def main():
    parser = argparse.ArgumentParser(description="Process TUSC data with linguistic features")
    parser.add_argument("--input_file", required=True, help="Path to input parquet file")
    parser.add_argument("--output_dir", required=True, help="Directory for output files")
    parser.add_argument("--split", choices=["city", "country"], default="country", help="Data split type")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with limited samples")
    parser.add_argument("--test_samples", type=int, default=10000, help="Number of samples in test mode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames based on input filename and split
    input_name = Path(args.input_file).stem
    output_parquet = os.path.join(args.output_dir, f"{input_name}_processed.parquet")
    output_tsv = os.path.join(args.output_dir, f"{input_name}_processed.tsv")
    
    if args.test_mode:
        output_parquet = os.path.join(args.output_dir, f"{input_name}_test.parquet")
        output_tsv = os.path.join(args.output_dir, f"{input_name}_test.tsv")
    
    track_pronoun_body_part_occurrence(
        input_file=args.input_file,
        output_file=output_parquet,
        output_csv=output_tsv,
        batch_size=args.batch_size,
        split=args.split,
        test_mode=args.test_mode,
        test_samples=args.test_samples
    )


if __name__ == "__main__":
    main()