"""
TUSC-specific user post collection pipeline.
Collects all posts by self-identified users and computes linguistic features.
Supports both single-machine and distributed parallel processing.
"""
import argparse
import logging
import csv
import pandas as pd

from core.cluster import setup_dask_cluster, cleanup_cluster
from core.io_utils import ensure_output_directory
from tusc.data_loader import load_tusc_file, determine_tusc_split
from tusc.user_loader import load_tusc_user_ids

logger = logging.getLogger("collect_user_posts_tusc")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_tusc_user_details(self_id_csv: str):
    """Load user details including birthyears and raw ages from self-identification CSV."""
    delimiter = "\t" if self_id_csv.lower().endswith(".tsv") else ","
    user_details = {}
    
    with open(self_id_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            # Get both possible user ID fields
            author_id = record.get("Author", "").strip()
            author_name = record.get("AuthorName", "").strip()
            
            # Extract birthyear and raw ages
            birth_year = None
            age_str = record.get("SelfIdentificationAgeMajorityVote", "").strip()
            raw_ages = record.get("SelfIdentificationRawAges", "").strip()
            
            if age_str:
                try:
                    age_val = int(float(age_str))
                    # Calculate birth year from age and post timestamp
                    created_at_str = record.get("PostCreatedAt", "").strip()
                    post_year_str = record.get("PostYear", "").strip()
                    
                    # Try to get year from PostYear field first, then from timestamp
                    if post_year_str:
                        try:
                            post_year = int(post_year_str)
                        except ValueError:
                            post_year = None
                    else:
                        if created_at_str:
                            try:
                                from datetime import datetime
                                if created_at_str.endswith('Z') or '+' in created_at_str:
                                    created_at_str_clean = created_at_str.replace('Z', '+00:00') if created_at_str.endswith('Z') else created_at_str
                                    dt = datetime.fromisoformat(created_at_str_clean)
                                    post_year = dt.year
                                else:
                                    dt = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S %z %Y')
                                    post_year = dt.year
                            except (ValueError, TypeError):
                                post_year = None
                        else:
                            post_year = None

                    if post_year is not None:
                        if 1800 <= age_val <= post_year:
                            birth_year = age_val
                        else:
                            birth_year = post_year - age_val
                except ValueError:
                    pass
            
            details = {
                'birth_year': birth_year,
                'raw_ages': raw_ages
            }
            
            # Store details for both author ID and name if available
            if author_id:
                user_details[author_id] = details
            if author_name:
                user_details[author_name] = details
    
    return user_details


def main():
    parser = argparse.ArgumentParser(description="Collect posts from self-identified TUSC users and compute linguistic features")
    parser.add_argument("--input_file", required=True, help="Path to input parquet file")
    parser.add_argument("--self_identified_csv", required=True, help="Output CSV from identify_self_users_tusc.py")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for user posts with linguistic features")
    parser.add_argument("--split", choices=["city", "country"], help="Data split type (auto-determined if not specified)")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with limited samples")
    parser.add_argument("--test_samples", type=int, default=10000, help="Number of samples in test mode")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")
    parser.add_argument("--n_workers", type=int, default=32, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for parallel processing")
    
    args = parser.parse_args()
    
    # Auto-determine split if not provided
    if args.split is None:
        args.split = determine_tusc_split(args.input_file)
    
    logger.info(f"Using split type: {args.split}")
    
    # Ensure output directory exists
    ensure_output_directory(args.output_csv)
    
    # Load target user IDs
    target_user_ids = load_tusc_user_ids(args.self_identified_csv)
    if not target_user_ids:
        logger.warning("No target user IDs found. Exiting.")
        return
    
    # Load user details including raw ages and birthyears
    user_details = load_tusc_user_details(args.self_identified_csv)
    logger.info(f"Loaded user details for {len(user_details)} users.")
    
    # Setup Dask cluster (only for non-test mode)
    client = None
    if not args.test_mode:
        client = setup_dask_cluster(
            n_workers=args.n_workers,
            memory_per_worker=args.memory_per_worker,
            use_slurm=args.use_slurm
        )
    
    try:
        # Process TUSC file
        result_df = load_tusc_file(
            input_file=args.input_file,
            target_user_ids=target_user_ids,
            split=args.split,
            mode="user_posts",
            test_mode=args.test_mode,
            test_samples=args.test_samples,
            include_features=True,
            chunk_size=args.chunk_size,
            client=client
        )
    
        logger.info(f"Processed {len(result_df)} rows from target users")

    
        # Add birthyear and raw ages columns to the dataframe
        user_id_col = "UserID" if args.split == "country" else "userID"
        user_name_col = "UserName" if args.split == "country" else "userName"
        
        # Create new columns
        result_df["AuthorBirthYear"] = ""
        result_df["AuthorRawAges"] = ""
        
        # Fill in the details for each row
        for idx, row in result_df.iterrows():
            user_id = str(row[user_id_col]) if not pd.isna(row[user_id_col]) else ""
            user_name = str(row[user_name_col]) if user_name_col in result_df.columns and not pd.isna(row[user_name_col]) else ""
            
            # Try to find details by user ID first, then by user name
            details = None
            if user_id and user_id in user_details:
                details = user_details[user_id]
            elif user_name and user_name in user_details:
                details = user_details[user_name]
            
            if details:
                result_df.at[idx, "AuthorBirthYear"] = details.get('birth_year', "")
                result_df.at[idx, "AuthorRawAges"] = details.get('raw_ages', "")
        
            # Write output
            separator = '\t' if args.output_tsv else ','
            file_extension = 'tsv' if args.output_tsv else 'csv'
            output_file = args.output_csv.replace('.csv', f'.{file_extension}') if args.output_tsv else args.output_csv
            
            result_df.to_csv(output_file, index=False, sep=separator)
            
            logger.info(f"Output written to {output_file}")
            
    finally:
        if client is not None:
            cleanup_cluster(client, args.use_slurm)


if __name__ == "__main__":
    main()