"""
Reddit-specific user post collection pipeline.
Collects all posts by self-identified users and computes linguistic features.
"""
import argparse
import logging
import csv
from tqdm import tqdm

from core.cluster import setup_dask_cluster, cleanup_cluster
from core.io_utils import ensure_output_directory
from reddit.data_loader import load_reddit_files_for_user_posts
from reddit.user_loader import load_reddit_user_birthyears

logger = logging.getLogger("collect_user_posts_reddit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_reddit_user_details(self_id_csv: str):
    """Load user details including birthyears and raw ages from self-identification CSV."""
    delimiter = "\t" if self_id_csv.lower().endswith(".tsv") else ","
    user_details = {}
    
    with open(self_id_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            author_name = record.get("Author", "").strip()
            if not author_name or author_name in ("AutoModerator", "Bot", "[deleted]"):
                continue
                
            # Extract birthyear and raw ages
            birth_year = None
            age_str = record.get("SelfIdentificationAgeMajorityVote", "").strip()
            raw_ages = record.get("SelfIdentificationRawAges", "").strip()
            
            if age_str:
                try:
                    age_val = int(float(age_str))
                    # Calculate birth year from age and post timestamp
                    created_utc_str = record.get("PostCreatedUtc", "").strip()
                    if created_utc_str:
                        try:
                            from datetime import datetime, timezone
                            created_utc = float(created_utc_str)
                            post_year = datetime.fromtimestamp(created_utc, timezone.utc).year
                        except (ValueError, TypeError, OSError):
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
            
            user_details[author_name] = {
                'birth_year': birth_year,
                'raw_ages': raw_ages
            }
    
    return user_details


def flatten_reddit_post_to_csv_row(post_data, user_details=None):
    """Flatten Reddit post data for CSV output."""
    row = {}
    
    # Post basic information
    row["PostID"] = post_data.get("id", "")
    row["PostSubreddit"] = post_data.get("subreddit", "")
    row["PostTitle"] = post_data.get("title", "")
    row["PostSelftext"] = post_data.get("selftext", "")
    row["PostCreatedUtc"] = post_data.get("created_utc", "")
    row["PostScore"] = post_data.get("score", "")
    row["PostNumComments"] = post_data.get("num_comments", "")
    row["PostPermalink"] = post_data.get("permalink", "")
    row["PostUrl"] = post_data.get("url", "")
    row["PostMediaPath"] = post_data.get("media_path", "")
    
    # Author information
    author = post_data.get("author", {})
    if isinstance(author, dict):
        author_name = author.get("name", "")
        row["AuthorName"] = author_name
        row["AuthorAge"] = author.get("age", "")
    else:
        author_name = str(author) if author else ""
        row["AuthorName"] = author_name
        row["AuthorAge"] = ""
    
    # Add birthyear and raw ages from user details
    if user_details and author_name in user_details:
        details = user_details[author_name]
        row["AuthorBirthYear"] = details.get('birth_year', "")
        row["AuthorRawAges"] = details.get('raw_ages', "")
    else:
        row["AuthorBirthYear"] = ""
        row["AuthorRawAges"] = ""
    
    # All linguistic features
    feature_fields = [
        "WordCount", 
        # VAD fields
        "NRCAvgValence", "NRCAvgArousal", "NRCAvgDominance",
        "NRCHasHighValenceWord", "NRCHasLowValenceWord", "NRCHasHighArousalWord", "NRCHasLowArousalWord",
        "NRCHasHighDominanceWord", "NRCHasLowDominanceWord",
        "NRCCountHighValenceWords", "NRCCountLowValenceWords", "NRCCountHighArousalWords", "NRCCountLowArousalWords",
        "NRCCountHighDominanceWords", "NRCCountLowDominanceWords",
        # Emotion fields
        "NRCHasAngerWord", "NRCHasAnticipationWord", "NRCHasDisgustWord", "NRCHasFearWord", "NRCHasJoyWord",
        "NRCHasNegativeWord", "NRCHasPositiveWord", "NRCHasSadnessWord", "NRCHasSurpriseWord", "NRCHasTrustWord",
        "NRCCountAngerWords", "NRCCountAnticipationWords", "NRCCountDisgustWords", "NRCCountFearWords", "NRCCountJoyWords",
        "NRCCountNegativeWords", "NRCCountPositiveWords", "NRCCountSadnessWords", "NRCCountSurpriseWords", "NRCCountTrustWords",
        # Anxiety/Calmness fields
        "NRCHasAnxietyWord", "NRCHasCalmnessWord", "NRCAvgAnxiety", "NRCAvgCalmness",
        "NRCHasHighAnxietyWord", "NRCCountHighAnxietyWords", "NRCHasHighCalmnessWord", "NRCCountHighCalmnessWords",
        # Moral Trust fields
        "NRCHasHighMoralTrustWord", "NRCCountHighMoralTrustWord", "NRCHasLowMoralTrustWord", "NRCCountLowMoralTrustWord", "NRCAvgMoralTrustWord",
        # Social Warmth fields
        "NRCHasHighSocialWarmthWord", "NRCCountHighSocialWarmthWord", "NRCHasLowSocialWarmthWord", "NRCCountLowSocialWarmthWord", "NRCAvgSocialWarmthWord",
        # Warmth fields
        "NRCHasHighWarmthWord", "NRCCountHighWarmthWord", "NRCHasLowWarmthWord", "NRCCountLowWarmthWord", "NRCAvgWarmthWord",
        # Tense fields
        "TIMEHasPastVerb", "TIMECountPastVerbs", "TIMEHasPresentVerb", "TIMECountPresentVerbs",
        "TIMEHasFutureModal", "TIMECountFutureModals", "TIMEHasPresentNoFuture", "TIMEHasFutureReference",
        # Body part mention fields
        "MyBPM", "YourBPM", "HerBPM", "HisBPM", "TheirBPM", "HasBPM",
        # Individual pronoun fields
        "PRNHasI", "PRNHasMe", "PRNHasMy", "PRNHasMine", "PRNHasWe", "PRNHasOur", "PRNHasOurs",
        "PRNHasYou", "PRNHasYour", "PRNHasYours",
        "PRNHasShe", "PRNHasHer", "PRNHasHers", "PRNHasHe", "PRNHasHim", "PRNHasHis",
        "PRNHasThey", "PRNHasThem", "PRNHasTheir", "PRNHasTheirs",
    ]
    
    # Add all feature fields with default empty values
    for field in feature_fields:
        row[field] = post_data.get(field, "")
    
    return row


def main():
    parser = argparse.ArgumentParser(description="Collect all posts written by self-identified Reddit users")
    parser.add_argument("--input_dir", required=True, help="Directory containing RS_*.jsonl files")
    parser.add_argument("--self_identified_csv", required=True, help="Output of identify_self_users_reddit.py")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for user posts")
    parser.add_argument("--split", choices=["text", "multimodal"], default="text", help="Split type")
    parser.add_argument("--min_words", type=int, default=5, help="Minimum word count filter")
    parser.add_argument("--max_words", type=int, default=1000, help="Maximum word count filter")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")

    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_output_directory(args.output_csv)
    
    # Load user birth years (for existing functionality)
    user_birthyears = load_reddit_user_birthyears(args.self_identified_csv)
    logger.info(f"Loaded {len(user_birthyears)} unique users with self-identification.")
    
    # Load user details including raw ages
    user_details = load_reddit_user_details(args.self_identified_csv)
    logger.info(f"Loaded user details for {len(user_details)} users.")
    
    # Setup Dask cluster
    client = setup_dask_cluster(
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm
    )
    
    try:
        # Process Reddit files
        results = load_reddit_files_for_user_posts(
            input_dir=args.input_dir,
            user_birthyears=user_birthyears,
            split=args.split,
            min_words=args.min_words,
            max_words=args.max_words,
            include_features=True,
            client=client
        )
        
        logger.info(f"Writing {len(results)} user posts to {args.output_csv}")
        
        # Write results
        separator = '\t' if args.output_tsv else ','
        file_extension = 'tsv' if args.output_tsv else 'csv'
        output_file = args.output_csv.replace('.csv', f'.{file_extension}') if args.output_tsv else args.output_csv
        
        if results:
            # Flatten results for CSV
            csv_rows = [flatten_reddit_post_to_csv_row(result, user_details) for result in results]
            
            # Define fieldnames
            fieldnames = [
                # Post information
                "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
                "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath",
                # Author information  
                "AuthorName", "AuthorAge", "AuthorBirthYear", "AuthorRawAges",
                # Basic features
                "WordCount",
                # VAD features
                "NRCAvgValence", "NRCAvgArousal", "NRCAvgDominance",
                "NRCHasHighValenceWord", "NRCHasLowValenceWord", "NRCHasHighArousalWord", "NRCHasLowArousalWord",
                "NRCHasHighDominanceWord", "NRCHasLowDominanceWord",
                "NRCCountHighValenceWords", "NRCCountLowValenceWords", "NRCCountHighArousalWords", "NRCCountLowArousalWords",
                "NRCCountHighDominanceWords", "NRCCountLowDominanceWords",
                # Emotion features
                "NRCHasAngerWord", "NRCHasAnticipationWord", "NRCHasDisgustWord", "NRCHasFearWord", "NRCHasJoyWord",
                "NRCHasNegativeWord", "NRCHasPositiveWord", "NRCHasSadnessWord", "NRCHasSurpriseWord", "NRCHasTrustWord",
                "NRCCountAngerWords", "NRCCountAnticipationWords", "NRCCountDisgustWords", "NRCCountFearWords", "NRCCountJoyWords",
                "NRCCountNegativeWords", "NRCCountPositiveWords", "NRCCountSadnessWords", "NRCCountSurpriseWords", "NRCCountTrustWords",
                # Anxiety/Calmness features
                "NRCHasAnxietyWord", "NRCHasCalmnessWord", "NRCAvgAnxiety", "NRCAvgCalmness",
                "NRCHasHighAnxietyWord", "NRCCountHighAnxietyWords", "NRCHasHighCalmnessWord", "NRCCountHighCalmnessWords",
                # Moral Trust features
                "NRCHasHighMoralTrustWord", "NRCCountHighMoralTrustWord", "NRCHasLowMoralTrustWord", "NRCCountLowMoralTrustWord", "NRCAvgMoralTrustWord",
                # Social Warmth features
                "NRCHasHighSocialWarmthWord", "NRCCountHighSocialWarmthWord", "NRCHasLowSocialWarmthWord", "NRCCountLowSocialWarmthWord", "NRCAvgSocialWarmthWord",
                # Warmth features
                "NRCHasHighWarmthWord", "NRCCountHighWarmthWord", "NRCHasLowWarmthWord", "NRCCountLowWarmthWord", "NRCAvgWarmthWord",
                # Tense features
                "TIMEHasPastVerb", "TIMECountPastVerbs", "TIMEHasPresentVerb", "TIMECountPresentVerbs",
                "TIMEHasFutureModal", "TIMECountFutureModals", "TIMEHasPresentNoFuture", "TIMEHasFutureReference",
                # Body part mention features
                "MyBPM", "YourBPM", "HerBPM", "HisBPM", "TheirBPM", "HasBPM",
                # Individual pronoun features
                "PRNHasI", "PRNHasMe", "PRNHasMy", "PRNHasMine", "PRNHasWe", "PRNHasOur", "PRNHasOurs",
                "PRNHasYou", "PRNHasYour", "PRNHasYours",
                "PRNHasShe", "PRNHasHer", "PRNHasHers", "PRNHasHe", "PRNHasHim", "PRNHasHis",
                "PRNHasThey", "PRNHasThem", "PRNHasTheir", "PRNHasTheirs",
            ]
            
            with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
                writer.writeheader()
                
                for row in tqdm(csv_rows, desc=f"Writing {file_extension.upper()}"):
                    writer.writerow(row)
        else:
            logger.warning("No results found. Creating empty CSV file.")
            # Create empty file with basic headers
            with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
                fieldnames = [
                    "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
                    "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath",
                    "AuthorName", "AuthorAge", "AuthorBirthYear", "AuthorRawAges", "WordCount"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
                writer.writeheader()
                
    finally:
        cleanup_cluster(client, args.use_slurm)


if __name__ == "__main__":
    main()