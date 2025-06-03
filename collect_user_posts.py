import argparse
import json
import logging
import os
import csv
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime, timezone

import dask.bag as db
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from tqdm import tqdm

from helpers import get_all_jsonl_files, filter_entry, extract_columns

# Assuming feature computation utilities live in compute_features.py (will be added next)
from compute_features import (
    emotions,
    compute_all_features,
    vad_dict,
    emotion_dict,
    worry_dict,
    tense_dict,
    moraltrust_dict,
    socialwarmth_dict,
    warmth_dict,
    load_body_parts,
)

logger = logging.getLogger("collect_user_posts")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def flatten_post_to_csv_row(post_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested post structure for CSV output with CapitalCase headers."""
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
    
    # Author information (flattened from nested structure)
    author = post_data.get("author", {})
    if isinstance(author, dict):
        row["AuthorName"] = author.get("name", "")
        row["AuthorAge"] = author.get("age", "")
    else:
        # Fallback for flat author field
        row["AuthorName"] = str(author) if author else ""
        row["AuthorAge"] = ""
    
    # All linguistic features - keep their original CapitalCase names
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
        # Body part mention fields (actual features from compute_prefixed_body_part_mentions)
        "MyBPM", "YourBPM", "HerBPM", "HisBPM", "TheirBPM", "HasBPM",
        # Individual pronoun fields (actual features from compute_individual_pronouns)
        "PRNHasI", "PRNHasMe", "PRNHasMy", "PRNHasMine", "PRNHasWe", "PRNHasOur", "PRNHasOurs",
        "PRNHasYou", "PRNHasYour", "PRNHasYours",
        "PRNHasShe", "PRNHasHer", "PRNHasHers", "PRNHasHe", "PRNHasHim", "PRNHasHis",
        "PRNHasThey", "PRNHasThem", "PRNHasTheir", "PRNHasTheirs",
    ]
    
    # Add all feature fields with default empty values
    for field in feature_fields:
        row[field] = post_data.get(field, "")
    
    return row


def load_user_ids(self_id_csv: str, data_source: str = "reddit") -> Dict[str, int]:
    """Return a mapping from username → inferred *birth_year*.

    We take the age from the self-identification CSV and combine it with the 
    timestamp of that post to approximate a birth year. Keys are usernames.
    """

    # Detect delimiter automatically – we support both comma-separated (.csv) and
    # tab-separated (.tsv) files. This allows the downstream `collect_user_posts`
    # script to work regardless of whether the previous self-identification
    # stage was executed with the `--output_tsv` flag.

    delimiter = "\t" if self_id_csv.lower().endswith(".tsv") else ","

    id_to_birth: Dict[str, int] = {}

    with open(self_id_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            # Skip rows without age data
            age_str = record.get("SelfIdentificationAgeMajorityVote", "").strip()
            if not age_str:
                continue
                
            try:
                age_val = int(age_str)
            except ValueError:
                continue  # non-numeric match – skip

            # Handle timestamp differences between data sources
            if data_source == "tusc":
                # TUSC uses PostCreatedAt with different timestamp format and PostYear
                created_at_str = record.get("PostCreatedAt", "").strip()
                post_year_str = record.get("PostYear", "").strip()
                
                # Try to get year from PostYear field first, then from timestamp
                if post_year_str:
                    try:
                        post_year = int(post_year_str)
                    except ValueError:
                        post_year = None
                else:
                    # Try to parse timestamp
                    if created_at_str:
                        try:
                            # TUSC timestamps are in different formats
                            # Try ISO format first, then Twitter format
                            if created_at_str.endswith('Z') or '+' in created_at_str:
                                # ISO format
                                created_at_str_clean = created_at_str.replace('Z', '+00:00') if created_at_str.endswith('Z') else created_at_str
                                dt = datetime.fromisoformat(created_at_str_clean)
                                post_year = dt.year
                            else:
                                # Twitter format: 'Mon Oct 10 20:16:13 +0000 2011'
                                dt = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S %z %Y')
                                post_year = dt.year
                        except (ValueError, TypeError):
                            continue  # invalid timestamp
                    else:
                        continue  # need timestamp to estimate birth year
                
                if not post_year:
                    continue
                    
            else:  # reddit
                created_utc_str = record.get("PostCreatedUtc", "").strip()
                if not created_utc_str:
                    continue  # need post timestamp to estimate birth year

                # Convert created_utc to float if it's a string
                try:
                    created_utc = float(created_utc_str)
                    post_year = datetime.fromtimestamp(created_utc, timezone.utc).year
                except (ValueError, TypeError, OSError):
                    continue  # invalid timestamp

            # Decide whether *age_val* is a literal age (e.g. "24") or a 4-digit
            # birth year (e.g. "1998").
            if 1800 <= age_val <= post_year:  # treat as YYYY birth year
                birth_year = age_val
            else:
                birth_year = post_year - age_val

            # Map username identifier - handle different data sources
            if data_source == "tusc":
                # TUSC can use either Author (UserID) or AuthorName depending on split
                author_name_val = record.get("Author", "").strip()
                # For TUSC, also try AuthorName if Author is empty
                if not author_name_val:
                    author_name_val = record.get("AuthorName", "").strip()
            else:  # reddit
                author_name_val = record.get("Author", "").strip()

            # Skip automated accounts - AutoModerator and Bot entries (Reddit-specific)
            if data_source == "reddit" and author_name_val in ("AutoModerator", "Bot"):
                continue

            if author_name_val and author_name_val not in {"", "[deleted]"}:
                id_to_birth.setdefault(author_name_val, birth_year)

    return id_to_birth


def process_file(file_path: str, user_birthyears: Dict[str, int], split: str, min_words: int, max_words: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Ensure we use the same identifier logic as in *identify_self_users.py*
            # Use .get() to safely handle missing fields in older posts
            author_fullname = entry.get("author_fullname")
            author_name = entry.get("author")

            # Skip automated accounts - AutoModerator and Bot entries
            if author_name in ("AutoModerator", "Bot"):
                continue

            # Retrieve inferred birth year if this post was written by a
            # self-identified user (lookup by author_fullname or username).
            birth_year = None

            # First try to match using author_fullname (preferred method)
            if author_fullname and author_fullname in user_birthyears:
                birth_year = user_birthyears[author_fullname]
            # Fallback to author name if author_fullname is not available or not found
            elif author_name and author_name in user_birthyears:
                birth_year = user_birthyears[author_name]
                # Log when we're using the fallback method
                if not author_fullname:
                    logger.debug(f"Using author name fallback for post {entry.get('id', 'unknown')} by {author_name}")
                else:
                    logger.debug(f"Author fullname {author_fullname} not found, using author name {author_name} for post {entry.get('id', 'unknown')}")

            if birth_year is None:
                continue

            if not filter_entry(entry, split=split, min_words=min_words, max_words=max_words):
                continue

            post_data = extract_columns(entry, None)
            # Replace flat author string with detailed object ------------------
            post_data["author"] = {
                "name": author_name,
                # Placeholders for future demographics ---------------------
                "age": None,
            }
            # Compute dynamic age if we have a birth year and timestamp ------
            created_utc = entry.get("created_utc")
            if created_utc and birth_year:
                try:
                    # Convert created_utc to float if it's a string
                    if isinstance(created_utc, str):
                        created_utc = float(created_utc)
                    post_year = datetime.fromtimestamp(created_utc, timezone.utc).year
                    age_val = max(0, post_year - birth_year)
                    post_data["author"]["age"] = age_val
                except (ValueError, TypeError, OSError):
                    # Invalid timestamp, skip age computation
                    pass
            # Compute linguistic features if possible.
            if compute_all_features:
                feats = compute_all_features(
                    post_data["selftext"],
                    vad_dict,
                    emotion_dict,
                    emotions,
                    worry_dict,
                    tense_dict,
                    moraltrust_dict,
                    socialwarmth_dict,
                    warmth_dict,
                )
                post_data.update(feats)

            results.append(post_data)

    return results


def run_pipeline(
    input_dir: str,
    self_id_csv: str,
    output_csv: str,
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
    output_tsv: bool = False,
):
    user_birthyears = load_user_ids(self_id_csv)
    logger.info(f"Loaded {len(user_birthyears)} unique users with self-identification.")

    files = get_all_jsonl_files(input_dir)
    if not files:
        raise ValueError(f"No RS_* files found inside {input_dir}")
    logger.info(f"Scanning {len(files)} JSONL files for posts written by these users.")

    if use_slurm:
        os.makedirs("logs", exist_ok=True)
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
        client = Client(n_workers=n_workers, threads_per_worker=1, memory_limit=memory_per_worker)
    logger.info(f"Dashboard {client.dashboard_link}")

    bag = db.from_sequence(files, npartitions=len(files))
    processed_bag = bag.map(
        lambda fp: process_file(fp, user_birthyears=user_birthyears, split=split, min_words=min_words, max_words=max_words)
    ).flatten()

    with ProgressBar():
        results = processed_bag.compute()

    logger.info(f"Writing {len(results)} user posts to {output_csv}")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    if results:
        # Flatten results for CSV
        csv_rows = [flatten_post_to_csv_row(result) for result in results]
        
        # Define all possible field names in logical order
        fieldnames = [
            # Post information
            "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
            "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath",
            # Author information  
            "AuthorName", "AuthorAge",
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
        
        # Write to CSV or TSV
        separator = '\t' if output_tsv else ','
        file_extension = 'tsv' if output_tsv else 'csv'
        output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
        
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()
            
            for row in tqdm(csv_rows, desc=f"Writing {file_extension.upper()}"):
                writer.writerow(row)
    else:
        logger.warning("No results found. Creating empty CSV file.")
        # Determine output file and separator for empty file case
        separator = '\t' if output_tsv else ','
        file_extension = 'tsv' if output_tsv else 'csv'
        output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
        
        # Create empty CSV with headers
        fieldnames = [
            "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc", 
            "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath",
            "AuthorName", "AuthorAge", "WordCount"
        ]
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()

    client.close()
    if use_slurm:
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all posts written by self-identified users")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--self_identified_csv", required=True, help="Output of identify_self_users.py")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for user posts")
    parser.add_argument("--split", choices=["text", "multimodal"], default="text")
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--memory_per_worker", type=str, default="4GB")
    parser.add_argument("--use_slurm", action="store_true")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")

    a = parser.parse_args()
    run_pipeline(
        input_dir=a.input_dir,
        self_id_csv=a.self_identified_csv,
        output_csv=a.output_csv,
        split=a.split,
        min_words=a.min_words,
        max_words=a.max_words,
        n_workers=a.n_workers,
        memory_per_worker=a.memory_per_worker,
        use_slurm=a.use_slurm,
        output_tsv=a.output_tsv,
    ) 