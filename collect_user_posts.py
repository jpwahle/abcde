import argparse
import json
import logging
import os
from typing import Dict, Any, List, Set
from datetime import datetime, timezone

import dask.bag as db
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
)

logger = logging.getLogger("collect_user_posts")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_user_ids(self_id_jsonl: str) -> Dict[str, int]:
    """Return a mapping from user identifier → inferred *birth_year*.

    We take the first age/birth-year statement found in the self-identification
    JSONL and combine it with the timestamp of that post to approximate a
    birth year. Keys include both author_id (if available) and the username so
    that we can match posts regardless of which identifier is present in a
    given monthly dump.
    """

    id_to_birth: Dict[str, int] = {}

    with open(self_id_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            self_id_info = record.get("self_identification", {})
            age_hits = self_id_info.get("age", [])
            if not age_hits:
                continue  # cannot infer birth year

            first_hit = age_hits[0]
            try:
                num = int(first_hit)
            except ValueError:
                continue  # non-numeric match – skip

            post_meta = record.get("post", {})
            created_utc = post_meta.get("created_utc")
            if created_utc is None:
                continue  # need post timestamp to estimate birth year

            # Convert created_utc to float if it's a string
            try:
                if isinstance(created_utc, str):
                    created_utc = float(created_utc)
                post_year = datetime.fromtimestamp(created_utc, timezone.utc).year
            except (ValueError, TypeError, OSError):
                continue  # invalid timestamp

            # Decide whether *num* is a literal age (e.g. "24") or a 4-digit
            # birth year (e.g. "1998").
            if 1800 <= num <= post_year:  # treat as YYYY birth year
                birth_year = num
            else:
                birth_year = post_year - num

            # Map both stable and username identifiers -------------------
            author_id_val = record["author_id"]
            author_name_val = record["author"]

            if author_id_val:
                id_to_birth.setdefault(author_id_val, birth_year)
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
            # Use .get() to safely handle missing author_id field in older posts
            author_id_raw = entry.get("author_id")
            author_id_or_fullname = entry.get("author_id") or entry.get("author_fullname")
            author_name = entry.get("author")

            # Retrieve inferred birth year if this post was written by a
            # self-identified user (lookup by either stable id or username).
            birth_year = None
            
            # First try to match using author_id or author_fullname (preferred method)
            if author_id_or_fullname and author_id_or_fullname in user_birthyears:
                birth_year = user_birthyears[author_id_or_fullname]
            # Fallback to author name if author_id is not available or not found
            elif author_name and author_name in user_birthyears:
                birth_year = user_birthyears[author_name]
                # Log when we're using the fallback method
                if not author_id_or_fullname:
                    logger.debug(f"Using author name fallback for post {entry.get('id', 'unknown')} by {author_name}")
                else:
                    logger.debug(f"Author ID {author_id_or_fullname} not found, using author name {author_name} for post {entry.get('id', 'unknown')}")

            if birth_year is None:
                continue

            if not filter_entry(entry, split=split, min_words=min_words, max_words=max_words):
                continue

            post_data = extract_columns(entry, None)
            # Replace flat author string with detailed object ------------------
            post_data["author"] = {
                "name": author_name,
                "id": author_id_raw,  # strictly the original *author_id* field
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
    self_id_jsonl: str,
    output_jsonl: str,
    split: str = "text",
    min_words: int = 5,
    max_words: int = 1000,
    n_workers: int = 16,
    memory_per_worker: str = "4GB",
    use_slurm: bool = False,
):
    user_birthyears = load_user_ids(self_id_jsonl)
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

    logger.info(f"Writing {len(results)} user posts to {output_jsonl}")
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for r in tqdm(results, desc="Writing"):
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    client.close()
    if use_slurm:
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all posts written by self-identified users")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--self_identified_jsonl", required=True, help="Output of identify_self_users.py")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--split", choices=["text", "multimodal"], default="text")
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--memory_per_worker", type=str, default="4GB")
    parser.add_argument("--use_slurm", action="store_true")

    a = parser.parse_args()
    run_pipeline(
        input_dir=a.input_dir,
        self_id_jsonl=a.self_identified_jsonl,
        output_jsonl=a.output_jsonl,
        split=a.split,
        min_words=a.min_words,
        max_words=a.max_words,
        n_workers=a.n_workers,
        memory_per_worker=a.memory_per_worker,
        use_slurm=a.use_slurm,
    ) 