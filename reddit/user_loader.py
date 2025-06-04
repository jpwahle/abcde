"""
Reddit-specific user data loading utilities.
"""
import csv
import logging
from typing import Dict, Set
from datetime import datetime, timezone

logger = logging.getLogger("reddit.user_loader")


def load_reddit_user_birthyears(self_id_csv: str) -> Dict[str, int]:
    """Return a mapping from username → inferred birth_year for Reddit data.

    Args:
        self_id_csv: Path to CSV file from identify_self_users.py (Reddit mode)
        
    Returns:
        Dictionary mapping usernames to birth years
    """
    delimiter = "\t" if self_id_csv.lower().endswith(".tsv") else ","
    id_to_birth: Dict[str, int] = {}

    with open(self_id_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            birth_year = None
            
            # Skip rows without age data
            age_str = record.get("SelfIdentificationAgeMajorityVote", "").strip()
            if not age_str:
                age_val = None
            else:
                try:
                    age_val = int(float(age_str))  # handles "25" as well as "25.0"
                except ValueError:
                    age_val = None

            # If we have age information, estimate birth year
            if age_val is not None:
                created_utc_str = record.get("PostCreatedUtc", "").strip()
                if created_utc_str:
                    try:
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

            # Map username identifier
            author_name_val = record.get("Author", "").strip()

            # Skip automated accounts
            if author_name_val in ("AutoModerator", "Bot"):
                continue

            # Always record the user—even if we could not derive a reliable birth year
            if author_name_val and author_name_val not in {"", "[deleted]"}:
                id_to_birth.setdefault(author_name_val, birth_year)

    return id_to_birth


def load_reddit_user_ids(self_id_csv: str) -> Set[str]:
    """Load set of Reddit user IDs from self-identification CSV.
    
    Args:
        self_id_csv: Path to CSV file from identify_self_users.py (Reddit mode)
        
    Returns:
        Set of user IDs/names
    """
    delimiter = "\t" if self_id_csv.lower().endswith(".tsv") else ","
    user_ids = set()
    
    with open(self_id_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            author_name = record.get("Author", "").strip()
            if author_name and author_name not in ("AutoModerator", "Bot", "[deleted]"):
                user_ids.add(author_name)
    
    logger.info(f"Loaded {len(user_ids)} unique Reddit user IDs from {self_id_csv}")
    return user_ids