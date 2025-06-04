"""
TUSC-specific user data loading utilities.
"""
import csv
import logging
from typing import Dict, Set
from datetime import datetime

logger = logging.getLogger("tusc.user_loader")


def load_tusc_user_birthyears(self_id_csv: str) -> Dict[str, int]:
    """Return a mapping from user ID → inferred birth_year for TUSC data.

    Args:
        self_id_csv: Path to CSV file from identify_self_users.py (TUSC mode)
        
    Returns:
        Dictionary mapping user IDs to birth years
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
                    age_val = int(float(age_str))
                except ValueError:
                    age_val = None

            # If we have age information, estimate birth year
            if age_val is not None:
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
                    if created_at_str:
                        try:
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

            # Map username identifier - handle different data sources
            # TUSC can use either Author (UserID) or AuthorName depending on split
            author_name_val = record.get("Author", "").strip()
            # For TUSC, also try AuthorName if Author is empty
            if not author_name_val:
                author_name_val = record.get("AuthorName", "").strip()

            # Always record the user—even if we could not derive a reliable birth year
            if author_name_val and author_name_val not in {"", "[deleted]"}:
                id_to_birth.setdefault(author_name_val, birth_year)

    return id_to_birth


def load_tusc_user_ids(self_id_csv: str) -> Set[str]:
    """Load the set of user IDs that have been self-identified from TUSC data.
    
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
    
    logger.info(f"Loaded {len(user_ids)} unique TUSC user IDs from {self_id_csv}")
    return user_ids