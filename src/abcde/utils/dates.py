"""Date parsing utilities."""

from datetime import datetime
from typing import Optional


def parse_tusc_created_at_year(created_at: str) -> Optional[int]:
    """Parse TUSC createdAt field and extract year from either format:
    - "Wed Apr 01 14:35:59 +0000 2020"
    - "2015-04-10T02:47:38.000Z"

    Returns None if parsing fails.
    """
    if not isinstance(created_at, str) or not created_at.strip():
        return None

    created_at = created_at.strip()

    # Try format 1: "Wed Apr 01 14:35:59 +0000 2020"
    try:
        dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        return dt.year
    except (ValueError, TypeError):
        pass

    # Try format 2: "2015-04-10T02:47:38.000Z"
    try:
        # Handle both with and without milliseconds
        if created_at.endswith("Z"):
            if "." in created_at:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        else:
            # Handle ISO format without Z
            if "." in created_at:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S")
        return dt.year
    except (ValueError, TypeError):
        pass

    return None
