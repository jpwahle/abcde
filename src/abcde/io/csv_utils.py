"""CSV/TSV reading and writing utilities."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from abcde.utils.dates import parse_tusc_created_at_year
from abcde.utils.text import clean_text_newlines


def ensure_output_directory(path: str) -> None:
    """Ensure the output directory exists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def get_csv_fieldnames(
    data_source: str,
    split: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[str]:
    """Get static CSV/TSV header fields based on data source and stage."""
    # User-level headers
    if stage == "users":
        user_cols = [
            "Author",
            "DMGMajorityBirthyear",
            "DMGRawBirthyearExtractions",
            "DMGRawExtractedAge",
            "DMGRawExtractedGender",
            "DMGRawExtractedCity",
            "DMGCountryMappedFromExtractedCity",
            "DMGRawExtractedCountry",
            "DMGRawExtractedReligion",
            "DMGMainReligionMappedFromExtractedReligion",
            "DMGMainCategoryMappedFromExtractedReligion",
            "DMGRawExtractedOccupation",
            "DMGSOCTitleMappedFromExtractedOccupation",
        ]
    else:
        user_cols = ["Author", "DMGAgeAtPost"]

    if data_source == "tusc":
        base = user_cols + [
            "PostID",
            "PostText",
            "PostCreatedAt",
            "PostYear",
            "PostMonth",
        ]
        if split == "city":
            loc = ["PostCity", "PostPlace", "PostPlaceID", "PostPlaceType"]
        else:
            loc = [
                "PostCountry",
                "PostMyCountry",
                "PostPlace",
                "PostPlaceID",
                "PostPlaceType",
            ]
        return base + loc

    # Reddit headers
    base = user_cols + [
        "PostID",
        "PostSubreddit",
        "PostTitle",
        "PostSelftext",
        "PostCreatedUtc",
        "PostScore",
        "PostNumComments",
        "PostPermalink",
        "PostUrl",
        "PostMediaPath",
    ]
    return base


def flatten_result_to_csv_row(
    result: Dict[str, Any],
    data_source: str,
    split: Optional[str] = None,
    stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Flatten a result dictionary to a CSV row."""
    row: Dict[str, Any] = {}

    # Author column
    if data_source == "tusc":
        uid_raw = result.get("UserID", "") or result.get("userID", "") or ""
        uid_str = str(uid_raw).strip()
        if uid_str.endswith(".0") and uid_str[:-2].isdigit():
            uid_str = uid_str[:-2]
        row["Author"] = uid_str
    else:
        row["Author"] = result.get("author", "") or ""

    # Compute majority birthyear and raw birthyear extractions when available
    if "self_identification" in result:
        self_id = result["self_identification"]
        if data_source == "tusc":
            try:
                ref_year = int(result.get("Year", ""))
            except (TypeError, ValueError):
                ref_year = parse_tusc_created_at_year(result.get("createdAt", ""))
        else:
            ts = result.get("post", {}).get("created_utc")
            ref_year = datetime.utcfromtimestamp(int(ts)).year

        raw_matches: List[str] = []
        majority_birthyear: Optional[int] = None
        if "resolved_age" in self_id:
            raw_matches = list(self_id["resolved_age"].get("raw_matches", []))
            age_val = self_id["resolved_age"].get("age")
            if isinstance(age_val, int):
                if stage == "users" and not (13 <= age_val <= 99):
                    return None
                majority_birthyear = ref_year - age_val
        else:
            raw_matches = list(self_id.get("age", []))
            if raw_matches:
                try:
                    val = int(raw_matches[0])
                except ValueError:
                    val = None
                if isinstance(val, int):
                    if (ref_year - 99) <= val <= (ref_year - 13):
                        majority_birthyear = val
                        if stage == "users":
                            age_at_ref = ref_year - val
                            if not (13 <= age_at_ref <= 99):
                                return None
                    elif 13 <= val <= 99:
                        majority_birthyear = ref_year - val
                        if stage == "users" and not (13 <= val <= 99):
                            return None

        raw_birthyears: List[int] = []
        for m in raw_matches:
            try:
                v = int(m)
            except ValueError:
                continue
            if (ref_year - 99) <= v <= (ref_year - 13):
                raw_birthyears.append(v)
            elif 13 <= v <= 99:
                raw_birthyears.append(ref_year - v)

        row["DMGMajorityBirthyear"] = majority_birthyear or ""
        row["DMGRawBirthyearExtractions"] = "|".join(str(x) for x in raw_birthyears)

    # Include age at posting if available (stage2)
    if "DMGAgeAtPost" in result:
        age_at_post = result.get("DMGAgeAtPost", "")
        if stage == "posts" and age_at_post != "":
            try:
                age_val = int(age_at_post)
                if not (13 <= age_val <= 99):
                    return None
            except (ValueError, TypeError):
                return None
        row["DMGAgeAtPost"] = age_at_post

    if data_source == "tusc":
        row["PostID"] = result.get("TweetID", "")
        row["PostText"] = result.get("Tweet", "")
        row["PostCreatedAt"] = result.get("createdAt", "")
        row["PostYear"] = result.get("Year", "")
        row["PostMonth"] = result.get("Month", "")
        if split == "city":
            row["PostCity"] = result.get("City", "")
        else:
            row["PostCountry"] = result.get("Country", "")
            row["PostMyCountry"] = result.get("MyCountry", "")
        row["PostPlace"] = result.get("Place", "")
        row["PostPlaceID"] = result.get("PlaceID", "")
        row["PostPlaceType"] = result.get("PlaceType", "")

        static_keys = {
            "TweetID", "Tweet", "createdAt", "Year", "Month", "City", "Country",
            "MyCountry", "Place", "PlaceID", "PlaceType", "UserID", "userID",
            "userName", "Author", "DMGAgeAtPost", "DMGMajorityBirthyear",
            "DMGRawBirthyearExtractions",
        }
        # Include demographic fields for both users and posts stages
        for key, val in result.items():
            if (
                key not in static_keys
                and key not in row
                and key not in {"File", "RowNum", "self_identification"}
                and key.startswith("DMG")
            ):
                row[key] = val
        # Include other dynamic fields only for posts stage
        if stage == "posts":
            for key, val in result.items():
                if (
                    key not in static_keys
                    and key not in row
                    and key not in {"File", "RowNum", "self_identification"}
                    and not key.startswith("DMG")
                ):
                    row[key] = val
    else:
        post = result.get("post", result)
        row["PostID"] = post.get("id", "")
        row["PostSubreddit"] = post.get("subreddit", "")
        row["PostTitle"] = clean_text_newlines(post.get("title", ""))
        row["PostSelftext"] = clean_text_newlines(post.get("selftext", ""))
        row["PostCreatedUtc"] = post.get("created_utc", "")
        row["PostScore"] = post.get("score", "")
        row["PostNumComments"] = post.get("num_comments", "")
        row["PostPermalink"] = post.get("permalink", "")
        row["PostUrl"] = post.get("url", "")
        row["PostMediaPath"] = post.get("media_path", "")

        static_keys = {
            "id", "subreddit", "title", "selftext", "created_utc", "score",
            "num_comments", "permalink", "url", "media_path", "author",
        }
        # Include demographic fields
        for key, val in result.items():
            if (
                key not in static_keys
                and key not in row
                and key not in {"File", "RowNum", "self_identification", "post"}
                and key.startswith("DMG")
            ):
                row[key] = val
        # Include other dynamic fields only for posts stage
        if stage == "posts":
            for key, val in post.items():
                if key not in static_keys and key not in {
                    "File", "RowNum", "self_identification",
                }:
                    row[key] = val

    return row


def write_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str,
    output_tsv: bool,
    data_source: str,
    split: Optional[str] = None,
) -> None:
    """Write results to a CSV/TSV file."""
    sep = "\t" if output_tsv else ","
    ext = "tsv" if output_tsv else "csv"
    out = output_file.replace(".csv", f".{ext}") if output_tsv else output_file
    ensure_output_directory(out)

    fname = os.path.basename(out)
    stage = "posts" if "posts" in fname else "users"

    if results:
        rows = [
            flatten_result_to_csv_row(r, data_source, split, stage) for r in results
        ]
        rows = [row for row in rows if row is not None]

        if rows:
            static_fields = get_csv_fieldnames(data_source, split, stage)
            extra_fields = sorted(
                {k for row in rows for k in row.keys() if k not in static_fields}
            )
            fieldnames = static_fields + extra_fields
            with open(out, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        else:
            with open(out, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=get_csv_fieldnames(data_source, split, stage),
                    delimiter=sep,
                )
                writer.writeheader()
    else:
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=get_csv_fieldnames(data_source, split, stage),
                delimiter=sep,
            )
            writer.writeheader()


def append_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str,
    output_tsv: bool,
    data_source: str,
    split: Optional[str] = None,
) -> None:
    """Append rows to a CSV/TSV while creating the file with a header if needed."""
    sep = "\t" if output_tsv else ","
    ext = "tsv" if output_tsv else "csv"
    out = output_file.replace(".csv", f".{ext}") if output_tsv else output_file
    ensure_output_directory(out)

    fname = os.path.basename(out)
    stage = "posts" if "posts" in fname else "users"

    if not results:
        return

    rows = [flatten_result_to_csv_row(r, data_source, split, stage) for r in results]
    rows = [row for row in rows if row is not None]

    if not rows:
        return

    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(sep)
        fieldnames = header
        write_header = False
    else:
        static_fields = get_csv_fieldnames(data_source, split, stage)
        extra_fields = sorted(
            {k for row in rows for k in row.keys() if k not in static_fields}
        )
        fieldnames = static_fields + extra_fields
        write_header = True

    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

