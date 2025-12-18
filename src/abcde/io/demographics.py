"""Demographics aggregation and formatting utilities."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from abcde.utils.dates import parse_tusc_created_at_year


def format_demographic_detections_for_output(
    detections: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Format demographic detections with user-specified field names."""
    output = {}

    # City fields
    if "city" in detections:
        city_data = detections["city"]
        output["DMGRawExtractedCity"] = city_data.get("raw", [])
        output["DMGCountryMappedFromExtractedCity"] = city_data.get(
            "country_mapped", []
        )

    # Religion fields
    if "religion" in detections:
        religion_data = detections["religion"]
        output["DMGRawExtractedReligion"] = religion_data.get("raw", [])
        output["DMGMainReligionMappedFromExtractedReligion"] = religion_data.get(
            "main_religion_mapped", []
        )
        output["DMGMainCategoryMappedFromExtractedReligion"] = religion_data.get(
            "category_mapped", []
        )

    # Occupation fields
    if "occupation" in detections:
        occupation_data = detections["occupation"]
        output["DMGRawExtractedOccupation"] = occupation_data.get("raw", [])
        output["DMGSOCTitleMappedFromExtractedOccupation"] = occupation_data.get(
            "soc_mapped", []
        )

    # Other raw fields (age, gender, country)
    if "age" in detections:
        output["DMGRawExtractedAge"] = detections["age"].get("raw", [])

    if "gender" in detections:
        output["DMGRawExtractedGender"] = detections["gender"].get("raw", [])

    if "country" in detections:
        output["DMGRawExtractedCountry"] = detections["country"].get("raw", [])

    # Normalize all DMG feature values to lowercase
    for k, v in output.items():
        if isinstance(v, list):
            output[k] = [s.lower() if isinstance(s, str) else s for s in v]
        elif isinstance(v, str):
            output[k] = v.lower()

    return output


def aggregate_user_demographics(df: pd.DataFrame, data_source: str) -> pd.DataFrame:
    """Aggregate demographics per author using majority vote across multiple posts."""
    from abcde.core.detector import SelfIdentificationDetector
    
    detector = SelfIdentificationDetector()

    def get_post_year(group):
        if data_source == "tusc":
            if "PostYear" in group.columns:
                yrs = (
                    pd.to_numeric(group["PostYear"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                if not yrs.empty:
                    return yrs.max()
            if "Year" in group.columns:
                yrs = pd.to_numeric(group["Year"], errors="coerce").dropna().astype(int)
                if not yrs.empty:
                    return yrs.max()
            year_candidates = []
            for col in ["PostCreatedAt", "createdAt"]:
                if col in group.columns:
                    for val in group[col].dropna():
                        y = parse_tusc_created_at_year(str(val))
                        if y is not None:
                            year_candidates.append(y)
            if year_candidates:
                return max(year_candidates)
            raise ValueError(
                "Unable to determine post year for TUSC group"
            )
        elif data_source == "reddit" and "PostCreatedUtc" in group.columns:
            years = []
            for utc in group["PostCreatedUtc"].dropna():
                try:
                    years.append(datetime.utcfromtimestamp(int(utc)).year)
                except (ValueError, TypeError):
                    pass
            if years:
                return max(years)
            raise ValueError(
                "Unable to determine post year for Reddit group"
            )
        else:
            raise ValueError("Unsupported data source or missing date columns")

    def aggregate_group(group):
        agg_row = {"Author": group.name}

        # Special handling for birthyear
        all_raw_birthyears = []
        for val in group.get("DMGRawBirthyearExtractions", pd.Series()).dropna():
            all_raw_birthyears.extend(str(val).split("|"))
        all_raw_birthyears = [r.strip() for r in all_raw_birthyears if r.strip()]

        if all_raw_birthyears:
            ref_year = get_post_year(group)
            resolution = detector.resolve_multiple_ages(
                all_raw_birthyears, current_year=ref_year
            )
            if resolution and resolution[1] >= 0.5:
                resolved_age, _ = resolution
                agg_row["DMGMajorityBirthyear"] = ref_year - resolved_age
            else:
                agg_row["DMGMajorityBirthyear"] = pd.NA
            agg_row["DMGRawBirthyearExtractions"] = "|".join(
                sorted(set(all_raw_birthyears))
            )
        else:
            agg_row["DMGMajorityBirthyear"] = pd.NA
            agg_row["DMGRawBirthyearExtractions"] = pd.NA

        # For other DMG fields
        for col in [
            c
            for c in group.columns
            if c.startswith("DMG")
            and c not in ["DMGMajorityBirthyear", "DMGRawBirthyearExtractions"]
        ]:
            if "Raw" in col:
                all_items = []
                for val in group[col].dropna():
                    all_items.extend(str(val).split("|"))
                all_items = [item.strip() for item in all_items if item.strip()]
                if all_items:
                    counts = Counter(all_items)
                    most_common = counts.most_common(1)
                    agg_row[col] = most_common[0][0] if most_common else pd.NA
                else:
                    agg_row[col] = pd.NA
            else:
                series = group[col].dropna()
                if not series.empty:
                    mode = series.mode()
                    agg_row[col] = mode[0] if not mode.empty else series.iloc[0]
                else:
                    agg_row[col] = pd.NA

        return pd.Series(agg_row)

    aggregated = (
        df.groupby("Author")
        .apply(aggregate_group, include_groups=False)
        .reset_index(drop=True)
    )

    return aggregated

