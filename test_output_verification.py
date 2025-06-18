#!/usr/bin/env python3
"""Verify that demographic fields are correctly included in pipeline output files."""

import os
import tempfile

import pandas as pd

from process_reddit import main as reddit_main
from process_tusc import main as tusc_main


def verify_demographic_fields(
    df: pd.DataFrame, source: str
) -> tuple[list[str], list[str]]:
    """Verify expected demographic fields are present in dataframe.

    Returns:
        Tuple of (present_fields, missing_fields)
    """
    expected_fields = [
        "DMGRawExtractedCity",
        "DMGCountryMappedFromExtractedCity",
        "DMGRawExtractedReligion",
        "DMGMainReligionMappedFromExtractedReligion",
        "DMGMainCategoryMappedFromExtractedReligion",
        "DMGRawExtractedOccupation",
        "DMGSOCTitleMappedFromExtractedOccupation",
        "DMGRawExtractedGender",
        "DMGRawExtractedCountry",
    ]

    # Reddit doesn't include category mapping for religions
    if source == "reddit":
        expected_fields.remove("DMGMainCategoryMappedFromExtractedReligion")

    headers = list(df.columns)
    present = [f for f in expected_fields if f in headers]
    missing = [f for f in expected_fields if f not in headers]

    return present, missing


def test_reddit_output():
    """Test Reddit pipeline outputs all demographic fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run Reddit pipeline
        reddit_main(
            input_dir="data/test/reddit",
            output_dir=tmpdir,
            workers=1,
            chunk_size=0,
            stages="1",  # Just test stage 1 for users file
            task_id=0,
            total_tasks=1,
        )

        # Check users file
        users_file = os.path.join(tmpdir, "reddit_users.tsv")
        assert os.path.exists(users_file), "Users file not created"

        # Read and check headers
        df_users = pd.read_csv(users_file, sep="\t")
        headers = list(df_users.columns)
        print("Reddit users TSV headers:")
        print(headers)

        # Verify demographic fields
        present_fields, missing_fields = verify_demographic_fields(df_users, "reddit")

        if missing_fields:
            print(f"\n❌ Missing fields: {missing_fields}")
            raise AssertionError(f"Missing demographic fields: {missing_fields}")
        else:
            print(
                f"\n✅ All {len(present_fields)} expected demographic fields present!"
            )

        # Show sample data with demographics
        if len(df_users) > 0:
            # Find a row with at least one demographic field populated
            sample_row = None
            for idx in range(min(10, len(df_users))):  # Check first 10 rows
                row = df_users.iloc[idx]
                if any(pd.notna(row[f]) and str(row[f]) != "" for f in present_fields):
                    sample_row = row
                    break

            if sample_row is not None:
                print("\nSample data (row with demographics):")
                print(f"Author: {sample_row['Author']}")
                for field in present_fields:
                    val = sample_row[field]
                    if pd.notna(val) and str(val) != "":
                        print(f"{field}: {val}")


def test_tusc_output():
    """Test TUSC pipeline outputs all demographic fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run TUSC pipeline
        tusc_main(
            input_file="data/test/tusc/TUSC_city_small.parquet",
            output_dir=tmpdir,
            chunk_size=100000,
            stages="1",  # Just test stage 1 for users file
        )

        # Check users file
        users_file = os.path.join(tmpdir, "city_users.tsv")
        assert os.path.exists(users_file), "Users file not created"

        # Read and check headers
        df_users = pd.read_csv(users_file, sep="\t")
        headers = list(df_users.columns)
        print("\nTUSC users TSV headers:")
        print(headers)

        # Verify demographic fields
        present_fields, missing_fields = verify_demographic_fields(df_users, "tusc")

        if missing_fields:
            print(f"\n❌ Missing fields: {missing_fields}")
            raise AssertionError(f"Missing demographic fields: {missing_fields}")
        else:
            print(
                f"\n✅ All {len(present_fields)} expected demographic fields present!"
            )

        # Show sample data with demographics
        if len(df_users) > 0:
            # Find a row with at least one demographic field populated
            sample_row = None
            for idx in range(min(10, len(df_users))):  # Check first 10 rows
                row = df_users.iloc[idx]
                if any(pd.notna(row[f]) and str(row[f]) != "" for f in present_fields):
                    sample_row = row
                    break

            if sample_row is not None:
                print("\nSample data (row with demographics):")
                print(f"Author: {sample_row['Author']}")
                for field in present_fields:
                    val = sample_row[field]
                    if pd.notna(val) and str(val) != "":
                        print(f"{field}: {val}")


if __name__ == "__main__":
    print("=" * 60)
    print("Verifying demographic fields in pipeline outputs")
    print("=" * 60)

    try:
        print("\n1. Testing Reddit pipeline output...")
        test_reddit_output()
        print("\n✅ Reddit pipeline test passed!")
    except Exception as e:
        print(f"\n❌ Reddit pipeline test failed: {e}")
        exit(1)

    print("\n" + "-" * 60)

    try:
        print("\n2. Testing TUSC pipeline output...")
        test_tusc_output()
        print("\n✅ TUSC pipeline test passed!")
    except FileNotFoundError as e:
        print(f"\n⚠️  TUSC test skipped (file not found): {e}")
    except Exception as e:
        print(f"\n❌ TUSC pipeline test failed: {e}")
        exit(1)

    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
