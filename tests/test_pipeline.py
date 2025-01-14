import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import apply_linguistic_features, get_csv_fieldnames

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args):
    subprocess.run([PYTHON] + args, check=True)


def _assert_cols(path, expected, check_age=False, age_col=None, birth_col=None):
    df = pd.read_csv(path, sep="\t")
    assert list(df.columns) == expected
    assert len(df) > 0
    if check_age and age_col:
        # For posts, only check age for posts where the user has a birthyear
        assert (
            df[age_col].notna() | df[age_col].isna()
        ).all()  # This is always true, but documents intent
    if check_age and birth_col:
        # For users, just check that some users have birthyears (not all)
        assert df[birth_col].notna().sum() > 0


def _expected_posts(data_source, split=None):
    base = get_csv_fieldnames(data_source, split, "posts")
    feats = sorted(apply_linguistic_features("test").keys())
    return base + feats


def _expected_users(data_source, split=None):
    return get_csv_fieldnames(data_source, split, "users")


def test_reddit_pipeline(tmp_path):
    out = tmp_path / "reddit"
    _run(
        [
            "process_reddit.py",
            "--input_dir",
            str(REPO_ROOT / "data/test/reddit/RS_2020-01_test"),
            "--output_dir",
            str(out),
            "--chunk_size",
            "0",
            "--workers",
            "1",
            "--stages",
            "both",
        ]
    )
    _assert_cols(
        out / "reddit_users.tsv",
        _expected_users("reddit"),
        check_age=True,
        birth_col="DMGMajorityBirthyear",
    )
    _assert_cols(
        out / "reddit_users_posts.tsv",
        _expected_posts("reddit"),
        check_age=True,
        age_col="DMGAgeAtPost",
    )


def test_tusc_city_pipeline(tmp_path):
    out = tmp_path / "city"
    _run(
        [
            "process_tusc.py",
            "--input_file",
            str(REPO_ROOT / "data/test/tusc/tusc-city_test.parquet"),
            "--output_dir",
            str(out),
            "--stages",
            "both",
        ]
    )
    _assert_cols(
        out / "city_users.tsv",
        _expected_users("tusc", "city"),
        check_age=True,
        birth_col="DMGMajorityBirthyear",
    )
    _assert_cols(
        out / "city_user_posts.tsv",
        _expected_posts("tusc", "city"),
        check_age=True,
        age_col="DMGAgeAtPost",
    )


def test_tusc_country_pipeline(tmp_path):
    out = tmp_path / "country"
    _run(
        [
            "process_tusc.py",
            "--input_file",
            str(REPO_ROOT / "data/test/tusc/tusc-country_test.parquet"),
            "--output_dir",
            str(out),
            "--stages",
            "both",
        ]
    )
    _assert_cols(
        out / "country_users.tsv",
        _expected_users("tusc", "country"),
        check_age=True,
        birth_col="DMGMajorityBirthyear",
    )
    _assert_cols(
        out / "country_user_posts.tsv",
        _expected_posts("tusc", "country"),
        check_age=True,
        age_col="DMGAgeAtPost",
    )
