import subprocess
import sys
from pathlib import Path
import pandas as pd
from helpers import get_csv_fieldnames, apply_linguistic_features

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args):
    subprocess.run([PYTHON] + args, check=True)


def _assert_cols(path, expected, check_age=False, age_col=None, birth_col=None):
    df = pd.read_csv(path, sep="\t")
    assert list(df.columns) == expected
    assert len(df) > 0
    if check_age and age_col:
        assert df[age_col].notna().all()
    if check_age and birth_col:
        assert df[birth_col].notna().all()


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
        out / "city_self_users.tsv",
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
        out / "country_self_users.tsv",
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
