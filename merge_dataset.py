#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from multiprocessing.pool import ThreadPool
from dask.dataframe.utils import make_meta
import multiprocessing as mp

# --- LOW-RAM CONFIG (edit as needed) ---
BASE_DIR = Path("/Users/jp/abcde_v1")
FILES = [
    # "ai-gen/pippa_data_features.tsv",
    # "ai-gen/raid_data_features.tsv",
    # "ai-gen/apt-paraphrase-dataset-gpt-3_features.tsv",
    # "ai-gen/m4_data_features.tsv",
    # "ai-gen/tinystories_data_features.tsv",
    "blogs/tiergroup-10/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-11/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-9/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-7/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-6/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-8/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-13/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-12/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-3/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-4/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-5/spinner_blog_posts_features.tsv",
    "blogs/tiergroup-2/spinner_blog_posts_features.tsv",
    "reddit/reddit_users_posts.tsv",
    "books/googlebooks-eng-fiction-top1M-5gram.tsv",
    "tusc/country_user_posts.tsv",
    "tusc/city_user_posts.tsv",
    "ai-gen/luar_lwd_data_features.tsv",
    "ai-gen/mage_data_features.tsv",
    "ai-gen/hh-rlhf_data_features.tsv",
    "ai-gen/prism_data_features.tsv",
    "ai-gen/anthropic_persuasiveness_data_features.tsv",
    "ai-gen/lmsys_data_features.tsv",
    "ai-gen/wildchat_data_features.tsv",
]

# Sources we might extract year/month from (do NOT include derived "Year"/"Month")
DATETIME_CANDIDATES = ("PostCreatedAt", "pubDate", "timestamp")
EPOCH_CANDIDATES = ("PostCreatedUtc",)
YEAR_SRC = ("PostYear", "year")  # exclude "Year" (derived)
MONTH_SRC = ("PostMonth", "month")  # exclude "Month" (derived)

# Performance knobs (small chunks + low concurrency = low RAM)
BLOCKSIZE = os.environ.get("MERGE_BLOCKSIZE", "64MB")  # e.g. 32MB, 64MB, 128MB
N_WORKERS = int(
    os.environ.get("MERGE_NWORKERS", mp.cpu_count())
)  # set to 1 if still tight

# Outputs
OUT_DIR_PARQUET = Path("merged_parquet")

# --- coercion helpers (no regex dependency for strings) ---
BPM_STRING_COLS = {"MyBPM", "YourBPM", "HisBPM", "HerBPM", "TheirBPM"}


def _to_bool_series(s: pd.Series) -> pd.Series:
    # normalize to pandas string dtype so .str works consistently
    s = s.astype("string")
    v = s.str.strip().str.lower()
    # accept 0/1/true/false only; everything else -> <NA>
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    is_booly = v.isin(["0", "1", "true", "false"]) | v.isna()
    if is_booly.any():
        mapped = v.map({"1": True, "0": False, "true": True, "false": False})
        out[is_booly] = mapped.astype("boolean")
    return out


def _to_numeric(s: pd.Series, kind: str) -> pd.Series:
    # kind in {"int32","int64","float32","float64"}
    num = pd.to_numeric(s, errors="coerce")
    if kind == "int32":
        return num.astype("Int32")
    if kind == "int64":
        return num.astype("Int64")
    if kind == "float32":
        return num.astype("Float32")
    return num.astype("Float64")


def _target_dtype_for(col: str) -> str:
    # desired output dtypes by column name pattern
    if col == "Year":
        return "Int16"
    if col == "Month":
        return "Int8"
    if col == "Dataset":
        return "string"
    if col == "match_count" or col == "MatchCount":
        return "Int32"
    if col == "HasBPM":
        return "boolean"
    if col in BPM_STRING_COLS:
        return "string"
    if ("Has" in col) and ("BPM" not in col):
        return "boolean"
    if "Count" in col:
        return "Int32"
    if "Avg" in col:
        return "Float32"
    return "string"


def _build_meta_coerced(cols) -> pd.DataFrame:
    dtypes = {c: _target_dtype_for(c) for c in cols}
    return make_meta(dtypes)


def _coerce_feature_types(pdf: pd.DataFrame) -> pd.DataFrame:
    cols = pdf.columns

    # booleans
    bool_cols = [c for c in cols if (c == "HasBPM") or ("Has" in c and "BPM" not in c)]
    for c in bool_cols:
        pdf[c] = _to_bool_series(pdf[c])

    # floats
    for c in [c for c in cols if "Avg" in c]:
        pdf[c] = _to_numeric(pdf[c], "float32")  # or "float64"

    # ints
    if "WordCount" in cols:
        pdf["WordCount"] = _to_numeric(pdf["WordCount"], "int32")

    # explicit match_count coercion (books) if present under any common casing
    for mc in ("match_count", "MatchCount"):
        if mc in cols:
            pdf[mc] = _to_numeric(pdf[mc], "int32")

    # any feature columns containing "Count" should be integer
    for c in [c for c in cols if ("Count" in c) and (c not in bool_cols)]:
        pdf[c] = _to_numeric(pdf[c], "int32")

    # BPM string exceptions (force string consistently)
    for c in BPM_STRING_COLS & set(cols):
        pdf[c] = pdf[c].astype("string")

    return pdf


def _debug_scan_file(ddf, path: Path, label: str) -> None:
    """Force-read each partition so we can pinpoint which file/partition fails."""
    nparts = ddf.npartitions
    for i in range(nparts):
        msg = f"[SCAN] {label} | {path} | partition {i+1}/{nparts}"
        print(msg, flush=True)
        try:
            # Count rows to force a full parse of the partition without materializing it.
            ddf.get_partition(i).map_partitions(lambda df: len(df)).sum().compute()
        except Exception as e:
            print(
                f"[FAIL] {label} | {path} | partition {i+1}/{nparts} -> {type(e).__name__}: {e}",
                flush=True,
            )
            raise


def _label(path: Path) -> str:
    parts = [x for x in path.parts if x not in (".", "")]
    return "/".join(parts[-3:]) if len(parts) >= 3 else str(path)


def _dataset_type_from_path(path: Path) -> str:
    """Map a file path to a coarse dataset type for export labeling."""
    top = ""
    try:
        top = path.relative_to(BASE_DIR).parts[0]
    except Exception:
        # Fallback: probe path components case-insensitively
        lower_parts = [p.lower() for p in path.parts]
        for key in ("ai-gen", "reddit", "tusc", "blogs", "books"):
            if key in lower_parts:
                top = key
                break

    mapping = {
        "ai-gen": "AI-Generated",
        "reddit": "Reddit",
        "tusc": "Twitter",
        "blogs": "Blogs",
        "books": "Books",
    }
    return mapping.get(top, "Unknown")


def _add_year_month(pdf: pd.DataFrame) -> pd.DataFrame:
    """Per-partition Year/Month derivation (NA-safe, low-RAM dtypes)."""
    y = pd.Series(pd.NA, index=pdf.index, dtype="Int16")  # smaller than Int64
    m = pd.Series(pd.NA, index=pdf.index, dtype="Int8")

    # numeric year/month from source columns
    for c in YEAR_SRC:
        if c in pdf.columns:
            s = pd.to_numeric(pdf[c], errors="coerce").astype("Int16")
            if not s.isna().all():
                y = s
                break
    for c in MONTH_SRC:
        if c in pdf.columns:
            s = pd.to_numeric(pdf[c], errors="coerce")
            s = s.where((s >= 1) & (s <= 12)).astype("Int8")
            if not s.isna().all():
                m = s
                break

    # datetime-like text columns
    if y.isna().all() or m.isna().all():
        for c in DATETIME_CANDIDATES:
            if c in pdf.columns:
                try:
                    dt = pd.to_datetime(
                        pdf[c], errors="coerce", utc=True, format="mixed"
                    )
                except TypeError:
                    dt = pd.to_datetime(pdf[c], errors="coerce", utc=True)
                if y.isna().all():
                    yy = dt.dt.year.astype("Int16")
                    if not yy.isna().all():
                        y = yy
                if m.isna().all():
                    mm = dt.dt.month.astype("Int8")
                    if not mm.isna().all():
                        m = mm
                if not y.isna().all() and not m.isna().all():
                    break

    # epoch seconds/ms heuristic
    if y.isna().all() or m.isna().all():
        for c in EPOCH_CANDIDATES:
            if c in pdf.columns:
                x = pd.to_numeric(pdf[c], errors="coerce")
                dt = pd.to_datetime(
                    x, unit="s", origin="unix", errors="coerce", utc=True
                )
                frac_nat = dt.isna().astype("float64").mean()
                frac_big = x.gt(10**12).astype("float64").mean()
                if (float(frac_nat) if pd.notna(frac_nat) else 0.0) > 0.5 and (
                    float(frac_big) if pd.notna(frac_big) else 0.0
                ) > 0.5:
                    dt = pd.to_datetime(
                        x, unit="ms", origin="unix", errors="coerce", utc=True
                    )
                if y.isna().all():
                    yy = dt.dt.year.astype("Int16")
                    if not yy.isna().all():
                        y = yy
                if m.isna().all():
                    mm = dt.dt.month.astype("Int8")
                    if not mm.isna().all():
                        m = mm
                if not y.isna().all() and not m.isna().all():
                    break

    pdf["Year"] = y
    pdf["Month"] = m
    return pdf


def _stitch_parts_to_single(parts_dir: Path, out_path: Path) -> None:
    part_files = sorted(parts_dir.glob("part-*.tsv"))
    if not part_files:
        raise SystemExit(f"No part files found in {parts_dir}")
    with out_path.open("wb") as w:
        # header from the first file
        with part_files[0].open("rb") as r0:
            w.write(r0.read())
        # append others without their headers
        for pf in part_files[1:]:
            with pf.open("rb") as r:
                first = True
                for line in r:
                    if first:
                        first = False
                        continue
                    w.write(line)


def main():
    # Use a tiny thread pool to limit concurrent partitions in memory.
    # Threads avoid per-process memory duplication on macOS.
    print(f"Using {N_WORKERS} workers.")
    pool = ThreadPool(N_WORKERS)
    dask.config.set(scheduler="threads", pool=pool)

    # --- discover files & headers quickly (header-only read) ---
    paths = [BASE_DIR / f for f in FILES if (BASE_DIR / f).is_file()]
    print(f"Found {len(paths)} input files.")
    if not paths:
        raise SystemExit("No input files found.")

    header_map = {}
    for p in paths:
        cols = pd.read_csv(p, sep="\t", nrows=0, dtype=str).columns
        header_map[p] = set(cols)

    common = set.intersection(*header_map.values()) if header_map else set()
    if not common:
        raise SystemExit("No common columns across the provided files.")

    # Keep all DMG-prefixed demographic columns even if they are not common to all datasets
    dmg_union = set()
    for cols in header_map.values():
        dmg_union.update(
            {c for c in cols if isinstance(c, str) and c.startswith("DMG")}
        )

    feature_cols = sorted(common | dmg_union)
    print(f"Found {len(feature_cols)} feature columns (common + DMG-prefixed).")

    # columns we might need to read from source files
    needed_for_ym_src = (
        set(DATETIME_CANDIDATES)
        | set(EPOCH_CANDIDATES)
        | set(YEAR_SRC)
        | set(MONTH_SRC)
    )

    dfs = []
    for p in paths:
        label = _label(p)
        header = header_map[p]
        # include optional match_count column if present in source (books)
        optional_cols = {"match_count", "MatchCount"}
        # always include any DMG-prefixed columns present in this file
        per_file_usecols = sorted(
            (header & (common | needed_for_ym_src | optional_cols | dmg_union))
        )
        if not per_file_usecols:
            raise SystemExit(f"No usable columns to read from {p}")

        # Forgiving parser: skip malformed rows; small blocks for low RAM
        read_kwargs = dict(
            sep="\t",
            dtype=str,
            assume_missing=True,
            blocksize=BLOCKSIZE,  # keeps partitions small
            usecols=per_file_usecols,
            on_bad_lines="skip",  # drop ragged/malformed lines
            sample=256000,  # limit dtype sample bytes (we force dtype=str anyway)
        )

        try:
            ddf = dd.read_csv(p.as_posix(), **read_kwargs)
        except Exception as e:
            # Fallback to Python engine (no block splitting) only if needed
            print(f"[WARN] C engine failed on {p}: {e}. Retrying with engine='python'.")
            rk2 = dict(read_kwargs)
            rk2.pop("blocksize", None)  # python engine can't split by block
            ddf = dd.read_csv(p.as_posix(), engine="python", **rk2)

        if os.environ.get("MERGE_DEBUG_SCAN", "1") == "1":
            _debug_scan_file(ddf, p, label)

        # Provide meta so Dask knows new columns + dtypes (use small extension dtypes)
        meta = ddf._meta.assign(
            Year=pd.Series(dtype="Int16"), Month=pd.Series(dtype="Int8")
        )

        # 1) derive Year/Month
        ddf = ddf.map_partitions(_add_year_month, meta=meta)

        # Ensure this dataframe has all DMG columns (add missing as NA so they persist in output)
        missing_dmg = sorted(dmg_union - set(ddf.columns))
        if missing_dmg:
            add_missing = {c: pd.NA for c in missing_dmg}
            ddf = ddf.assign(**add_missing)

        # 2) build meta for the *post-coercion* schema and coerce types
        post_cols = list(ddf._meta.columns)  # includes Year/Month + features (+ DMG)
        meta_coerced = _build_meta_coerced(post_cols)
        ddf = ddf.map_partitions(_coerce_feature_types, meta=meta_coerced)

        # Normalize match count column name if present (output should be 'MatchCount')
        has_mc_lower = "match_count" in header
        has_mc_camel = "MatchCount" in header
        if has_mc_lower and not has_mc_camel:
            ddf = ddf.rename(columns={"match_count": "MatchCount"})

        # Ensure match_count policy: 1 for non-books, source value for books
        ds_type = _dataset_type_from_path(p)
        if ds_type != "Books":
            ddf = ddf.assign(MatchCount=1)

        # 3) add Dataset label, Dataset Type, and select final columns
        ddf = ddf.assign(Dataset=label, **{"Dataset Type": ds_type})[
            feature_cols + ["Year", "Month", "MatchCount", "Dataset", "Dataset Type"]
        ]
        # Enforce dtype for match_count post-assignment
        ddf = ddf.assign(MatchCount=ddf["MatchCount"].astype("Int32"))
        # Category = smaller memory when materialized (pyarrow writes dictionary)
        ddf = ddf.assign(
            Dataset=ddf["Dataset"].astype("category"),
            **{"Dataset Type": ddf["Dataset Type"].astype("category")},
        )
        dfs.append(ddf)

    # Concatenate lazily; no persist() to avoid caching the whole graph in RAM.
    merged = dd.concat(dfs, interleave_partitions=True)

    ProgressBar().register()

    # --- Writes (streaming via scheduler; no in-memory persist) ---
    OUT_DIR_PARQUET.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(
        OUT_DIR_PARQUET.as_posix(),
        write_index=False,
        compression="snappy",
        engine="pyarrow",
        write_metadata_file=False,  # avoid large global metadata gather
        # partition_on=["Year", "Month"],  # uncomment if you want directory partitioning
    )
    print(f"[DONE] Wrote parquet directory: {OUT_DIR_PARQUET}")


if __name__ == "__main__":
    main()
