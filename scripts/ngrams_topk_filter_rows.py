#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def log_with_timestamp(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def strip_pos_tags(ngram: str) -> str:
    tokens = ngram.split()
    cleaned_tokens = []
    for token in tokens:
        if "_" in token:
            cleaned_tokens.append(token.rsplit("_", 1)[0])
        else:
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


def load_topk(path: Path) -> Set[str]:
    keep: Set[str] = set()
    with open(path, "r", encoding="utf-8") as r:
        header = r.readline()
        for line in r:
            try:
                ngram, cnt = line.rstrip("\n").split("\t")
                _ = int(cnt)
            except ValueError:
                continue
            keep.add(ngram)
    return keep


def _read_header(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as r:
        header_line = r.readline().rstrip("\n")
        return header_line.split("\t") if header_line else []


def _infer_union_header(files: List[Path]) -> Tuple[List[str], Dict[Path, List[str]]]:
    # Start with canonical first four columns in expected order if present
    base_order = ["ngram", "year", "match_count", "book_count"]
    union_cols: List[str] = []
    per_file_headers: Dict[Path, List[str]] = {}
    # Add base columns first (they will be de-duped when scanning real headers)
    for col in base_order:
        if col not in union_cols:
            union_cols.append(col)
    for fp in files:
        hdr = _read_header(fp)
        per_file_headers[fp] = hdr
        for col in hdr:
            if col and col not in union_cols:
                union_cols.append(col)
    return union_cols, per_file_headers


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter annotated 5-gram TSVs to rows whose cleaned ngram is in the given top-K list. Preserves all feature columns by inferring them from input headers.")
    ap.add_argument("--input_dir", required=True, help="Directory with annotated ngram files (tsv)")
    ap.add_argument("--pattern", default="*5gram*", help="Glob pattern for files")
    ap.add_argument("--topk_path", required=True, help="Path to top-K TSV produced by reducer")
    ap.add_argument("--output_path", required=True, help="Path to write filtered rows TSV (all inferred columns)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    topk = load_topk(Path(args.topk_path))
    if not topk:
        log_with_timestamp("Top-K list is empty; nothing to filter")
        return

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        log_with_timestamp(f"No files matched pattern {args.pattern} under {in_dir}")
        return

    # Infer union header across files and capture per-file headers
    union_cols, per_file_headers = _infer_union_header(files)
    # Quick sanity: ensure required core columns exist in union
    required = {"ngram", "year", "match_count", "book_count"}
    if not required.issubset(set(union_cols)):
        missing = required.difference(set(union_cols))
        log_with_timestamp(f"WARN: Missing expected columns in inputs: {sorted(missing)}")

    total_rows = 0
    kept_rows = 0

    with open(out_path, "w", encoding="utf-8") as w:
        # Write union header
        w.write("\t".join(union_cols) + "\n")
        for fp in files:
            hdr = per_file_headers.get(fp, [])
            if not hdr:
                # Skip empty or headerless files
                continue
            col_to_idx: Dict[str, int] = {c: i for i, c in enumerate(hdr)}
            # Ensure ngram column exists in this file
            if "ngram" not in col_to_idx:
                continue
            with open(fp, "r", encoding="utf-8") as r:
                # Skip header line
                _ = r.readline()
                for line in r:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    total_rows += 1
                    raw_ngram = parts[col_to_idx["ngram"]] if col_to_idx["ngram"] < len(parts) else ""
                    cleaned_ngram = strip_pos_tags(raw_ngram)
                    # Keep only 5-grams after cleaning
                    if len(cleaned_ngram.split()) != 5:
                        continue
                    if cleaned_ngram not in topk:
                        continue
                    # Build row aligned to union columns
                    row_vals: List[str] = []
                    for col in union_cols:
                        if col == "ngram":
                            row_vals.append(cleaned_ngram)
                            continue
                        idx = col_to_idx.get(col)
                        if idx is None or idx >= len(parts):
                            row_vals.append("")
                        else:
                            row_vals.append(parts[idx])
                    w.write("\t".join(row_vals) + "\n")
                    kept_rows += 1

    log_with_timestamp(
        f"Filtered rows written to {out_path}. Kept {kept_rows:,} of {total_rows:,} lines."
    )


if __name__ == "__main__":
    main()


