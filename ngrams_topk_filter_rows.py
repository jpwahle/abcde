#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter raw 5-gram files to only rows whose cleaned ngram is in the given top-K list. Writes a compact TSV for merging.")
    ap.add_argument("--input_dir", required=True, help="Directory with raw ngram files (tsv)")
    ap.add_argument("--pattern", default="*5gram*", help="Glob pattern for files")
    ap.add_argument("--topk_path", required=True, help="Path to top-K TSV produced by reducer")
    ap.add_argument("--output_path", required=True, help="Path to write filtered rows TSV (ngram, year, match_count, book_count)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    topk = load_topk(Path(args.topk_path))
    if not topk:
        log_with_timestamp("Top-K list is empty; nothing to filter")
        return

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0

    with open(out_path, "w", encoding="utf-8") as w:
        w.write("ngram\tyear\tmatch_count\tbook_count\n")
        for fp in sorted(in_dir.glob(args.pattern)):
            with open(fp, "r", encoding="utf-8") as r:
                for line in r:
                    parts = line.rstrip("\n").split("\t")
                    # Accept annotated TSVs that may contain additional columns beyond the first four
                    if len(parts) < 4:
                        continue
                    ngram_raw, year_s, mc_s, bc_s = parts[:4]
                    total_rows += 1
                    ngram = strip_pos_tags(ngram_raw)
                    if len(ngram.split()) != 5:
                        continue
                    if ngram not in topk:
                        continue
                    w.write(f"{ngram}\t{year_s}\t{mc_s}\t{bc_s}\n")
                    kept_rows += 1

    log_with_timestamp(
        f"Filtered rows written to {out_path}. Kept {kept_rows:,} of {total_rows:,} lines."
    )


if __name__ == "__main__":
    main()


