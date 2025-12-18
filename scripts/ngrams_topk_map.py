#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


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


def iter_assigned_files(files: List[Path], task_id: int, total_tasks: int) -> Iterable[Path]:
    if total_tasks <= 1:
        for f in files:
            yield f
        return
    for idx, f in enumerate(files):
        if idx % total_tasks == task_id:
            yield f


def parse_line(line: str):
    # ngram TAB year TAB match_count TAB book_count
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        return None
    ngram, year_s, match_s, book_s = parts
    try:
        year = int(year_s)
        match_count = int(match_s)
        _ = int(book_s)
    except ValueError:
        return None
    cleaned = strip_pos_tags(ngram)
    # Keep exactly 5 tokens after cleaning
    if len(cleaned.split()) != 5:
        return None
    return cleaned, match_count


def flush_counts(counts: Dict[str, int], out_dir: Path, task_id: int, part_idx: int) -> int:
    out_path = out_dir / f"map_task{task_id}_part{part_idx}.tsv"
    with open(out_path, "w", encoding="utf-8") as w:
        for ngram, cnt in counts.items():
            w.write(f"{ngram}\t{cnt}\n")
    log_with_timestamp(f"Wrote partial counts: {out_path} ({len(counts)} unique)")
    return part_idx + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Map step: compute partial counts for 5-grams (cleaned) with streaming flushes.")
    ap.add_argument("--input_dir", required=True, help="Directory containing ngram files (tsv)")
    ap.add_argument("--pattern", default="*5gram*", help="Glob pattern to match files")
    ap.add_argument("--output_dir", required=True, help="Directory to write partial map outputs")
    ap.add_argument("--task_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--total_tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--flush_every", type=int, default=5_000_000, help="Flush partial counts when unique n-grams reach this size")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        log_with_timestamp("No input files matched")
        sys.exit(1)

    assigned = list(iter_assigned_files(files, args.task_id, args.total_tasks))
    if not assigned:
        log_with_timestamp(f"Task {args.task_id}/{args.total_tasks}: no files assigned")
        return

    counts: Dict[str, int] = defaultdict(int)
    part_idx = 0
    total_lines = 0
    total_kept = 0

    for fp in assigned:
        log_with_timestamp(f"Task {args.task_id}: processing {fp.name}")
        with open(fp, "r", encoding="utf-8") as r:
            for line in r:
                total_lines += 1
                parsed = parse_line(line)
                if not parsed:
                    continue
                ngram, mc = parsed
                counts[ngram] += mc
                total_kept += 1
                if len(counts) >= args.flush_every:
                    part_idx = flush_counts(counts, out_dir, args.task_id, part_idx)
                    counts.clear()

    if counts:
        part_idx = flush_counts(counts, out_dir, args.task_id, part_idx)

    log_with_timestamp(
        f"Task {args.task_id} done. Read {total_lines} lines; kept {total_kept}. Parts written: {part_idx}"
    )


if __name__ == "__main__":
    main()


