#!/usr/bin/env python
from __future__ import annotations

import argparse
import heapq
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def log_with_timestamp(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def iter_parts(map_dir: Path) -> Iterable[Path]:
    for p in sorted(map_dir.glob("map_task*_part*.tsv")):
        if p.is_file():
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduce step: aggregate partial counts and emit global top-K 5-grams by match_count.")
    ap.add_argument("--map_dir", required=True, help="Directory with map outputs (map_task*_part*.tsv)")
    ap.add_argument("--output_path", required=True, help="Path to write sorted global counts TSV")
    ap.add_argument("--top_k", type=int, default=1_000_000, help="Keep only top-K by total match_count")
    ap.add_argument("--memory_cap_unique", type=int, default=50_000_000, help="Approx cap for unique n-grams in memory before partial heap flush")
    args = ap.parse_args()

    map_dir = Path(args.map_dir)
    parts = list(iter_parts(map_dir))
    if not parts:
        log_with_timestamp("No map parts found")
        return

    # We'll aggregate in chunks to control memory:
    # - Read batches of part files, aggregate into counts dict
    # - When unique size grows too large, push into a heap limited to top_k and clear
    heap: List[Tuple[int, str]] = []  # (count, ngram)

    def merge_counts_into_heap(counts: Dict[str, int]):
        nonlocal heap
        for ngram, cnt in counts.items():
            if cnt <= 0:
                continue
            if len(heap) < args.top_k:
                heapq.heappush(heap, (cnt, ngram))
            else:
                if cnt > heap[0][0]:
                    heapq.heapreplace(heap, (cnt, ngram))

    counts: Dict[str, int] = defaultdict(int)
    files_processed = 0

    for part in parts:
        with open(part, "r", encoding="utf-8") as r:
            for line in r:
                try:
                    ngram, cnt_s = line.rstrip("\n").split("\t")
                    cnt = int(cnt_s)
                except ValueError:
                    continue
                counts[ngram] += cnt
        files_processed += 1

        if len(counts) >= args.memory_cap_unique:
            log_with_timestamp(
                f"Merging {len(counts):,} uniques into heap (current heap {len(heap):,})"
            )
            merge_counts_into_heap(counts)
            counts.clear()

    # Final merge
    if counts:
        merge_counts_into_heap(counts)
        counts.clear()

    # Drain heap to sorted list descending by count
    top = heapq.nlargest(len(heap), heap)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        w.write("ngram\tmatch_count\n")
        for cnt, ngram in top:
            w.write(f"{ngram}\t{cnt}\n")

    log_with_timestamp(
        f"Reduced {files_processed} parts → wrote {len(top):,} rows to {out_path}"
    )


if __name__ == "__main__":
    main()


