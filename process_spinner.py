#!/usr/bin/env python3
"""
Process spinner dataset: extract blog posts from XML files and compute linguistic features.

Key features
------------
* Robust sanitisation – no raw new‑lines/tabs or stray HTML entities in the TSV.
* Fast, dependency‑free language gate – keeps **only posts whose main body is
  *probably* English**.
* Tab‑safe UTF‑8 output with explicit escaping.
"""

import argparse
import csv
import glob
import html
import os
import re
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional
from langdetect import detect, DetectorFactory

import pandas as pd

from helpers import apply_linguistic_features, print_banner


_entity_re = re.compile(r"&#x?[0-9a-fA-F]+;", flags=re.I)
_whitespace = re.compile(r"\s+")


def _sanitize(value: Any, *, strip_tags: bool = False) -> Any:
    if not isinstance(value, str):
        return value
    text = html.unescape(html.unescape(value))
    text = _entity_re.sub("", text)
    if strip_tags:
        text = re.sub(r"<[^>]+>", "", text)
    text = _whitespace.sub(" ", text)
    return text.strip()

_SANITISE_FIELD = partial(_sanitize, strip_tags=False)
_SANITISE_CONTENT = partial(_sanitize, strip_tags=True)

def _is_probably_english(text: str) -> bool:
    try:
        return detect(text[:4000]) == "en"    # 4 KB is plenty and saves time
    except Exception:
        return False


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception as exc:
        log(f"Error reading {path}: {exc}")
        return None


def find_field(content: str, tag: str) -> Optional[str]:
    for pat in [fr"<{tag}>(.*?)</{tag}>", fr"<[^:>]*:{tag}>(.*?)</[^:>]*:{tag}>", fr"<{tag}[^>]*>(.*?)</{tag}>"]:
        m = re.search(pat, content, re.DOTALL)
        if m:
            return html.unescape(m.group(1).strip())
    return None

# quick helpers
find_descriptions = lambda xml: re.findall(r"<content:encoded>(.*?)</content:encoded>|<description>(.*?)</description>", xml, re.DOTALL)
find_categories   = lambda xml: re.findall(r"<category>(.*?)</category>", xml, re.DOTALL)

def process_xml(path: str) -> List[Dict[str, Any]]:
    xml = read_file(path)
    if not xml:
        return []

    out: List[Dict[str, Any]] = []
    for item in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL):
        item_xml = f"<item>{item}</item>"
        desc_matches = find_descriptions(item_xml)
        if not desc_matches:
            continue
        raw_desc = next(d for tup in desc_matches for d in tup if d)
        clean_desc = _SANITISE_CONTENT(raw_desc)
        if len(clean_desc) < 250 or not _is_probably_english(clean_desc):
            continue

        rec = {
            "file_path": path,
            "title": find_field(item_xml, "title") or "",
            "link": find_field(item_xml, "link") or "",
            "guid": find_field(item_xml, "guid") or "",
            "pubDate": find_field(item_xml, "pubDate") or "",
            "description_raw": raw_desc,
            "description": clean_desc,
            "categories": "|".join(find_categories(item_xml)),
        }
        try:
            rec.update(apply_linguistic_features(clean_desc))
        except Exception as exc:
            log(f"skip features: {exc}")
            continue
        out.append({k: (_SANITISE_CONTENT(v) if k.startswith("description") else _SANITISE_FIELD(v)) for k, v in rec.items()})
    return out


def main(inp: str, out_dir: str, max_files: Optional[int]):
    print_banner()
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(inp, "**", "*.xml"), recursive=True))
    if max_files:
        files = files[:max_files]
        log(f"Debug: only {max_files} files")

    rows: List[Dict[str, Any]] = []
    for idx, f in enumerate(files, 1):
        log(f"[{idx}/{len(files)}] {f}")
        rows.extend(process_xml(f))

    if not rows:
        log("No English posts – nothing written")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "spinner_blog_posts_features.tsv")
    log(f"Writing {len(df)} rows → {out_path}")
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    assert not df.apply(lambda c: c.astype(str).str.contains(r"[\n\r\t]").any()).any(), "raw newlines or tabs slipped in"
    log("Done ✨")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_files", type=int)
    a = p.parse_args()
    main(a.input_dir, a.output_dir, a.max_files)