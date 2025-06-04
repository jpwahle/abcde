#!/usr/bin/env python3
"""
Shared helper functions for Reddit and TUSC data processing.
Contains functions for self-identification detection, linguistic feature computation,
flattening results, and simple I/O utilities.
"""
from __future__ import annotations

import os
import re
import json
import csv
import random
from typing import Any, Callable, Dict, Generator, List, Optional, Pattern, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd

# -------------------- #
# Self-Identification Detection
# -------------------- #

class SelfIdentificationDetector:
    """Detect self-identification statements (age, etc.) inside free text.

    The detector is data-source agnostic – it expects plain strings and can be
    reused for Reddit, Twitter, blogs, or any other textual resource.
    """
    def __init__(self) -> None:
        self.patterns: Dict[str, List[Pattern[str]]] = {
            "age": [
                re.compile(r"\bI(?:\s+am|'m)\s+([1-9]\d?)\s+years?\s+old\b", re.I),
                re.compile(
                    r"\bI(?:\s+am|'m)\s+([1-9]\d?)(?=\s*(?:years?(?:\s+old|-old)?|yo|yrs?)?\b)"
                    r"(?!\s*[%$°#@&*+=<>()[\]{}|\\~`^_])",
                    re.I | re.VERBOSE
                ),
                re.compile(
                    r"\bI(?:\s+was|\s+am|'m)\s+born\s+in\s+(19\d{2}|20(?:0\d|1\d|2[0-4]|25))\b",
                    re.I,
                ),
                re.compile(
                    r"\bI\s+was\s+born\s+on\s+\d{1,2}\s+\w+\s+"
                    r"(19\d{2}|20(?:0\d|1\d|2[0-4]|25))\b",
                    re.I,
                ),
                re.compile(
                    r"\bI(?:\s*'m\s*turning|\s+turn(?:ed)?)\s+([1-9]\d?)(?=\s*[.!?;,]|\s*$)"
                    r"(?!\s*[%$°#@&*+=<>()[\]{}|\\~`^_])",
                    re.I,
                ),
            ],
        }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Return category → unique matched strings for self-identification."""
        if not isinstance(text, str) or not text:
            return {}
        text = text.strip()
        matches: Dict[str, List[str]] = {}
        for category, regs in self.patterns.items():
            cat_matches: List[str] = []
            for reg in regs:
                for m in reg.finditer(text):
                    cat_matches.append(m.group(1) if m.groups() else m.group(0))
            if cat_matches:
                uniq: List[str] = []
                for cm in cat_matches:
                    if cm not in uniq:
                        uniq.append(cm)
                matches[category] = uniq
        return matches


    def resolve_multiple_ages(
        self,
        age_matches: List[str],
        current_year: Optional[int] = None,
    ) -> Optional[Tuple[int, float]]:
        """Resolve multiple age extractions using clustering and confidence scoring."""
        if not age_matches:
            return None
        birth_year_candidates: List[Tuple[int, float]] = []
        for age_str in age_matches:
            try:
                age_val = int(age_str)
            except ValueError:
                continue
            if 1900 <= age_val <= current_year:
                birth_year, weight = age_val, 1.0
            elif 13 <= age_val <= 100:  # Filter out ages below 13
                birth_year, weight = current_year - age_val, 0.8
            else:
                continue
            birth_year_candidates.append((birth_year, weight))
        if not birth_year_candidates:
            return None
        clusters: Dict[int, List[Tuple[int, float]]] = {}
        for by, wt in birth_year_candidates:
            key = next((k for k in clusters if abs(by - k) <= 2), None)
            if key is None:
                key = by
                clusters[key] = []
            clusters[key].append((by, wt))
        best_cluster = None
        best_score = 0.0
        for center, members in clusters.items():
            total_weight = sum(w for _, w in members)
            score = total_weight + len(members) * 0.1
            if score > best_score:
                best_score = score
                best_cluster = members
        if not best_cluster:
            return None
        total_weight = sum(w for _, w in best_cluster)
        weighted_year = sum(by * w for by, w in best_cluster) / total_weight
        resolved_age = current_year - int(round(weighted_year))
        confidence = min(1.0, best_score / (len(age_matches) * 1.0))
        if not (13 <= resolved_age <= 100):  # Filter out resolved ages below 13
            return None
        return resolved_age, confidence


def detect_self_identification_in_entry(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, List[str]]:
    """Detect self-identification in a Reddit-style entry (title+body)."""
    title = entry.get("title", "") or ""
    body = entry.get("selftext", "") or ""
    combined = f"{title} {body}".strip()
    return detector.detect(combined)

def detect_self_identification_with_resolved_age(entry: Dict[str, Any], detector: "SelfIdentificationDetector") -> Dict[str, Any]:
    """Detect self identification with age resolution for multiple age extractions.
    
    Parameters
    ----------
    entry : Dict[str, Any]
        Reddit-style entry with title and body
    detector : SelfIdentificationDetector
        Detector instance
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with original matches plus resolved_age info:
        {
            "age": ["25", "1998"],  # original extractions
            "resolved_age": {"age": 27, "confidence": 0.9, "raw_matches": ["25", "1998"]}
        }
    """
    matches = detect_self_identification_in_entry(entry, detector)
    
    # If age matches found, resolve them
    if "age" in matches:
        # Extract post year for age resolution
        if "Year" in entry:  # TUSC data
            ref_year = int(entry.get("Year", ""))
        else:  # Reddit data
            ts = entry.get("created_utc") or entry.get("post", {}).get("created_utc")
            ref_year = datetime.utcfromtimestamp(int(ts)).year
        
        age_resolution = detector.resolve_multiple_ages(matches["age"], current_year=ref_year)
        if age_resolution is not None:
            resolved_age, confidence = age_resolution
            matches["resolved_age"] = {
                "age": resolved_age,
                "confidence": confidence,
                "raw_matches": matches["age"].copy()
            }
    
    return matches


# -------------------- #
# Lexicon Loading and Feature Computation
# -------------------- #

_DATA_DIR = Path("data")

def _safe_read(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

def _load_lexicon(
    filename: str,
    key_col: int = 0,
    value_col: int = 2,
    skip_header: bool = False,
    value_type: str = "int",
    key_transform: Callable[[str], str] = lambda x: x.lower(),
    value_transform: Callable[[str], Any] = lambda x: x,
    accumulate: bool = False,
) -> Dict[str, Any]:
    lines = _safe_read(_DATA_DIR / filename)
    if skip_header and lines:
        lines = lines[1:]
    result: Any
    if value_type in ("set", "list") and accumulate:
        from collections import defaultdict
        if value_type == "set":
            result = defaultdict(set)
        else:
            result = defaultdict(list)
    else:
        result = {}
    for line in lines:
        if not line or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) <= max(key_col, value_col):
            continue
        key = key_transform(parts[key_col])
        raw = value_transform(parts[value_col])
        if value_type == "int":
            val = int(raw)
        elif value_type == "float":
            val = float(raw)
        elif value_type == "str":
            val = str(raw)
        elif value_type == "set" and accumulate:
            if len(parts) > value_col + 1 and int(parts[value_col + 1]) == 1:
                result[key].add(raw)
            continue
        elif value_type == "list" and accumulate:
            result.setdefault(key, []).append(raw)
            continue
        else:
            val = raw
        result[key] = val
    return result

def _load_nrc_vad_lexicon() -> Dict[str, Dict[str, float]]:
    vad_dict: Dict[str, Dict[str, float]] = {}
    for line in _safe_read(_DATA_DIR / "NRC-VAD-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        w, v, a, d = line.split("\t")
        vad_dict[w.lower()] = {"valence": float(v), "arousal": float(a), "dominance": float(d)}
    return vad_dict

def _load_nrc_emotion_lexicon() -> Dict[str, set]:
    from collections import defaultdict
    em: Dict[str, set] = defaultdict(set)
    for line in _safe_read(_DATA_DIR / "NRC-Emotion-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        w, emo, flag = line.split("\t")
        if int(flag) == 1:
            em[w.lower()].add(emo)
    return em

def _load_nrc_worrywords_lexicon() -> Dict[str, int]:
    return _load_lexicon("NRC-WorryWords-Lexicon.txt", skip_header=True, value_type="int")

def _load_eng_tenses_lexicon() -> Dict[str, List[str]]:
    return _load_lexicon(
        "eng-word-tenses.txt", key_col=1, value_col=2, value_type="list", accumulate=True
    )

def _load_nrc_moraltrust_lexicon() -> Dict[str, int]:
    return _load_lexicon("NRC-MoralTrustworthy-Lexicon.txt", skip_header=True, value_type="int")

def _load_nrc_socialwarmth_lexicon() -> Dict[str, int]:
    return _load_lexicon("NRC-SocialWarmth-Lexicon.txt", skip_header=True, value_type="int")

def _load_nrc_warmth_lexicon() -> Dict[str, int]:
    return _load_lexicon("NRC-CombinedWarmth-Lexicon.txt", skip_header=True, value_type="int")

vad_dict = _load_nrc_vad_lexicon()
emotion_dict = _load_nrc_emotion_lexicon()
worry_dict = _load_nrc_worrywords_lexicon()
moraltrust_dict = _load_nrc_moraltrust_lexicon()
socialwarmth_dict = _load_nrc_socialwarmth_lexicon()
warmth_dict = _load_nrc_warmth_lexicon()
tense_dict = _load_eng_tenses_lexicon()
emotions = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]

def compute_vad_and_emotions(
    text: str,
    vad_dict: Dict[str, Dict[str, float]],
    emotion_dict: Dict[str, set],
    emotions: List[str],
    worry_dict: Dict[str, int],
    moraltrust_dict: Dict[str, int],
    socialwarmth_dict: Dict[str, int],
    warmth_dict: Dict[str, int],
) -> Dict[str, Any]:
    words = text.lower().split() if isinstance(text, str) else []
    vad_scores = {dim: [] for dim in ("valence", "arousal", "dominance")}
    high_flags = {dim: 0 for dim in vad_scores}
    low_flags = {dim: 0 for dim in vad_scores}
    high_counts = {dim: 0 for dim in vad_scores}
    low_counts = {dim: 0 for dim in vad_scores}
    emotion_counts = {emo: 0 for emo in emotions}
    emotion_flags = {emo: 0 for emo in emotions}
    sum_anx = count_anx = sum_calm = count_calm = 0
    has_anx = has_calm = has_hi_anx = has_hi_calm = 0
    hi_anx_count = hi_calm_count = 0
    sum_moral = count_moral = has_hi_moral = has_lo_moral = 0
    hi_moral_count = lo_moral_count = 0
    sum_soc = count_soc = has_hi_soc = has_lo_soc = 0
    hi_soc_count = lo_soc_count = 0
    sum_warm = count_warm = has_hi_warm = has_lo_warm = 0
    hi_warm_count = lo_warm_count = 0
    for w in words:
        if w in vad_dict:
            for dim in vad_scores:
                sc = vad_dict[w][dim]
                vad_scores[dim].append(sc)
                if sc > 0.67:
                    high_flags[dim] = 1; high_counts[dim] += 1
                if sc < 0.33:
                    low_flags[dim] = 1; low_counts[dim] += 1
        if w in emotion_dict:
            for emo in emotion_dict[w]:
                emotion_flags[emo] = 1; emotion_counts[emo] += 1
        if worry_dict and w in worry_dict:
            oc = worry_dict[w]
            if oc > 0:
                has_anx = 1; sum_anx += oc; count_anx += 1
                if oc == 3: has_hi_anx = 1; hi_anx_count += 1
            elif oc < 0:
                has_calm = 1; sum_calm += oc; count_calm += 1
                if oc == -3: has_hi_calm = 1; hi_calm_count += 1
        if moraltrust_dict and w in moraltrust_dict:
            oc = moraltrust_dict[w]; sum_moral += oc; count_moral += 1
            if oc == 3: has_hi_moral = 1; hi_moral_count += 1
            if oc == -3: has_lo_moral = 1; lo_moral_count += 1
        if socialwarmth_dict and w in socialwarmth_dict:
            oc = socialwarmth_dict[w]; sum_soc += oc; count_soc += 1
            if oc == 3: has_hi_soc = 1; hi_soc_count += 1
            if oc == -3: has_lo_soc = 1; lo_soc_count += 1
        if warmth_dict and w in warmth_dict:
            oc = warmth_dict[w]; sum_warm += oc; count_warm += 1
            if oc == 3: has_hi_warm = 1; hi_warm_count += 1
            if oc == -3: has_lo_warm = 1; lo_warm_count += 1
    avg_vad = {f"NRCAvg{dim.capitalize()}": sum(vals)/len(vals) if vals else 0 for dim, vals in vad_scores.items()}
    emo_cols = {f"NRCHas{emo.capitalize()}Word": flag for emo, flag in emotion_flags.items()}
    emo_cnt_cols = {f"NRCCount{emo.capitalize()}Words": cnt for emo, cnt in emotion_counts.items()}
    vad_thr = {f"NRCHasHigh{dim.capitalize()}Word": high_flags[dim] for dim in high_flags}
    vad_thr.update({f"NRCHasLow{dim.capitalize()}Word": low_flags[dim] for dim in low_flags})
    vad_cnt = {f"NRCCountHigh{dim.capitalize()}Words": high_counts[dim] for dim in high_counts}
    vad_cnt.update({f"NRCCountLow{dim.capitalize()}Words": low_counts[dim] for dim in low_counts})
    word_count = {"WordCount": len(words)}
    avg_anx = sum_anx/count_anx if count_anx else 0
    avg_calm = sum_calm/count_calm if count_calm else 0
    worry_cols = {
        "NRCHasAnxietyWord": has_anx, "NRCHasCalmnessWord": has_calm,
        "NRCAvgAnxiety": avg_anx, "NRCAvgCalmness": avg_calm,
        "NRCHasHighAnxietyWord": has_hi_anx, "NRCCountHighAnxietyWords": hi_anx_count,
        "NRCHasHighCalmnessWord": has_hi_calm, "NRCCountHighCalmnessWords": hi_calm_count,
    }
    avg_moral = sum_moral/count_moral if count_moral else 0
    moral_cols = {
        "NRCHasHighMoralTrustWord": has_hi_moral, "NRCCountHighMoralTrustWord": hi_moral_count,
        "NRCHasLowMoralTrustWord": has_lo_moral, "NRCCountLowMoralTrustWord": lo_moral_count,
        "NRCAvgMoralTrustWord": avg_moral,
    }
    avg_soc = sum_soc/count_soc if count_soc else 0
    soc_cols = {
        "NRCHasHighSocialWarmthWord": has_hi_soc, "NRCCountHighSocialWarmthWord": hi_soc_count,
        "NRCHasLowSocialWarmthWord": has_lo_soc, "NRCCountLowSocialWarmthWord": lo_soc_count,
        "NRCAvgSocialWarmthWord": avg_soc,
    }
    avg_warmth = sum_warm/count_warm if count_warm else 0
    warm_cols = {
        "NRCHasHighWarmthWord": has_hi_warm, "NRCCountHighWarmthWord": hi_warm_count,
        "NRCHasLowWarmthWord": has_lo_warm, "NRCCountLowWarmthWord": lo_warm_count,
        "NRCAvgWarmthWord": avg_warmth,
    }
    return {
        **avg_vad, **vad_thr, **emo_cols, **emo_cnt_cols,
        **vad_cnt, **word_count, **worry_cols, **moral_cols, **soc_cols, **warm_cols,
    }

def load_body_parts(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip()]

BODY_PARTS = load_body_parts(_DATA_DIR / "bodywords-full.txt")


def compute_prefixed_body_part_mentions(text: str, body_parts: List[str]) -> Dict[str, Any]:
    lower = text.lower()
    prefixes = [("my ","MyBPM"),("your ","YourBPM"),("her ","HerBPM"),("his ","HisBPM"),("their ","TheirBPM")]
    res: Dict[str, Any] = {}
    for pref,label in prefixes:
        res[label] = ", ".join(p for p in (pref+bp for bp in body_parts) if p in lower)
    res["HasBPM"] = any(bp in lower for bp in body_parts)
    return res

def compute_individual_pronouns(text: str) -> Dict[str, int]:
    pronouns = {
        "PRNHasI":["i"],"PRNHasMe":["me"],"PRNHasMy":["my"],"PRNHasMine":["mine"],
        "PRNHasWe":["we"],"PRNHasOur":["our"],"PRNHasOurs":["ours"],
        "PRNHasYou":["you"],"PRNHasYour":["your"],"PRNHasYours":["yours"],
        "PRNHasShe":["she"],"PRNHasHer":["her"],"PRNHasHers":["hers"],
        "PRNHasHe":["he"],"PRNHasHim":["him"],"PRNHasHis":["his"],
        "PRNHasThey":["they"],"PRNHasThem":["them"],"PRNHasTheir":["their"],"PRNHasTheirs":["theirs"],
    }
    words = set(text.lower().split()) if isinstance(text,str) else set()
    return {col: int(any(w in words for w in lst)) for col,lst in pronouns.items()}

# -------------------- #
# TUSC-specific Helper
# -------------------- #

def detect_self_identification_in_tusc_entry(
    entry: Dict[str, Any], detector: SelfIdentificationDetector
) -> Dict[str, List[str]]:
    tweet = entry.get("Tweet", "") or ""
    # Create combined entry preserving original metadata
    combined_entry = entry.copy()
    combined_entry.update({"title": "", "selftext": tweet})
    return detect_self_identification_with_resolved_age(combined_entry, detector)

def apply_linguistic_features(text: str, include_features: bool = True) -> Dict[str, Any]:
    if not include_features or not text:
        return {}
    features = compute_vad_and_emotions(
        text, vad_dict, emotion_dict, emotions, worry_dict,
        moraltrust_dict, socialwarmth_dict, warmth_dict
    )
    features.update(compute_individual_pronouns(text))
    features.update(compute_prefixed_body_part_mentions(text, BODY_PARTS))
    return features

# -------------------- #
# Flattening and I/O Utilities
# -------------------- #

def flatten_result_to_csv_row(
    result: Dict[str, Any], data_source: str, split: Optional[str] = None
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    # Author column
    if data_source == "tusc":
        row["Author"] = result.get("userID", "") or ""
    else:
        row["Author"] = result.get("author", "") or ""

    # Compute majority birthyear and raw birthyear extractions
    self_id = result.get("self_identification", {})
    if data_source == "tusc":
        ref_year = int(result.get("Year", ""))
    else:
        ts = result.get("post", {}).get("created_utc") or result.get("created_utc")
        ref_year = datetime.utcfromtimestamp(int(ts)).year
    raw_matches: List[str] = []
    majority_birthyear: Optional[int] = None
    if "resolved_age" in self_id:
        raw_matches = list(self_id["resolved_age"].get("raw_matches", []))
        age_val = self_id["resolved_age"].get("age")
        if isinstance(age_val, int):
            majority_birthyear = ref_year - age_val
    else:
        raw_matches = list(self_id.get("age", []))
        if raw_matches:
            try:
                val = int(raw_matches[0])
            except ValueError:
                val = None
            if isinstance(val, int):
                if 1900 <= val <= ref_year:
                    majority_birthyear = val
                elif 1 <= val <= 120:
                    majority_birthyear = ref_year - val
    # Convert raw matches to birthyears
    raw_birthyears: List[int] = []
    for m in raw_matches:
        try:
            v = int(m)
        except ValueError:
            continue
        if 1900 <= v <= ref_year:
            raw_birthyears.append(v)
        elif 1 <= v <= 120:
            raw_birthyears.append(ref_year - v)
    row["DMGMajorityBirthyear"] = majority_birthyear or ""
    row["DMGRawBirthyearExtractions"] = "|".join(str(x) for x in raw_birthyears)

    # Include age at posting if available (stage2)
    if "DMGAgeAtPost" in result:
        row["DMGAgeAtPost"] = result.get("DMGAgeAtPost", "")

    if data_source == "tusc":
        row["PostID"] = result.get("TweetID", "")
        row["PostText"] = result.get("Tweet", "")
        row["PostCreatedAt"] = result.get("createdAt", "")
        row["PostYear"] = result.get("Year", "")
        row["PostMonth"] = result.get("Month", "")
        if split == "city":
            row["PostCity"] = result.get("City", "")
            loc_fields = ["PostCity"]
        else:
            row["PostCountry"] = result.get("Country", "")
            row["PostMyCountry"] = result.get("MyCountry", "")
            loc_fields = ["PostCountry", "PostMyCountry"]
        row["PostPlace"] = result.get("Place", "")
        row["PostPlaceID"] = result.get("PlaceID", "")
        row["PostPlaceType"] = result.get("PlaceType", "")
    else:
        post = result.get("post", result)
        row["PostID"] = post.get("id", "")
        row["PostSubreddit"] = post.get("subreddit", "")
        row["PostTitle"] = post.get("title", "")
        row["PostSelftext"] = post.get("selftext", "")
        row["PostCreatedUtc"] = post.get("created_utc", "")
        row["PostScore"] = post.get("score", "")
        row["PostNumComments"] = post.get("num_comments", "")
        row["PostPermalink"] = post.get("permalink", "")
        row["PostUrl"] = post.get("url", "")
        row["PostMediaPath"] = post.get("media_path", "")
        static_keys = {
            "id", "subreddit", "title", "selftext", "created_utc",
            "score", "num_comments", "permalink", "url", "media_path",
            "author"
        }
        for key, val in post.items():
            if key not in static_keys:
                row[key] = val
    return row

def get_csv_fieldnames(
    data_source: str,
    split: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[str]:
    """
    Get static CSV/TSV header fields based on data source and stage ('users' or 'posts').
    """
    # User-level headers: DMG birthyear info
    if stage == "users":
        user_cols = ["Author", "DMGMajorityBirthyear", "DMGRawBirthyearExtractions"]
    else:
        # Post-level headers: DMG age at post
        user_cols = ["Author", "DMGAgeAtPost"]
    if data_source == "tusc":
        base = user_cols + ["PostID", "PostText", "PostCreatedAt", "PostYear", "PostMonth"]
        if split == "city":
            loc = ["PostCity", "PostPlace", "PostPlaceID", "PostPlaceType"]
        else:
            loc = ["PostCountry", "PostMyCountry", "PostPlace", "PostPlaceID", "PostPlaceType"]
        return base + loc
    # Reddit headers
    base = user_cols + [
        "PostID", "PostSubreddit", "PostTitle", "PostSelftext", "PostCreatedUtc",
        "PostScore", "PostNumComments", "PostPermalink", "PostUrl", "PostMediaPath",
    ]
    return base

def ensure_output_directory(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def write_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str,
    output_tsv: bool,
    data_source: str,
    split: Optional[str] = None,
) -> None:
    sep = "\t" if output_tsv else ","
    ext = "tsv" if output_tsv else "csv"
    out = output_file.replace('.csv', f'.{ext}') if output_tsv else output_file
    ensure_output_directory(out)
    # Determine whether writing users or posts file based on filename
    fname = os.path.basename(out)
    stage = 'posts' if 'posts' in fname else 'users'
    if results:
        rows = [flatten_result_to_csv_row(r, data_source, split) for r in results]
        # determine header including any feature columns (e.g., for Reddit stage2)
        if data_source != "tusc":
            static_fields = get_csv_fieldnames(data_source, split, stage)
            extra_fields = sorted({k for row in rows for k in row.keys() if k not in static_fields})
            fieldnames = static_fields + extra_fields
        else:
            fieldnames = get_csv_fieldnames(data_source, split, stage)
        with open(out, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with open(out, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=get_csv_fieldnames(data_source, split, stage),
                delimiter=sep
            )
            writer.writeheader()

# -------------------- #
# JSONL File Handling & Filtering
# -------------------- #

def get_all_jsonl_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for root, _, fnames in os.walk(path):
        for nm in fnames:
            if nm.startswith('RS_'):
                files.append(os.path.join(root, nm))
    return files

def clean_text_newlines(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'(?<!\s)\n(?!\s)', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def filter_entry(
    entry: Dict[str, Any],
    split: str,
    min_words: int,
    max_words: int,
) -> bool:
    if entry.get('over_18', False):
        return False
    if entry.get('promoted') is True or entry.get('whitelist_status') == 'promo_specified':
        return False
    text = entry.get('selftext', '')
    if not text.strip():
        return False
    n = len(text.strip().split())
    if n < min_words or n > max_words:
        return False
    has_vid = bool(entry.get('is_video', False))
    url = entry.get('url', '') or ''
    has_img = any(url.lower().endswith(ext) for ext in ('.jpg','.png','.jpeg','.gif'))
    if split == 'text' and (has_vid or has_img):
        return False
    if split == 'multimodal' and not (has_vid or has_img):
        return False
    return True

def extract_columns(entry: Dict[str, Any], local_media_path: Optional[str]) -> Dict[str, Any]:
    return {
        'id': entry.get('id'),
        'subreddit': entry.get('subreddit'),
        'title': entry.get('title',''),
        'selftext': clean_text_newlines(entry.get('selftext','')),
        'created_utc': entry.get('created_utc'),
        'score': entry.get('score'),
        'num_comments': entry.get('num_comments'),
        'author': entry.get('author'),
        'permalink': entry.get('permalink'),
        'url': entry.get('url'),
        'media_path': local_media_path,
    }