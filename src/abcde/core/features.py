"""Feature extraction functions for linguistic analysis."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# Lazy-loaded lexicons (loaded on first use)
_lexicons_loaded = False
_vad_dict: Dict[str, Dict[str, float]] = {}
_emotion_dict: Dict[str, set] = {}
_worry_dict: Dict[str, int] = {}
_moraltrust_dict: Dict[str, int] = {}
_socialwarmth_dict: Dict[str, int] = {}
_warmth_dict: Dict[str, int] = {}
_tense_dict: Dict[str, List[str]] = {}
_cog_dict: Dict[str, Set[str]] = {}
_body_parts: List[str] = []
_emotions: List[str] = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "negative",
    "positive",
    "sadness",
    "surprise",
    "trust",
]

# Default cognitive verbs
DEFAULT_COGNITIVE_VERBS: List[str] = []


def _ensure_lexicons_loaded() -> None:
    """Ensure lexicons are loaded (lazy loading)."""
    global _lexicons_loaded, _vad_dict, _emotion_dict, _worry_dict
    global _moraltrust_dict, _socialwarmth_dict, _warmth_dict
    global _tense_dict, _cog_dict, _body_parts, DEFAULT_COGNITIVE_VERBS

    if _lexicons_loaded:
        return

    from abcde.lexicons.loaders import (
        load_nrc_vad_lexicon,
        load_nrc_emotion_lexicon,
        load_nrc_worrywords_lexicon,
        load_nrc_moraltrust_lexicon,
        load_nrc_socialwarmth_lexicon,
        load_nrc_warmth_lexicon,
        load_eng_tenses_lexicon,
        load_cog_lexicon,
        load_body_parts,
    )

    _vad_dict = load_nrc_vad_lexicon()
    _emotion_dict = load_nrc_emotion_lexicon()
    _worry_dict = load_nrc_worrywords_lexicon()
    _moraltrust_dict = load_nrc_moraltrust_lexicon()
    _socialwarmth_dict = load_nrc_socialwarmth_lexicon()
    _warmth_dict = load_nrc_warmth_lexicon()
    _tense_dict = load_eng_tenses_lexicon()
    _cog_dict = load_cog_lexicon()
    _body_parts = load_body_parts()

    # Build default cognitive verbs from the cognitive lexicon
    DEFAULT_COGNITIVE_VERBS = sorted(
        {w for terms in _cog_dict.values() for w in terms if w.isalpha()}
    )

    _lexicons_loaded = True


def compute_vad_and_emotions(
    text: str,
    vad_dict: Optional[Dict[str, Dict[str, float]]] = None,
    emotion_dict: Optional[Dict[str, set]] = None,
    emotions: Optional[List[str]] = None,
    worry_dict: Optional[Dict[str, int]] = None,
    moraltrust_dict: Optional[Dict[str, int]] = None,
    socialwarmth_dict: Optional[Dict[str, int]] = None,
    warmth_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Compute VAD (Valence-Arousal-Dominance) and emotion features from text."""
    _ensure_lexicons_loaded()

    # Use defaults if not provided
    vad_dict = vad_dict or _vad_dict
    emotion_dict = emotion_dict or _emotion_dict
    emotions = emotions or _emotions
    worry_dict = worry_dict or _worry_dict
    moraltrust_dict = moraltrust_dict or _moraltrust_dict
    socialwarmth_dict = socialwarmth_dict or _socialwarmth_dict
    warmth_dict = warmth_dict or _warmth_dict

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
                    high_flags[dim] = 1
                    high_counts[dim] += 1
                if sc < 0.33:
                    low_flags[dim] = 1
                    low_counts[dim] += 1
        if w in emotion_dict:
            for emo in emotion_dict[w]:
                emotion_flags[emo] = 1
                emotion_counts[emo] += 1
        if worry_dict and w in worry_dict:
            oc = worry_dict[w]
            if oc > 0:
                has_anx = 1
                sum_anx += oc
                count_anx += 1
                if oc == 3:
                    has_hi_anx = 1
                    hi_anx_count += 1
            elif oc < 0:
                has_calm = 1
                sum_calm += oc
                count_calm += 1
                if oc == -3:
                    has_hi_calm = 1
                    hi_calm_count += 1
        if moraltrust_dict and w in moraltrust_dict:
            oc = moraltrust_dict[w]
            sum_moral += oc
            count_moral += 1
            if oc == 3:
                has_hi_moral = 1
                hi_moral_count += 1
            if oc == -3:
                has_lo_moral = 1
                lo_moral_count += 1
        if socialwarmth_dict and w in socialwarmth_dict:
            oc = socialwarmth_dict[w]
            sum_soc += oc
            count_soc += 1
            if oc == 3:
                has_hi_soc = 1
                hi_soc_count += 1
            if oc == -3:
                has_lo_soc = 1
                lo_soc_count += 1
        if warmth_dict and w in warmth_dict:
            oc = warmth_dict[w]
            sum_warm += oc
            count_warm += 1
            if oc == 3:
                has_hi_warm = 1
                hi_warm_count += 1
            if oc == -3:
                has_lo_warm = 1
                lo_warm_count += 1

    avg_vad = {
        f"NRCAvg{dim.capitalize()}": sum(vals) / len(vals) if vals else 0
        for dim, vals in vad_scores.items()
    }
    emo_cols = {
        f"NRCHas{emo.capitalize()}Word": flag for emo, flag in emotion_flags.items()
    }
    emo_cnt_cols = {
        f"NRCCount{emo.capitalize()}Words": cnt for emo, cnt in emotion_counts.items()
    }
    vad_thr = {
        f"NRCHasHigh{dim.capitalize()}Word": high_flags[dim] for dim in high_flags
    }
    vad_thr.update(
        {f"NRCHasLow{dim.capitalize()}Word": low_flags[dim] for dim in low_flags}
    )
    vad_cnt = {
        f"NRCCountHigh{dim.capitalize()}Words": high_counts[dim] for dim in high_counts
    }
    vad_cnt.update(
        {f"NRCCountLow{dim.capitalize()}Words": low_counts[dim] for dim in low_counts}
    )
    word_count = {"WordCount": len(words)}
    avg_anx = sum_anx / count_anx if count_anx else 0
    avg_calm = sum_calm / count_calm if count_calm else 0
    worry_cols = {
        "NRCHasAnxietyWord": has_anx,
        "NRCHasCalmnessWord": has_calm,
        "NRCAvgAnxiety": avg_anx,
        "NRCAvgCalmness": avg_calm,
        "NRCHasHighAnxietyWord": has_hi_anx,
        "NRCCountHighAnxietyWords": hi_anx_count,
        "NRCHasHighCalmnessWord": has_hi_calm,
        "NRCCountHighCalmnessWords": hi_calm_count,
    }
    avg_moral = sum_moral / count_moral if count_moral else 0
    moral_cols = {
        "NRCHasHighMoralTrustWord": has_hi_moral,
        "NRCCountHighMoralTrustWord": hi_moral_count,
        "NRCHasLowMoralTrustWord": has_lo_moral,
        "NRCCountLowMoralTrustWord": lo_moral_count,
        "NRCAvgMoralTrustWord": avg_moral,
    }
    avg_soc = sum_soc / count_soc if count_soc else 0
    soc_cols = {
        "NRCHasHighSocialWarmthWord": has_hi_soc,
        "NRCCountHighSocialWarmthWord": hi_soc_count,
        "NRCHasLowSocialWarmthWord": has_lo_soc,
        "NRCCountLowSocialWarmthWord": lo_soc_count,
        "NRCAvgSocialWarmthWord": avg_soc,
    }
    avg_warmth = sum_warm / count_warm if count_warm else 0
    warm_cols = {
        "NRCHasHighWarmthWord": has_hi_warm,
        "NRCCountHighWarmthWord": hi_warm_count,
        "NRCHasLowWarmthWord": has_lo_warm,
        "NRCCountLowWarmthWord": lo_warm_count,
        "NRCAvgWarmthWord": avg_warmth,
    }
    return {
        **avg_vad,
        **vad_thr,
        **emo_cols,
        **emo_cnt_cols,
        **vad_cnt,
        **word_count,
        **worry_cols,
        **moral_cols,
        **soc_cols,
        **warm_cols,
    }


def compute_prefixed_body_part_mentions(
    text: str, body_parts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute body part mentions with possessive prefixes."""
    _ensure_lexicons_loaded()
    body_parts = body_parts or _body_parts

    lower = text.lower()
    prefixes = [
        ("my ", "MyBPM"),
        ("your ", "YourBPM"),
        ("her ", "HerBPM"),
        ("his ", "HisBPM"),
        ("their ", "TheirBPM"),
    ]
    res: Dict[str, Any] = {}
    for pref, label in prefixes:
        res[label] = ", ".join(
            p for p in (pref + bp for bp in body_parts) if p in lower
        )
    res["HasBPM"] = any(bp in lower for bp in body_parts)
    return res


def compute_individual_pronouns(text: str) -> Dict[str, int]:
    """Compute individual pronoun presence flags."""
    pronouns = {
        "PRNHasI": ["i"],
        "PRNHasMe": ["me"],
        "PRNHasMy": ["my"],
        "PRNHasMine": ["mine"],
        "PRNHasWe": ["we"],
        "PRNHasOur": ["our"],
        "PRNHasOurs": ["ours"],
        "PRNHasUs": ["us"],
        "PRNHasYou": ["you"],
        "PRNHasYour": ["your"],
        "PRNHasYours": ["yours"],
        "PRNHasShe": ["she"],
        "PRNHasHer": ["her"],
        "PRNHasHers": ["hers"],
        "PRNHasHe": ["he"],
        "PRNHasHim": ["him"],
        "PRNHasHis": ["his"],
        "PRNHasIt": ["it"],
        "PRNHasThey": ["they"],
        "PRNHasThem": ["them"],
        "PRNHasTheir": ["their"],
        "PRNHasTheirs": ["theirs"],
    }
    words = set(text.lower().split()) if isinstance(text, str) else set()
    return {col: int(any(w in words for w in lst)) for col, lst in pronouns.items()}


def compute_tense_features(
    text: str, tense_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, int]:
    """Compute tense-related features from text."""
    _ensure_lexicons_loaded()
    tense_dict = tense_dict or _tense_dict

    words = text.lower().split() if isinstance(text, str) else []

    # Tense-related features
    past_count = 0
    present_count = 0
    future_modals = {"will", "shall", "should", "going to"}
    future_modal_count = 0

    for w in words:
        # Check future modals directly
        if w in future_modals:
            future_modal_count += 1

        # Check tense_dict if available
        if tense_dict and w in tense_dict:
            for tag in tense_dict[w]:
                # Check for past tense
                if "PST" in tag:
                    past_count += 1
                # Check for present tense
                elif "PRS" in tag:
                    present_count += 1

    # Check for future time reference words
    future_time_words = {"tomorrow", "next day", "next year", "next month"}
    joined_text = " ".join(words)
    has_future_time_reference = any(
        phrase in joined_text for phrase in future_time_words
    )

    return {
        "TIMEHasPastVerb": 1 if past_count > 0 else 0,
        "TIMECountPastVerbs": past_count,
        "TIMEHasPresentVerb": 1 if present_count > 0 else 0,
        "TIMECountPresentVerbs": present_count,
        "TIMEHasFutureModal": 1 if future_modal_count > 0 else 0,
        "TIMECountFutureModals": future_modal_count,
        "TIMEHasPresentNoFuture": (
            1 if (present_count > 0 and future_modal_count == 0) else 0
        ),
        "TIMEHasFutureReference": 1 if has_future_time_reference else 0,
    }


def compute_cognitive_features(
    text: str, cog_dict: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, int]:
    """Compute cognitive/thinking word features."""
    _ensure_lexicons_loaded()
    cog_dict = cog_dict or _cog_dict

    if not isinstance(text, str) or not text.strip():
        return {f"COGHas{cat}Word": 0 for cat in cog_dict}
    words = set(text.lower().split())
    return {
        f"COGHas{cat}Word": int(bool(words & terms)) for cat, terms in cog_dict.items()
    }


def _to_ing_form(verb: str) -> str:
    """Naive but practical conversion of lemma -> -ing."""
    v = verb.lower()
    # Irregular/special
    if v == "be":
        return "being"
    if v.endswith("ie") and len(v) > 2:
        return v[:-2] + "ying"  # die->dying, tie->tying
    # keep final 'e' for 'see', 'flee', 'knee' (and similar)
    if v in {"see", "flee", "knee", "agree", "free", "decree"}:
        return v + "ing"
    if v.endswith("e") and len(v) > 2:
        return v[:-1] + "ing"  # make->making
    # CVC doubling (except w,x,y)
    vowels = set("aeiou")
    consonants = set("bcdfghjklmnpqrstvz")  # (skip w,x,y)
    if (
        len(v) >= 3
        and v[-1] in consonants
        and v[-2] in vowels
        and v[-3] in consonants
        and v[-1] not in {"w", "x", "y"}
    ):
        return v + v[-1] + "ing"  # plan->planning
    return v + "ing"


def _build_ing_regex_group(verbs: List[str]) -> str:
    """Return an alternation group of -ing forms for regex (longest first)."""
    ings = sorted({_to_ing_form(v) for v in verbs}, key=len, reverse=True)
    return r"(" + "|".join(re.escape(v) for v in ings) + r")"


def compute_embodied_cognitive_verbs(
    text: str,
    cognitive_verbs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect 'embodied' cognitive verbs: SUBJECT + BE/BEEN/BEING + (adv/neg)* + VBG
    e.g., "I'm thinking", "you were analyzing", "she has been reflecting".
    Returns per-subject matched phrases and an overall boolean.
    """
    _ensure_lexicons_loaded()

    if not isinstance(text, str) or not text.strip():
        return {
            "ICOG": "",
            "YouCOG": "",
            "WeCOG": "",
            "HeCOG": "",
            "SheCOG": "",
            "TheyCOG": "",
            "HasCOG": False,
        }

    cognitive_verbs = cognitive_verbs or DEFAULT_COGNITIVE_VERBS
    verbs_group = _build_ing_regex_group(cognitive_verbs)

    # Accept straight/curly apostrophes in contractions
    apos = r"(?:'|')"
    # Auxiliaries (progressive forms + perfect progressive)
    # Keep concise but robust; add modals + 'be' for "might be thinking"
    AUX = (
        rf"(?:{apos}m|am|are|{apos}re|is|{apos}s|was|were|"
        rf"have\s+been|{apos}ve\s+been|has\s+been|had\s+been|"
        rf"(?:might|may|can|could|should|would|will|shall|must)\s+be|"
        rf"be|been|being)"
    )

    # Up to 3 optional adverbs/negation words between aux and verb (e.g., "just", "really", "not")
    MID = r"(?:\s+(?:not|\w+(?:-\w+)?)){0,3}"

    subjects = [
        (r"\bI\b", "ICOG"),
        (r"\byou\b", "YouCOG"),
        (r"\bwe\b", "WeCOG"),
        (r"\bhe\b", "HeCOG"),
        (r"\bshe\b", "SheCOG"),
        (r"\bthey\b", "TheyCOG"),
    ]

    # Build one regex per subject to keep column separation simple
    patterns = {
        label: re.compile(rf"{subj}\s+{AUX}{MID}\s+{verbs_group}\b", re.IGNORECASE)
        for subj, label in subjects
    }

    lower = text  # keep original case for extracting matched phrase
    out: Dict[str, str] = {}
    any_hits = False

    for label, pat in patterns.items():
        hits = [m.group(0).strip() for m in pat.finditer(lower)]
        # Deduplicate but preserve order
        seen = set()
        uniq = []
        for h in hits:
            if h.lower() not in seen:
                seen.add(h.lower())
                uniq.append(h)
        out[label] = ", ".join(uniq)
        any_hits = any_hits or bool(uniq)

    out["HasCOG"] = any_hits
    return out


def apply_linguistic_features(
    text: str, include_features: bool = True
) -> Dict[str, Any]:
    """Apply all linguistic feature extraction to text.
    
    This is the main entry point for feature extraction.
    """
    if not include_features:
        return {}
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Text for linguistic feature extraction must be a non-empty string"
        )
    
    _ensure_lexicons_loaded()
    
    features = compute_vad_and_emotions(text)
    features.update(compute_individual_pronouns(text))
    features.update(compute_prefixed_body_part_mentions(text))
    features.update(compute_tense_features(text))
    features.update(compute_cognitive_features(text))
    features.update(compute_embodied_cognitive_verbs(text))
    return features

