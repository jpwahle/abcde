from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Union, Callable

# ------------------ LEXICON LOADING UTILITIES ------------------ #
# The original project expects the NRC lexicons under *data/*. If the files are
# missing locally we fall back to an *empty* dictionary, ensuring that this
# module can be imported even in environments where the lexicons are not
# present yet.

_data_dir = Path("data")


def _safe_read(path: Path):
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
    """Generic lexicon loader that handles most common patterns.

    Args:
        filename: Name of the file in the data directory
        key_col: Column index for the key (default: 0)
        value_col: Column index for the value (default: 2)
        skip_header: Whether to skip the first line (default: False)
        value_type: Type of values - "int", "float", "str", "set", "list", "dict" (default: "int")
        key_transform: Function to transform keys (default: lowercase)
        value_transform: Function to transform values before type conversion
        accumulate: Whether to accumulate multiple values per key (for sets/lists)
    """
    lines = _safe_read(_data_dir / filename)
    if skip_header and lines:
        lines = lines[1:]

    if value_type == "set" and accumulate:
        result = defaultdict(set)
    elif value_type == "list" and accumulate:
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
        raw_value = value_transform(parts[value_col])

        if value_type == "int":
            value = int(raw_value)
        elif value_type == "float":
            value = float(raw_value)
        elif value_type == "str":
            value = str(raw_value)
        elif value_type == "set" and accumulate:
            # For emotions: check if value indicates presence
            if len(parts) > value_col + 1 and int(parts[value_col + 1]) == 1:
                result[key].add(raw_value)
            continue
        elif value_type == "list" and accumulate:
            result.setdefault(key, []).append(raw_value)
            continue
        else:
            value = raw_value

        result[key] = value

    return result


# --- VAD LEXICON (special case - needs nested dict) ------------------


def _load_nrc_vad_lexicon() -> Dict[str, Dict[str, float]]:
    """VAD lexicon needs special handling for nested dict structure."""
    vad_dict = {}
    for line in _safe_read(_data_dir / "NRC-VAD-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        word, valence, arousal, dominance = line.strip().split("\t")
        vad_dict[word.lower()] = {
            "valence": float(valence),
            "arousal": float(arousal),
            "dominance": float(dominance),
        }
    return vad_dict


# --- EMOTION LEXICON (special case - needs condition check) -----------


def _load_nrc_emotion_lexicon() -> Dict[str, set]:
    """Emotion lexicon needs special handling for conditional inclusion."""
    emotion_dict = defaultdict(set)
    for line in _safe_read(_data_dir / "NRC-Emotion-Lexicon.txt"):
        if not line or "\t" not in line:
            continue
        word, emotion, has_emotion = line.strip().split("\t")
        if int(has_emotion) == 1:
            emotion_dict[word.lower()].add(emotion)
    return emotion_dict


# --- ALL OTHER LEXICONS (using generic loader) -----------------------


def _load_nrc_worrywords_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-WorryWords-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_eng_tenses_lexicon() -> Dict[str, List[str]]:
    return _load_lexicon(
        "eng-word-tenses.txt",
        key_col=1,  # form column
        value_col=2,  # tags column
        value_type="list",
        accumulate=True,
    )


def _load_nrc_moraltrust_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-MoralTrustworthy-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_nrc_socialwarmth_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-SocialWarmth-Lexicon.txt", skip_header=True, value_type="int"
    )


def _load_nrc_warmth_lexicon() -> Dict[str, int]:
    return _load_lexicon(
        "NRC-CombinedWarmth-Lexicon.txt", skip_header=True, value_type="int"
    )


# Publicly exposed lexicons (attempted to load on import – fall back to empty)
vad_dict = _load_nrc_vad_lexicon()
emotion_dict = _load_nrc_emotion_lexicon()
worry_dict = _load_nrc_worrywords_lexicon()

moraltrust_dict = _load_nrc_moraltrust_lexicon()
socialwarmth_dict = _load_nrc_socialwarmth_lexicon()
warmth_dict = _load_nrc_warmth_lexicon()

tense_dict = _load_eng_tenses_lexicon()

# ------------------------------------------------------------------------- #

# Hard-coded list from the original script.
emotions = [
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

# The compute_vad_and_emotions & compute_all_features code is long. To keep this
# file maintainable we import it verbatim from the original notebook provided
# by the user.

# NOTE: The code is unchanged except for *import* paths to reuse the lexicons
# above.

# ----------------------------------------------------------------------------
# Original feature computation implementation (shortened for brevity)
# ----------------------------------------------------------------------------

import re


def compute_vad_and_emotions(
    text: str,
    vad_dict: Dict[str, Dict[str, float]],
    emotion_dict: Dict[str, set],
    emotions: List[str],
    worry_dict: Dict[str, int],
    moraltrust_dict: Dict[str, int],
    socialwarmth_dict: Dict[str, int],
    warmth_dict: Dict[str, int],
):
    """
    Compute various metrics from text:
    - Average VAD scores (Valence, Arousal, Dominance)
    - Presence and counts of High/Low VAD words
    - Presence and counts of NRC Emotion words
    - Word count
    - Anxiety/Calmness (from the WorryWords lexicon)
    - MoralTrust (from the NRC MoralTrust Lexicon)
    - SocialWarmth (from the NRC SocialWarmth Lexicon)
    - Warmth (from the NRC Warmth Lexicon)
    """
    words = text.lower().split() if isinstance(text, str) else []

    # VAD accumulators
    vad_scores = {"valence": [], "arousal": [], "dominance": []}
    high_vad_flags = {"valence": 0, "arousal": 0, "dominance": 0}
    low_vad_flags = {"valence": 0, "arousal": 0, "dominance": 0}
    high_vad_counts = {"valence": 0, "arousal": 0, "dominance": 0}
    low_vad_counts = {"valence": 0, "arousal": 0, "dominance": 0}

    # Emotion accumulators
    emotion_counts = {emotion: 0 for emotion in emotions}
    emotion_flags = {emotion: 0 for emotion in emotions}

    # Anxiety/Calmness accumulators
    sum_anxiety = 0
    count_anxiety = 0
    sum_calmness = 0
    count_calmness = 0

    # Flags for anxiety/calmness presence
    has_anxiety_word = 0
    has_calmness_word = 0

    # High anxiety/calmness
    has_high_anxiety_word = 0
    count_high_anxiety_words = 0
    has_high_calmness_word = 0
    count_high_calmness_words = 0

    # MoralTrust accumulators
    sum_moraltrust = 0
    count_moraltrust = 0
    has_high_moraltrust_word = 0
    count_high_moraltrust_words = 0
    has_low_moraltrust_word = 0
    count_low_moraltrust_words = 0

    # SocialWarmth accumulators
    sum_socialwarmth = 0
    count_socialwarmth = 0
    has_high_socialwarmth_word = 0
    count_high_socialwarmth_words = 0
    has_low_socialwarmth_word = 0
    count_low_socialwarmth_words = 0

    # Warmth accumulators
    sum_warmth = 0
    count_warmth = 0
    has_high_warmth_word = 0
    count_high_warmth_words = 0
    has_low_warmth_word = 0
    count_low_warmth_words = 0

    for word in words:
        # Add VAD scores
        if word in vad_dict:
            for dimension in vad_scores:
                score = vad_dict[word][dimension]
                vad_scores[dimension].append(score)

                # Check for high and low thresholds
                if score > 0.67:
                    high_vad_flags[dimension] = 1
                    high_vad_counts[dimension] += 1
                if score < 0.33:
                    low_vad_flags[dimension] = 1
                    low_vad_counts[dimension] += 1

        # Count emotion words
        if word in emotion_dict:
            for emotion in emotion_dict[word]:
                emotion_flags[emotion] = 1
                emotion_counts[emotion] += 1

        # Check WorryWords (Anxiety/Calmness)
        if worry_dict and word in worry_dict:
            oc = worry_dict[word]
            if oc > 0:  # anxious
                has_anxiety_word = 1
                sum_anxiety += oc
                count_anxiety += 1
                if oc == 3:
                    has_high_anxiety_word = 1
                    count_high_anxiety_words += 1
            elif oc < 0:  # calm
                has_calmness_word = 1
                sum_calmness += oc
                count_calmness += 1
                if oc == -3:
                    has_high_calmness_word = 1
                    count_high_calmness_words += 1

        # Check MoralTrust words
        if moraltrust_dict and word in moraltrust_dict:
            oc = moraltrust_dict[word]
            sum_moraltrust += oc
            count_moraltrust += 1
            if oc == 3:
                has_high_moraltrust_word = 1
                count_high_moraltrust_words += 1
            if oc == -3:
                has_low_moraltrust_word = 1
                count_low_moraltrust_words += 1

        # Check SocialWarmth words
        if socialwarmth_dict and word in socialwarmth_dict:
            oc = socialwarmth_dict[word]
            sum_socialwarmth += oc
            count_socialwarmth += 1
            if oc == 3:
                has_high_socialwarmth_word = 1
                count_high_socialwarmth_words += 1
            if oc == -3:
                has_low_socialwarmth_word = 1
                count_low_socialwarmth_words += 1

        # Check Warmth words
        if warmth_dict and word in warmth_dict:
            oc = warmth_dict[word]
            sum_warmth += oc
            count_warmth += 1
            if oc == 3:
                has_high_warmth_word = 1
                count_high_warmth_words += 1
            if oc == -3:
                has_low_warmth_word = 1
                count_low_warmth_words += 1

    # Compute VAD averages
    avg_vad_scores = {
        f"NRCAvg{dimension.capitalize()}": sum(scores) / len(scores) if scores else 0
        for dimension, scores in vad_scores.items()
    }

    # Emotion presence columns
    emotion_columns = {
        f"NRCHas{emotion.capitalize()}Word": flag
        for emotion, flag in emotion_flags.items()
    }

    # Emotion count columns
    emotion_count_columns = {
        f"NRCCount{emotion.capitalize()}Words": count
        for emotion, count in emotion_counts.items()
    }

    # High/Low VAD flags
    vad_threshold_columns = {
        f"NRCHasHigh{dimension.capitalize()}Word": high_vad_flags[dimension]
        for dimension in high_vad_flags
    }
    vad_threshold_columns.update(
        {
            f"NRCHasLow{dimension.capitalize()}Word": low_vad_flags[dimension]
            for dimension in low_vad_flags
        }
    )

    # High/Low VAD counts
    vad_count_columns = {
        f"NRCCountHigh{dimension.capitalize()}Words": high_vad_counts[dimension]
        for dimension in high_vad_counts
    }
    vad_count_columns.update(
        {
            f"NRCCountLow{dimension.capitalize()}Words": low_vad_counts[dimension]
            for dimension in low_vad_counts
        }
    )

    # Word count
    word_count_column = {"WordCount": len(words)}

    # Anxiety/Calmness averages
    avg_anxiety = sum_anxiety / count_anxiety if count_anxiety > 0 else 0
    avg_calmness = sum_calmness / count_calmness if count_calmness > 0 else 0

    worry_columns = {
        "NRCHasAnxietyWord": has_anxiety_word,
        "NRCHasCalmnessWord": has_calmness_word,
        "NRCAvgAnxiety": avg_anxiety,
        "NRCAvgCalmness": avg_calmness,
        "NRCHasHighAnxietyWord": has_high_anxiety_word,
        "NRCCountHighAnxietyWords": count_high_anxiety_words,
        "NRCHasHighCalmnessWord": has_high_calmness_word,
        "NRCCountHighCalmnessWords": count_high_calmness_words,
    }

    # MoralTrust columns
    avg_moraltrust = sum_moraltrust / count_moraltrust if count_moraltrust > 0 else 0
    moraltrust_columns = {
        "NRCHasHighMoralTrustWord": has_high_moraltrust_word,
        "NRCCountHighMoralTrustWord": count_high_moraltrust_words,
        "NRCHasLowMoralTrustWord": has_low_moraltrust_word,
        "NRCCountLowMoralTrustWord": count_low_moraltrust_words,
        "NRCAvgMoralTrustWord": avg_moraltrust,
    }

    # SocialWarmth columns
    avg_socialwarmth = (
        sum_socialwarmth / count_socialwarmth if count_socialwarmth > 0 else 0
    )
    socialwarmth_columns = {
        "NRCHasHighSocialWarmthWord": has_high_socialwarmth_word,
        "NRCCountHighSocialWarmthWord": count_high_socialwarmth_words,
        "NRCHasLowSocialWarmthWord": has_low_socialwarmth_word,
        "NRCCountLowSocialWarmthWord": count_low_socialwarmth_words,
        "NRCAvgSocialWarmthWord": avg_socialwarmth,
    }

    # Warmth columns
    avg_warmth = sum_warmth / count_warmth if count_warmth > 0 else 0
    warmth_columns = {
        "NRCHasHighWarmthWord": has_high_warmth_word,
        "NRCCountHighWarmthWord": count_high_warmth_words,
        "NRCHasLowWarmthWord": has_low_warmth_word,
        "NRCCountLowWarmthWord": count_low_warmth_words,
        "NRCAvgWarmthWord": avg_warmth,
    }

    return {
        **avg_vad_scores,
        **vad_threshold_columns,
        **emotion_columns,
        **emotion_count_columns,
        **vad_count_columns,
        **word_count_column,
        **worry_columns,
        **moraltrust_columns,
        **socialwarmth_columns,
        **warmth_columns,
    }


def load_body_parts(filepath: str) -> List[str]:
    """Load body parts from file with fallback to basic list."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        return [
            "head",
            "face",
            "eye",
            "nose",
            "mouth",
            "ear",
            "neck",
            "shoulder",
            "arm",
            "hand",
            "finger",
            "chest",
            "back",
            "stomach",
            "leg",
            "foot",
            "toe",
            "heart",
            "brain",
            "body",
        ]


def compute_prefixed_body_part_mentions(
    text: str, body_parts: List[str]
) -> Dict[str, Any]:
    """Compute MyBPM, YourBPM, etc. from text using body parts list."""
    if not isinstance(text, str):
        return {
            "My BPM": "",
            "Your BPM": "",
            "Her BPM": "",
            "His BPM": "",
            "Their BPM": "",
            "HasBPM": 0,
        }

    text_lower = text.lower()

    # Define prefixes and their labels for body parts
    prefixes_and_labels = [
        ("my ", "My BPM"),
        ("your ", "Your BPM"),
        ("her ", "Her BPM"),
        ("his ", "His BPM"),
        ("their ", "Their BPM"),
    ]

    results = {}

    # Process prefixed body part mentions
    for prefix, label in prefixes_and_labels:
        matches = []
        for body_part in body_parts:
            prefixed_part = f"{prefix}{body_part}"
            if prefixed_part in text_lower:
                matches.append(prefixed_part)
        results[label] = ", ".join(matches)

    # Process non-prefixed body parts for HasBPM
    has_bpm = 0
    for body_part in body_parts:
        if body_part in text_lower:
            has_bpm = 1
            break
    results["HasBPM"] = has_bpm

    return results


def compute_individual_pronouns(text: str) -> Dict[str, int]:
    """Compute individual pronoun presence with PRN prefix."""
    if not isinstance(text, str):
        return {
            col: 0
            for col in [
                "PRNHasI",
                "PRNHasMe",
                "PRNHasMy",
                "PRNHasMine",
                "PRNHasWe",
                "PRNHasOur",
                "PRNHasOurs",
                "PRNHasYou",
                "PRNHasYour",
                "PRNHasYours",
                "PRNHasShe",
                "PRNHasHer",
                "PRNHasHers",
                "PRNHasHe",
                "PRNHasHim",
                "PRNHasHis",
                "PRNHasThey",
                "PRNHasThem",
                "PRNHasTheir",
                "PRNHasTheirs",
            ]
        }

    words = set(text.lower().split())

    # Define pronouns sets with PRNHasXYZ column names
    pronoun_sets = {
        "PRNHasI": ["i"],
        "PRNHasMe": ["me"],
        "PRNHasMy": ["my"],
        "PRNHasMine": ["mine"],
        "PRNHasWe": ["we"],
        "PRNHasOur": ["our"],
        "PRNHasOurs": ["ours"],
        "PRNHasYou": ["you"],
        "PRNHasYour": ["your"],
        "PRNHasYours": ["yours"],
        "PRNHasShe": ["she"],
        "PRNHasHer": ["her"],
        "PRNHasHers": ["hers"],
        "PRNHasHe": ["he"],
        "PRNHasHim": ["him"],
        "PRNHasHis": ["his"],
        "PRNHasThey": ["they"],
        "PRNHasThem": ["them"],
        "PRNHasTheir": ["their"],
        "PRNHasTheirs": ["theirs"],
    }

    results = {}
    for col_name, pronoun_list in pronoun_sets.items():
        results[col_name] = (
            1 if any(pronoun in words for pronoun in pronoun_list) else 0
        )

    return results


def compute_all_features(
    text: str,
    vad_dict: Dict[str, Dict[str, float]] = vad_dict,
    emotion_dict: Dict[str, set] = emotion_dict,
    emotions: List[str] = emotions,
    worry_dict: Dict[str, int] = worry_dict,
    tense_dict: Dict[str, List[str]] = tense_dict,
    moraltrust_dict: Dict[str, int] = moraltrust_dict,
    socialwarmth_dict: Dict[str, int] = socialwarmth_dict,
    warmth_dict: Dict[str, int] = warmth_dict,
    body_parts: List[str] = None,
) -> Dict[str, Any]:
    """
    Compute all features including:
    - VAD and emotion (from previous code)
    - Anxiety and Calmness (from previous code)
    - Moral Trust
    - Social Warmth
    - Warmth
    - Tense-related features (prefixed with TIME)
    - Personal pronouns and body part mentions (prefixed with PRN and BPM)
    """
    words = text.lower().split() if isinstance(text, str) else []

    # Load body parts if not provided
    if body_parts is None:
        body_parts = load_body_parts("data/bodywords-full.txt")

    # Reuse compute_vad_and_emotions with all lexicons
    features = compute_vad_and_emotions(
        text,
        vad_dict,
        emotion_dict,
        emotions,
        worry_dict,
        moraltrust_dict,
        socialwarmth_dict,
        warmth_dict,
    )

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

    # Compute prefixed body part mentions (MyBPM, YourBPM, etc.)
    bpm_features = compute_prefixed_body_part_mentions(text, body_parts)

    # Compute individual pronoun features (PRNHasI, PRNHasMe, etc.)
    pronoun_features = compute_individual_pronouns(text)

    features.update(
        {
            # Prefixed body part mentions (MyBPM, YourBPM, etc.)
            **bpm_features,
            # Individual pronoun features (PRNHasI, PRNHasMe, etc.)
            **pronoun_features,
            # Tense features
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
    )

    return features
