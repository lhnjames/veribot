"""Source-local, narrowly defined wording overlap in sampled account posts."""

from __future__ import annotations

import difflib
import hashlib
import itertools
import re
import unicodedata
from typing import Sequence

import numpy as np


MAX_POSTS = 40
MIN_CHARACTERS = 25
MIN_TOKENS = 4
MIN_TOKEN_JACCARD = 0.70
MIN_SEQUENCE_RATIO = 0.88
FACT_NAMES = ("first40_high_wording_overlap_fraction",
              "first40_high_wording_overlap_pair_log1p")


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\d+", " ", value)
    return " ".join(re.findall(r"[#@]?\w+", value))


def analyze(tweets: Sequence[str | None]) -> tuple[tuple[float, float], list[dict], int]:
    eligible: list[tuple[int, str, set[str]]] = []
    for position, text in enumerate(tweets[:MAX_POSTS]):
        original = str(text or "")
        if original.lstrip().casefold().startswith("rt @"):
            continue
        value = normalize(original)
        tokens = set(value.split())
        if len(value) >= MIN_CHARACTERS and len(value.split()) >= MIN_TOKENS:
            eligible.append((position, value, tokens))
    if not eligible:
        return (float("nan"), float("nan")), [], 0
    pairs = []
    participants = set()
    pair_count = 0
    for (i, a, at), (j, b, bt) in itertools.combinations(eligible, 2):
        jaccard = len(at & bt) / len(at | bt)
        if jaccard < MIN_TOKEN_JACCARD:
            continue
        sequence = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        if sequence < MIN_SEQUENCE_RATIO:
            continue
        pair_count += 1
        participants.update((i, j))
        if len(pairs) < 5:
            pairs.append({"tweet_indices": [i, j],
                          "normalized_sha256": [hashlib.sha256(v.encode()).hexdigest()
                                                for v in (a, b)],
                          "token_jaccard": jaccard,
                          "sequence_ratio": sequence})
    return (len(participants) / len(eligible), float(np.log1p(pair_count))), \
        pairs, len(eligible)
