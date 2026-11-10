"""Label-free, raw-tweet lexical and interaction-concentration facts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import ijson
import numpy as np


FACT_NAMES = (
    "normalized_template_repeat_fraction",
    "top_retweet_target_fraction",
    "top_mentioned_account_fraction",
    "top_hashtag_fraction",
    "lexical_type_token_ratio",
    "mean_words_per_tweet",
    "mean_hashtags_per_tweet",
    "mean_mentions_per_tweet",
)
URL = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION = re.compile(r"(?<!\w)@(\w+)")
HASHTAG = re.compile(r"(?<!\w)#(\w+)")
RETWEET_TARGET = re.compile(r"^\s*RT\s+@(\w+)", re.IGNORECASE)
NUMERIC = re.compile(r"\d+")
TOKEN = re.compile(r"\w+", re.UNICODE)
SPACES = re.compile(r"\s+")


def extract(tweets: list[str]) -> np.ndarray:
    texts = [str(value or "") for value in tweets]
    n = len(texts)
    if n == 0:
        return np.full(len(FACT_NAMES), np.nan, dtype=np.float64)
    normalized = []
    retweet_targets: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    hashtags: Counter[str] = Counter()
    all_tokens: list[str] = []
    token_total = 0
    for text in texts:
        target = RETWEET_TARGET.search(text)
        if target:
            retweet_targets[target.group(1).casefold()] += 1
        without_urls = URL.sub(" URL ", text)
        mentions.update(value.casefold() for value in MENTION.findall(without_urls))
        hashtags.update(value.casefold() for value in HASHTAG.findall(without_urls))
        cleaned = without_urls
        cleaned = MENTION.sub(" USER ", cleaned)
        cleaned = NUMERIC.sub(" NUM ", cleaned)
        cleaned = SPACES.sub(" ", cleaned.casefold()).strip()
        normalized.append(cleaned)
        tokens = TOKEN.findall(cleaned)
        all_tokens.extend(tokens)
        token_total += len(tokens)

    def top_fraction(counts: Counter[str]) -> float:
        total = sum(counts.values())
        return max(counts.values()) / total if total else float("nan")

    return np.asarray([
        1.0 - len(set(normalized)) / n,
        top_fraction(retweet_targets),
        top_fraction(mentions),
        top_fraction(hashtags),
        len(set(all_tokens)) / token_total if token_total else float("nan"),
        token_total / n,
        sum(hashtags.values()) / n,
        sum(mentions.values()) / n,
    ], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(archive: Path, output: Path, base_facts: Path) -> dict:
    ids, split, positions, values = [], [], [], []
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    ids.append(str(row["ID"]))
                    split.append(part)
                    positions.append(ordinal)
                    values.append(extract(row.get("tweet") or []))
    with np.load(base_facts, allow_pickle=False) as base:
        if ids != base["ids"].astype(str).tolist() or \
                split != base["split"].astype(str).tolist() or \
                positions != base["positions"].astype(int).tolist():
            raise ValueError("lexical fact population differs from official base fact cache")
    matrix = np.stack(values)
    if matrix.shape != (11826, len(FACT_NAMES)):
        raise ValueError("lexical fact matrix or official population differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids), split=np.asarray(split),
                        positions=np.asarray(positions), names=np.asarray(FACT_NAMES),
                        values=matrix)
    manifest = {
        "dataset": "twibot20", "account_count": len(ids),
        "fact_names": list(FACT_NAMES),
        "provenance_level": "official_raw_tweet_samples",
        "source_locator": "ZIP member train/dev/test + array ordinal + account ID + tweet list",
        "source_archive_sha256": sha256(archive),
        "base_facts_sha256": sha256(base_facts),
        "facts_sha256": sha256(output),
        "label_fields_read": False,
        "formula": {
            "normalized_template_repeat_fraction": "1 - unique casefolded tweet templates / sampled tweets; replace URL, mention, digit sequence and normalize whitespace",
            "top_retweet_target_fraction": "largest RT @target count / RT @target count",
            "top_mentioned_account_fraction": "largest @account mention count / all @account mention count",
            "top_hashtag_fraction": "largest hashtag count / all hashtag count",
            "lexical_type_token_ratio": "unique Unicode word tokens / all Unicode word tokens after URL, mention and digit replacement",
            "mean_words_per_tweet": "all normalized Unicode word tokens / sampled tweets",
            "mean_hashtags_per_tweet": "all hashtag occurrences / sampled tweets",
            "mean_mentions_per_tweet": "all mention occurrences / sampled tweets",
        },
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
