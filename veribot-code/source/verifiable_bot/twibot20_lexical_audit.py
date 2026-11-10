"""Independent ZIP-to-cache population audit for TwiBot-20 lexical facts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import zipfile

import ijson
import numpy as np

from .twibot20_lexical_facts import FACT_NAMES, sha256


def recompute(tweets: list[str]) -> np.ndarray:
    messages = [str(item or "") for item in tweets]
    if not messages:
        return np.full(8, np.nan)
    templates: set[str] = set()
    retweet_users: Counter[str] = Counter()
    mentioned_users: Counter[str] = Counter()
    hashtag_words: Counter[str] = Counter()
    words: set[str] = set()
    word_count = 0
    for message in messages:
        retweet = re.match(r"\s*RT\s+@(\w+)", message, flags=re.I)
        if retweet is not None:
            retweet_users[retweet.group(1).casefold()] += 1
        without_links = re.sub(r"https?://\S+", " URL ", message, flags=re.I)
        mentioned_users.update(match.group(1).casefold() for match in
                               re.finditer(r"(?<!\w)@(\w+)", without_links))
        hashtag_words.update(match.group(1).casefold() for match in
                             re.finditer(r"(?<!\w)#(\w+)", without_links))
        template = re.sub(r"(?<!\w)@\w+", " USER ", without_links)
        template = re.sub(r"\d+", " NUM ", template)
        template = " ".join(template.casefold().split())
        templates.add(template)
        current_words = re.findall(r"\w+", template, flags=re.UNICODE)
        words.update(current_words)
        word_count += len(current_words)

    def concentration(counter: Counter[str]) -> float:
        denominator = counter.total()
        return max(counter.values()) / denominator if denominator else float("nan")

    n = len(messages)
    return np.asarray([
        (n - len(templates)) / n,
        concentration(retweet_users),
        concentration(mentioned_users),
        concentration(hashtag_words),
        len(words) / word_count if word_count else float("nan"),
        word_count / n,
        hashtag_words.total() / n,
        mentioned_users.total() / n,
    ])


def audit(archive: Path, base_facts: Path, lexical_facts: Path) -> dict:
    manifest = json.loads(lexical_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    source_hash = sha256(archive)
    if source_hash != manifest["source_archive_sha256"] or \
            sha256(base_facts) != manifest["base_facts_sha256"] or \
            sha256(lexical_facts) != manifest["facts_sha256"]:
        raise ValueError("TwiBot-20 lexical source or cache hashes differ")
    with np.load(lexical_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        split = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        names = cache["names"].astype(str).tolist()
        stored = cache["values"].copy()
    if names != list(FACT_NAMES) or stored.shape != (11826, len(names)):
        raise ValueError("lexical fact vocabulary or official population differs")
    missing = np.zeros(len(names), dtype=np.int64)
    mismatch = np.zeros(len(names), dtype=np.int64)
    offset = 0
    split_counts = {}
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test"):
            count = 0
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    if offset >= len(ids) or ids[offset] != str(row["ID"]) or \
                            split[offset] != part or positions[offset] != ordinal:
                        raise ValueError("source account ID, split or ordinal differs")
                    actual = recompute(row.get("tweet") or [])
                    missing += ~np.isfinite(actual)
                    mismatch += ~np.isclose(actual, stored[offset], rtol=0,
                                            atol=1e-12, equal_nan=True)
                    offset += 1
                    count += 1
            split_counts[part] = count
    if offset != len(ids) or np.any(mismatch):
        raise ValueError(f"lexical cache differs from official ZIP: {mismatch.tolist()}")
    return {
        "dataset": "twibot20",
        "status": "verified_all_labeled_account_lexical_fact_cells_against_official_zip",
        "provenance_level": "official_raw_tweet_samples",
        "source_archive_sha256": source_hash,
        "lexical_facts_sha256": manifest["facts_sha256"],
        "account_count": len(ids), "split_counts": split_counts,
        "fact_count": len(names), "verified_fact_cells": int(stored.size),
        "missing_by_fact": dict(zip(names, map(int, missing))),
        "mismatch_by_fact": dict(zip(names, map(int, mismatch))),
        "label_fields_read": False,
    }
