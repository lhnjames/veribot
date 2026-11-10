"""Label-blind exact repeated-text fact from Fox8's sampled raw tweets."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FACT_NAME = "sampled_exact_repeat_tweet_fraction"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: object) -> str:
    """Collapse whitespace and case only; preserve all remaining characters."""
    return " ".join(str(text or "").casefold().split())


def account_fact(tweets: list[dict]) -> tuple[float, int, list[dict]]:
    positions: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, tweet in enumerate(tweets):
        text = normalize(tweet.get("text"))
        if text:
            tweet_id = str(tweet.get("id_str") or tweet.get("id") or index)
            positions[text].append((index, tweet_id))
    denominator = sum(len(group) for group in positions.values())
    if not denominator:
        return float("nan"), 0, []
    repeats = []
    numerator = 0
    for text, group in sorted(positions.items()):
        if len(group) < 2:
            continue
        numerator += len(group)
        repeats.append({
            "normalized_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "tweet_indices": [index for index, _ in group],
            "tweet_ids": [tweet_id for _, tweet_id in group],
        })
    return numerator / denominator, denominator, repeats


def build(raw_path: Path, features_csv: Path, output: Path) -> dict:
    with features_csv.open("r", encoding="utf-8", newline="") as stream:
        ids = [str(row["user_id"]) for row in csv.DictReader(stream)]
    if len(ids) != 2279 or len(ids) != len(set(ids)):
        raise ValueError("Fox8 feature account order differs")
    known = set(ids)
    seen = set()
    facts: dict[str, tuple[float, int]] = {}
    trace_path = output.with_name("exact_repeat_sources.jsonl.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "rt", encoding="utf-8") as source, \
            gzip.open(trace_path, "wt", encoding="utf-8") as trace:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in seen:
                continue
            seen.add(uid)
            if uid not in known:
                raise ValueError("raw Fox8 account absent from feature table")
            value, support, repeats = account_fact(row.get("user_tweets") or [])
            facts[uid] = (value, support)
            trace.write(json.dumps({
                "account_id": uid, "raw_line": line_number,
                "nonempty_sampled_tweets": support,
                "repeated_tweet_count": int(round(value * support)) if support else 0,
                "repeated_groups": repeats,
            }, separators=(",", ":")) + "\n")
    if seen != known:
        raise ValueError("raw Fox8 population differs from feature table")
    values = np.asarray([facts[uid][0] for uid in ids], dtype=np.float64)[:, None]
    supports = np.asarray([facts[uid][1] for uid in ids], dtype=np.int32)[:, None]
    np.savez_compressed(output, ids=np.asarray(ids), names=np.asarray([FACT_NAME]),
                        values=values, supports=supports)
    manifest = {
        "dataset": "fox8", "account_count": len(ids),
        "fact_name": FACT_NAME,
        "definition": "number of nonempty sampled tweets whose casefolded whitespace-collapsed text appears at least twice / number of nonempty sampled tweets",
        "missing_rule": "NaN when no nonempty sampled tweets",
        "source_locator": "raw gzip line, tweet index, tweet ID, SHA-256 of normalized text for every repeated group",
        "provenance_level": "sampled_raw_tweet_text",
        "source_sha256": sha256(raw_path),
        "features_csv_sha256": sha256(features_csv),
        "facts_sha256": sha256(output),
        "trace_sha256": sha256(trace_path),
        "label_fields_read": False,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
