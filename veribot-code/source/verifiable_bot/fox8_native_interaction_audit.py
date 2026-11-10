"""Independently audit Fox8 interaction facts and every saved source locator."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .fox8_native_interaction_facts import FACT_NAMES, sha256


def audit(raw: Path, features: Path, facts: Path) -> dict:
    manifest = json.loads(facts.with_suffix(".json").read_text(encoding="utf-8"))
    index_path = facts.with_name("interaction_event_index.json")
    source_index = json.loads(index_path.read_text(encoding="utf-8"))
    if (manifest["raw_source_sha256"] != sha256(raw)
            or manifest["features_csv_sha256"] != sha256(features)
            or manifest["facts_sha256"] != sha256(facts)
            or manifest["event_index_sha256"] != sha256(index_path)
            or source_index["raw_source_sha256"] != manifest["raw_source_sha256"]):
        raise ValueError("Fox8 source, fact cache or event index differs")
    with features.open("r", encoding="utf-8", newline="") as stream:
        feature_ids = [str(row["user_id"]) for row in csv.DictReader(stream)]
    with np.load(facts, allow_pickle=False) as cache:
        if (feature_ids != cache["ids"].astype(str).tolist()
                or list(FACT_NAMES) != cache["names"].astype(str).tolist()):
            raise ValueError("Fox8 source account order or fact vocabulary differs")
        actual = cache["values"].astype(float)
    known = set(feature_ids)
    pair_events: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    target_counts: Counter[tuple[str, str]] = Counter()
    seen = set()
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            source = str(row["user_id"])
            if source in seen:
                continue
            seen.add(source)
            for position, tweet in enumerate(row.get("user_tweets") or []):
                event_id = str(tweet.get("id_str") or tweet.get("id") or position)
                reply = tweet.get("in_reply_to_user_id_str") or \
                    tweet.get("in_reply_to_user_id")
                retweet = tweet.get("retweeted_status")
                target_specs = []
                if reply is not None:
                    target_specs.append(("reply", str(reply),
                                         "in_reply_to_user_id_str|in_reply_to_user_id"))
                if isinstance(retweet, dict):
                    user = retweet.get("user") or {}
                    target = user.get("id_str") or user.get("id")
                    if target is not None:
                        target_specs.append(("retweet", str(target),
                                             "retweeted_status.user.id_str|id"))
                for kind, target, field in target_specs:
                    target_counts[source, kind] += 1
                    if target in known and target != source:
                        pair_events[source, target, kind].append({
                            "line": line_number, "tweet_index": position,
                            "field": field, "tweet_id": event_id})
    if seen != known:
        raise ValueError("Fox8 account population differs")
    recorded = {(item["source_id"], item["target_id"], item["relation"]): item
                for item in source_index["events"]}
    if set(recorded) != set(pair_events):
        raise ValueError("typed event index pairs differ")
    for key, locators in pair_events.items():
        if recorded[key]["count"] != len(locators) or \
                recorded[key]["locators"] != locators:
            raise ValueError(f"event locator mismatch: {key}")
    outgoing: dict[tuple[str, str], set[str]] = defaultdict(set)
    incoming: dict[tuple[str, str], set[str]] = defaultdict(set)
    out_counts: Counter[tuple[str, str]] = Counter()
    in_counts: Counter[tuple[str, str]] = Counter()
    for (source, target, kind), locators in pair_events.items():
        outgoing[source, kind].add(target)
        incoming[target, kind].add(source)
        out_counts[source, kind] += len(locators)
        in_counts[target, kind] += len(locators)
    expected = np.zeros_like(actual)
    for i, uid in enumerate(feature_ids):
        for kind, offset in (("reply", 0), ("retweet", 4)):
            out = outgoing[uid, kind]
            incoming_sources = incoming[uid, kind]
            expected[i, offset:offset+4] = np.log1p([
                out_counts[uid, kind], in_counts[uid, kind],
                len(out), len(incoming_sources)])
            expected[i, 8 if kind == "reply" else 9] = np.log1p(
                len(out & incoming_sources))
            total = target_counts[uid, kind]
            expected[i, 10 if kind == "reply" else 11] = (
                out_counts[uid, kind] / total if total else np.nan)
    missing_match = bool(np.array_equal(np.isnan(actual), np.isnan(expected)))
    maximum = float(np.max(np.abs(np.nan_to_num(actual) - np.nan_to_num(expected))))
    if not missing_match or maximum > 1e-12:
        raise ValueError("native interaction facts differ from raw tweet targets")
    return {
        "dataset": "fox8",
        "status": "verified_all_native_interaction_fact_cells_and_event_locators",
        "account_count": len(feature_ids),
        "fact_count": len(FACT_NAMES),
        "fact_cells_including_missing": int(actual.size),
        "nonmissing_fact_cells": int(np.isfinite(actual).sum()),
        "typed_internal_pairs": len(pair_events),
        "source_events": int(sum(map(len, pair_events.values()))),
        "max_absolute_numeric_difference": maximum,
        "missingness_match": missing_match,
        "event_locator_match": True,
        "label_fields_read": False,
        "raw_source_sha256": manifest["raw_source_sha256"],
        "fact_cache_sha256": manifest["facts_sha256"],
        "event_index_sha256": manifest["event_index_sha256"],
    }
