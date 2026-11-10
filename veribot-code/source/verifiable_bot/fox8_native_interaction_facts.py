"""Account-level facts from Fox8's sampled native reply and retweet targets."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FACT_NAMES = (
    "native_reply_out_tweet_count",
    "native_reply_in_tweet_count",
    "native_reply_out_unique_targets",
    "native_reply_in_unique_sources",
    "native_retweet_out_tweet_count",
    "native_retweet_in_tweet_count",
    "native_retweet_out_unique_targets",
    "native_retweet_in_unique_sources",
    "native_reply_reciprocal_neighbors",
    "native_retweet_reciprocal_neighbors",
    "native_reply_internal_target_fraction",
    "native_retweet_internal_target_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(raw_path: Path, features_csv: Path, output: Path) -> dict:
    with features_csv.open("r", encoding="utf-8", newline="") as stream:
        ids = [str(row["user_id"]) for row in csv.DictReader(stream)]
    if len(ids) != 2279 or len(ids) != len(set(ids)):
        raise ValueError("Fox8 feature account population differs")
    known = set(ids)
    seen = set()
    event_count: Counter[tuple[str, str, str]] = Counter()
    all_targets: Counter[tuple[str, str]] = Counter()
    internal_targets: Counter[tuple[str, str]] = Counter()
    locators: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in seen:
                continue
            seen.add(uid)
            for ordinal, tweet in enumerate(row.get("user_tweets") or []):
                targets = []
                reply = tweet.get("in_reply_to_user_id_str") or \
                    tweet.get("in_reply_to_user_id")
                if reply is not None:
                    targets.append(("reply", str(reply),
                                    "in_reply_to_user_id_str|in_reply_to_user_id"))
                retweet = tweet.get("retweeted_status")
                if isinstance(retweet, dict):
                    author = retweet.get("user") or {}
                    target = author.get("id_str") or author.get("id")
                    if target is not None:
                        targets.append(("retweet", str(target),
                                        "retweeted_status.user.id_str|id"))
                for kind, target, field in targets:
                    all_targets[uid, kind] += 1
                    if target in known and target != uid:
                        internal_targets[uid, kind] += 1
                        event_count[uid, target, kind] += 1
                        locators[uid, target, kind].append({
                            "line": line_number, "tweet_index": ordinal,
                            "field": field,
                            "tweet_id": str(tweet.get("id_str") or
                                            tweet.get("id") or ordinal),
                        })
    if seen != known:
        raise ValueError("Fox8 raw account population differs from features")
    out_peers: dict[tuple[str, str], set[str]] = defaultdict(set)
    in_peers: dict[tuple[str, str], set[str]] = defaultdict(set)
    out_events: Counter[tuple[str, str]] = Counter()
    in_events: Counter[tuple[str, str]] = Counter()
    for (source, target, kind), count in event_count.items():
        out_peers[source, kind].add(target)
        in_peers[target, kind].add(source)
        out_events[source, kind] += count
        in_events[target, kind] += count
    values = []
    for uid in ids:
        reply_out = out_peers[uid, "reply"]
        reply_in = in_peers[uid, "reply"]
        retweet_out = out_peers[uid, "retweet"]
        retweet_in = in_peers[uid, "retweet"]
        values.append([
            np.log1p(out_events[uid, "reply"]),
            np.log1p(in_events[uid, "reply"]),
            np.log1p(len(reply_out)), np.log1p(len(reply_in)),
            np.log1p(out_events[uid, "retweet"]),
            np.log1p(in_events[uid, "retweet"]),
            np.log1p(len(retweet_out)), np.log1p(len(retweet_in)),
            np.log1p(len(reply_out & reply_in)),
            np.log1p(len(retweet_out & retweet_in)),
            (internal_targets[uid, "reply"] / all_targets[uid, "reply"]
             if all_targets[uid, "reply"] else np.nan),
            (internal_targets[uid, "retweet"] / all_targets[uid, "retweet"]
             if all_targets[uid, "retweet"] else np.nan),
        ])
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (2279, len(FACT_NAMES)):
        raise ValueError("Fox8 native fact matrix differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids), names=np.asarray(FACT_NAMES),
                        values=matrix)
    event_index_path = output.with_name("interaction_event_index.json")
    event_index_path.write_text(json.dumps({
        "dataset": "fox8", "raw_source_sha256": sha256(raw_path),
        "events": [
            {"source_id": source, "target_id": target, "relation": kind,
             "count": int(count), "locators": locators[source, target, kind]}
            for (source, target, kind), count in sorted(event_count.items())
        ]}, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "fox8", "account_count": len(ids),
        "fact_names": list(FACT_NAMES),
        "raw_source_sha256": sha256(raw_path),
        "features_csv_sha256": sha256(features_csv),
        "facts_sha256": sha256(output),
        "event_index_sha256": sha256(event_index_path),
        "provenance_level": "sampled_raw_tweet_reply_and_retweet_targets",
        "native_typed_pairs": len(event_count),
        "internal_reply_tweets": int(sum(count for (_, _, kind), count in
                                         event_count.items() if kind == "reply")),
        "internal_retweet_tweets": int(sum(count for (_, _, kind), count in
                                           event_count.items() if kind == "retweet")),
        "source_locator": "raw gzip line + tweet index + target field + tweet ID",
        "formula": "log1p counts of sampled internal target events, distinct peers and reciprocal peers; internal-target fraction among recorded targets",
        "label_fields_read": False,
        "k_nearest_neighbor_graph_used": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
