"""Independently rederive all nine added facts from raw Fox8 tweet records."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from verifiable_bot.fox8_atomic_sources import EXTENDED_TWEET_FIELDS
from verifiable_bot.fox8_evidence_dimensions import FACT_DIMENSIONS, GRAPH_DIMENSIONS
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS


def independent_tweet_facts(tweets: list[dict]) -> dict[str, float]:
    """Compute facts directly from JSON, without the atomic-source classes."""
    if not tweets:
        return {name: 0.0 for name in EXTENDED_TWEET_FIELDS}
    n = len(tweets)
    timestamps = sorted(int(datetime.strptime(
        item["created_at"], "%a %b %d %H:%M:%S %z %Y").timestamp())
        for item in tweets if isinstance(item.get("created_at"), str)
        and item.get("created_at"))
    gaps = np.diff(timestamps) / 3600 if len(timestamps) > 1 else np.array([])
    languages = [item.get("lang", "und") for item in tweets]
    counts = np.asarray([languages.count(value) for value in set(languages)],
                        dtype=np.float64)
    fractions = counts / n
    texts = [item.get("text", "") or "" for item in tweets]
    text_hashes = [sha256(value.encode("utf-8")).hexdigest() for value in texts]
    def entities(item: dict, field: str) -> int:
        return len(((item.get("entities") or {}).get(field)) or [])
    return {
        "observed_span_hours": ((timestamps[-1] - timestamps[0]) / 3600
                                if len(timestamps) > 1 else 0.0),
        "median_intertweet_gap_hours": float(np.median(gaps)) if len(gaps) else 0.0,
        "gap_cv": (float(np.std(gaps) / np.mean(gaps))
                   if len(gaps) > 1 and np.mean(gaps) > 0 else 0.0),
        "mention_rate": sum(entities(item, "user_mentions") for item in tweets) / n,
        "hashtag_rate": sum(entities(item, "hashtags") for item in tweets) / n,
        "url_rate": sum(entities(item, "urls") for item in tweets) / n,
        "quote_ratio": sum(bool(item.get("is_quote_status", False)) for item in tweets) / n,
        "exact_text_duplicate_ratio": 1 - len(set(text_hashes)) / n,
        "language_entropy": float(-np.sum(fractions * np.log(fractions))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "output/two_stage_fox8_evidence/dimension_audit.json")
    args = parser.parse_args()
    raw_path = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    inventory_path = ROOT / "data/processed/fox8_v3/ids.json"
    if inventory_path.exists():
        ids = set(json.loads(inventory_path.read_text(encoding="utf-8")))
    else:
        # Only first account records are counted, matching the feature builder.
        ids = None
    seen = set()
    positive = {name: 0 for name in EXTENDED_TWEET_FIELDS}
    maximum_difference = {name: 0.0 for name in EXTENDED_TWEET_FIELDS}
    account_count = 0
    tweet_count = 0
    from verifiable_bot.fox8_atomic_sources import load_account_atoms
    # First pass gets the exact account inventory and source-order contract.
    if ids is None:
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            ids = {str(json.loads(line)["user_id"]) for line in stream}
    atoms = load_account_atoms(raw_path, ids)
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in atoms or uid in seen:
                continue
            seen.add(uid)
            expected = independent_tweet_facts(row["user_tweets"])
            observed = atoms[uid].extended_features()
            if atoms[uid].line_number != line_number:
                raise ValueError("first-record source line differs")
            for name in EXTENDED_TWEET_FIELDS:
                difference = abs(expected[name] - observed[name])
                maximum_difference[name] = max(maximum_difference[name], difference)
                if difference > 1e-6:
                    raise ValueError(f"raw fact differs: {uid} {name} {difference}")
                positive[name] += int(observed[name] > 0)
            account_count += 1
            tweet_count += len(row["user_tweets"])
    if seen != set(atoms):
        raise ValueError("account inventory incomplete")
    report = {
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "account_count": account_count,
        "tweet_observation_count": tweet_count,
        "named_node_fact_count": len(ALL_FACTS),
        "dimension_count_including_graph": len(FACT_DIMENSIONS) + 1,
        "fact_dimensions": FACT_DIMENSIONS,
        "graph_relation_dimensions": GRAPH_DIMENSIONS,
        "added_fact_positive_account_counts": positive,
        "added_fact_maximum_raw_rederivation_difference": maximum_difference,
        "interpretation": "Exact source-derived measurements; class meaning requires model response and independent ground truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "account_count", "tweet_observation_count", "named_node_fact_count",
        "dimension_count_including_graph", "added_fact_positive_account_counts",
        "added_fact_maximum_raw_rederivation_difference")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
