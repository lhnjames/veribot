"""Independently rebuild each reported minimal Fox8 source-pair coalition."""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_group_flip_source_pairs import encoded
from audit_fox8_interaction_all_tweet_deletions import native_values, pointers
from audit_fox8_interaction_raw_tweet_deletions import selected_raw_rows, sha256
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/certificates/fox8_minimal_source_pair_coalitions.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_minimal_source_pair_coalitions_readback.json"


def run() -> dict:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (source["fixed_test_labels_read"] is not False or
            source["event_index_sha256"] != sha256(EVENTS)):
        raise ValueError("source coalition certificate or event index changed")
    events = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    counts = Counter({
        (str(e["source_id"]), str(e["target_id"]), str(e["relation"])):
            int(e["count"]) for e in events})
    target_ids = {row["target_account_id"] for r in source["runs"]
                  for row in r["targets"]}
    raw_rows = selected_raw_rows(target_ids)
    output_runs = []
    for prior in source["runs"]:
        runtime = Fox8NeighborAtomicRuntime(ROOT, prior["seed"])
        table = pd.read_csv(runtime.features_path)
        if table["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("published source account order differs")
        published = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        max_difference = 0.0
        checked_locators = 0
        for row in prior["targets"]:
            target = row["target_account_id"]
            removed_neighbors = set(row["lowest_bitmask_minimal_neighbor_ids"])
            if (len(removed_neighbors) != row["minimum_source_pairs_changing_class"]
                    or not removed_neighbors <= set(runtime.neighbors[runtime.index[target]])):
                raise ValueError("minimal source neighbor set differs")
            actual_events = [e for e in events
                             if frozenset((str(e["source_id"]),
                                           str(e["target_id"]))) in {
                                 frozenset((target, peer)) for peer in removed_neighbors}]
            actual_locators = {
                (str(e["source_id"]), str(e["target_id"]), str(e["relation"]),
                 int(loc["line"]), int(loc["tweet_index"]))
                for e in actual_events for loc in e["locators"]}
            saved_locators = {
                (x["source_account_id"], x["old_target_account_id"],
                 x["relation"], x["source_line"], x["tweet_index"])
                for x in row["first_minimal_raw_pointers"]}
            if actual_locators != saved_locators:
                raise ValueError("minimal coalition raw source locators differ")
            checked_locators += len(actual_locators)
            after_counts = counts.copy()
            for e in actual_events:
                after_counts[(str(e["source_id"]), str(e["target_id"]),
                              str(e["relation"]))] = 0
            _, _, raw = raw_rows[target]
            denominator = Counter(kind for tweet in raw["user_tweets"]
                                  for kind, _, _ in pointers(tweet, target,
                                                             set(runtime.ids)))
            fractions = tuple(
                (sum(n for (source_uid, _, relation), n in after_counts.items()
                     if source_uid == target and relation == kind) / denominator[kind])
                if denominator[kind] else np.nan
                for kind in ("reply", "retweet"))
            idx = runtime.index[target]
            revised = runtime.facts[idx].copy()
            revised[31:43] = native_values(target, after_counts, fractions)
            remaining = [runtime.index[uid] for uid in runtime.neighbors[idx]
                         if uid not in removed_neighbors]
            if remaining:
                values = published[remaining]
                support = np.isfinite(values).sum(axis=0).astype(np.int32)
                means = np.nansum(values, axis=0) / np.maximum(support, 1)
                means[support == 0] = np.nan
                revised[43:74] = means.astype(np.float32)
            else:
                support = np.zeros(31, dtype=np.int32)
                revised[43:74] = np.nan
            matrix = encoded(runtime, revised.reshape(1, -1),
                             support.reshape(1, -1))
            probability = float(expit(runtime.model.booster_.predict(
                matrix, raw_score=True)[0]))
            difference = abs(probability-row["probability_after_first_minimal"])
            max_difference = max(max_difference, difference)
            if difference > 1e-12:
                raise ValueError("independently rebuilt source coalition differs")
            original_class = row["probability_before"] >= runtime.threshold
            if (probability >= runtime.threshold) == original_class:
                raise ValueError("reported minimal source coalition does not change class")
        output_runs.append({
            "seed": prior["seed"],
            "minimal_source_coalitions_verified": len(prior["targets"]),
            "raw_locators_verified": checked_locators,
            "maximum_probability_rebuild_difference": max_difference,
        })
        print(json.dumps(output_runs[-1]), flush=True)
    result = {
        "status": "all_reported_minimal_fox8_source_pair_coalitions_independently_rebuilt",
        "source_certificate_sha256": sha256(SOURCE),
        "event_index_sha256": sha256(EVENTS),
        "fixed_test_labels_read": False,
        "runs": output_runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
