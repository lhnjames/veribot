"""Rebuild reported minimal atomic-pointer edits from raw locators and events."""

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
SOURCE = ROOT / "research/certificates/fox8_minimal_atomic_pointer_coalitions.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_minimal_atomic_pointer_coalitions_readback.json"


def run() -> dict:
    prior = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (prior["fixed_test_labels_read"] is not False or
            prior["event_index_sha256"] != sha256(EVENTS)):
        raise ValueError("source pointer certificate or event index changed")
    events = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    all_ids = {x["target_account_id"] for run in prior["runs"]
               for x in run["rows"]}
    all_ids.update(x["source_account_id"] for run in prior["runs"]
                   for row in run["rows"]
                   for x in row["first_minimal_pointer_locators"])
    raw_rows = selected_raw_rows(all_ids)
    output_runs = []
    for run in prior["runs"]:
        runtime = Fox8NeighborAtomicRuntime(ROOT, run["seed"])
        table = pd.read_csv(runtime.features_path)
        if table["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("published source population differs")
        published = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        max_difference = 0.0
        checked_locators = 0
        for row in run["rows"]:
            target = row["target_account_id"]
            selected = row["first_minimal_pointer_locators"]
            if len(selected) != row["minimum_raw_pointers_changing_class"]:
                raise ValueError("minimal pointer count differs")
            removed = {(p["source_account_id"], p["old_target_account_id"],
                        p["relation"], p["source_line"], p["tweet_index"])
                       for p in selected}
            if len(removed) != len(selected):
                raise ValueError("duplicate raw source pointer")
            for p in selected:
                line, line_hash, raw = raw_rows[p["source_account_id"]]
                if line != p["source_line"] or line_hash != p["source_line_sha256"]:
                    raise ValueError("source line or hash differs")
                tweet = raw["user_tweets"][p["tweet_index"]]
                if (p["relation"], p["old_target_account_id"], True) not in pointers(
                        tweet, p["source_account_id"], set(runtime.ids)):
                    raise ValueError("raw pointer no longer targets claimed account")
            rebuilt = Counter()
            found = set()
            for event in events:
                source = str(event["source_id"])
                destination = str(event["target_id"])
                kind = str(event["relation"])
                surviving = 0
                for locator in event["locators"]:
                    key = (source, destination, kind, int(locator["line"]),
                           int(locator["tweet_index"]))
                    if key in removed:
                        found.add(key)
                    else:
                        surviving += 1
                rebuilt[(source, destination, kind)] = surviving
            if found != removed:
                raise ValueError("raw pointer absent from event index")
            checked_locators += len(found)
            _, _, raw_target = raw_rows[target]
            denominator = Counter(
                kind for tweet in raw_target["user_tweets"]
                for kind, _, _ in pointers(tweet, target, set(runtime.ids)))
            fractions = tuple(
                (sum(n for (source, _, relation), n in rebuilt.items()
                     if source == target and relation == kind) / denominator[kind])
                if denominator[kind] else np.nan
                for kind in ("reply", "retweet"))
            idx = runtime.index[target]
            revised = runtime.facts[idx].copy()
            revised[31:43] = native_values(target, rebuilt, fractions)
            remaining_ids = {
                peer for (source, destination, _), n in rebuilt.items()
                if n > 0 for peer in ([destination] if source == target else
                                      [source] if destination == target else [])}
            if not remaining_ids <= set(runtime.neighbors[idx]):
                raise ValueError("edited graph gained a native neighbor")
            remaining = [runtime.index[uid] for uid in runtime.neighbors[idx]
                         if uid in remaining_ids]
            if remaining:
                values = published[remaining]
                supports = np.isfinite(values).sum(axis=0).astype(np.int32)
                means = np.nansum(values, axis=0) / np.maximum(supports, 1)
                means[supports == 0] = np.nan
                revised[43:74] = means.astype(np.float32)
            else:
                supports = np.zeros(31, dtype=np.int32)
                revised[43:74] = np.nan
            matrix = encoded(runtime, revised.reshape(1, -1),
                             supports.reshape(1, -1))
            probability = float(expit(runtime.model.booster_.predict(
                matrix, raw_score=True)[0]))
            difference = abs(probability-row["first_minimal_probability"])
            max_difference = max(max_difference, difference)
            if difference > 1e-12:
                raise ValueError("independent raw pointer replay differs")
            if ((probability >= runtime.threshold) ==
                    (row["probability_before"] >= runtime.threshold)):
                raise ValueError("reported minimal pointer coalition does not flip class")
        output_runs.append({
            "seed": run["seed"],
            "minimum_pointer_coalitions_verified": len(run["rows"]),
            "raw_source_pointers_verified": checked_locators,
            "maximum_probability_replay_difference": max_difference,
        })
        print(json.dumps(output_runs[-1]), flush=True)
    result = {
        "status": "all_reported_minimal_fox8_atomic_pointer_coalitions_independently_rebuilt",
        "source_certificate_sha256": sha256(SOURCE),
        "event_index_sha256": sha256(EVENTS),
        "fixed_test_labels_read": False,
        "runs": output_runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
