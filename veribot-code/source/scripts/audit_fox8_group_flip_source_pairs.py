"""Test every native-neighbor group-pivotal Fox8 judgment at source-pair level.

For each native graph neighbor, redirect every raw reply/retweet target pointer
between the two accounts to an out-of-population identifier. Preserve tweet
content, reply/retweet status, embedded profile, reference and frozen model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_interaction_all_tweet_deletions import native_values, pointers
from audit_fox8_interaction_raw_tweet_deletions import selected_raw_rows, sha256
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, TWEET_FIELDS, tweet_features
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
GROUP = ROOT / "research/certificates/best_source_group_effects.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_group_flip_source_pairs.json"


def encoded(runtime: Fox8NeighborAtomicRuntime, facts: np.ndarray,
            supports: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 43:74] *= supports / (supports + 5.0)
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def run() -> dict:
    group = json.loads(GROUP.read_text(encoding="utf-8"))
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))
    events = event_index["events"]
    counts = Counter({
        (str(e["source_id"]), str(e["target_id"]), str(e["relation"])):
            int(e["count"]) for e in events})
    pair_events = defaultdict(list)
    for event in events:
        key = frozenset((str(event["source_id"]), str(event["target_id"])))
        if len(key) == 2:
            pair_events[key].append(event)
    fixed = []
    all_source_ids = set()
    for seed in (0, 1, 2):
        row = next(r for r in group["runs"]
                   if r["dataset"] == "fox8" and r["seed"] == seed)
        with np.load(row["group_effect_path"], allow_pickle=False) as z:
            names = z["group_names"].astype(str).tolist()
            j = names.index("native_neighbor_means")
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            without = z["without_group_probability"][:, j].astype(float)
            threshold = float(z["threshold"])
        selected = ids[(full >= threshold) != (without >= threshold)].tolist()
        if len(selected) != row["source_groups"]["native_neighbor_means"]["class_changed_accounts"]:
            raise ValueError("group-pivotal target population changed")
        fixed.append((seed, selected, ids.tolist(), full, threshold))
        all_source_ids.update(selected)
    runtimes = {}
    for seed, targets, _, _, _ in fixed:
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        runtimes[seed] = runtime
        for uid in targets:
            all_source_ids.update(runtime.neighbors[runtime.index[uid]])
    raw_rows = selected_raw_rows(all_source_ids)
    known = set(runtimes[0].ids)
    source_denominators = {}
    for uid, (line_number, line_hash, record) in raw_rows.items():
        tweets = record.get("user_tweets") or []
        source_denominators[uid] = Counter(
            kind for tweet in tweets for kind, _, _ in pointers(tweet, uid, known))
    locator_checked = set()
    for pair, pair_rows in pair_events.items():
        if not pair <= all_source_ids:
            continue
        for event in pair_rows:
            source = str(event["source_id"])
            target = str(event["target_id"])
            kind = str(event["relation"])
            line_number, _, record = raw_rows[source]
            locators = event["locators"]
            if len(locators) != int(event["count"]):
                raise ValueError("event count differs from raw locator count")
            for loc in locators:
                position = int(loc["tweet_index"])
                if int(loc["line"]) != line_number:
                    raise ValueError("event does not locate first raw account record")
                tweet = record["user_tweets"][position]
                if (kind, target, True) not in pointers(tweet, source, known):
                    raise ValueError("raw event target pointer does not replay")
                locator_checked.add((source, target, kind, line_number, position))
    output_runs = []
    for seed, targets, ids, full, threshold in fixed:
        runtime = runtimes[seed]
        if not set(ids) <= set(runtime.ids) or runtime.threshold != threshold:
            raise ValueError("frozen model population or threshold changed")
        target_idx = np.asarray([runtime.index[uid] for uid in targets])
        base = expit(np.asarray(runtime.model.booster_.predict(
            encoded(runtime, runtime.facts[target_idx],
                    runtime.supports[target_idx]), raw_score=True), dtype=float))
        with np.load(next(r["group_effect_path"] for r in group["runs"]
                          if r["dataset"] == "fox8" and r["seed"] == seed),
                     allow_pickle=False) as z:
            original_by_id = dict(zip(ids, full))
        if max(abs(float(base[k]) - original_by_id[uid])
               for k, uid in enumerate(targets)) > 1e-12:
            raise ValueError("original model probability does not replay")
        table = pd.read_csv(runtime.features_path)
        if table["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("source table order differs")
        published = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        runs = []
        alternative_facts = []
        alternative_supports = []
        for target in targets:
            target_i = runtime.index[target]
            neighbors = runtime.neighbors[target_i]
            for peer in neighbors:
                pair = frozenset((target, peer))
                rows = pair_events.get(pair, [])
                if not rows:
                    raise ValueError("native neighbor lacks published relation source")
                edited_counts = counts.copy()
                removed_outgoing = Counter()
                locators = []
                for event in rows:
                    source = str(event["source_id"])
                    destination = str(event["target_id"])
                    kind = str(event["relation"])
                    key = (source, destination, kind)
                    removed_outgoing[(source, kind)] += edited_counts[key]
                    edited_counts[key] = 0
                    locators.extend({
                        "source_account_id": source,
                        "source_line": int(loc["line"]),
                        "source_line_sha256": raw_rows[source][1],
                        "tweet_index": int(loc["tweet_index"]),
                        "relation": kind,
                        "old_target_account_id": destination,
                    } for loc in event["locators"])
                if not locators:
                    raise ValueError("pair source has no raw pointers")
                updated = runtime.facts[target_i].copy()
                old_fractions = []
                new_fractions = []
                for kind in ("reply", "retweet"):
                    denominator = source_denominators[target][kind]
                    numerator = sum(n for (src, _, k), n in counts.items()
                                    if src == target and k == kind)
                    old_fractions.append(numerator / denominator if denominator else np.nan)
                    new_fractions.append((numerator - removed_outgoing[(target, kind)]) /
                                         denominator if denominator else np.nan)
                baseline_native = native_values(target, counts, tuple(old_fractions))
                if not np.allclose(baseline_native, runtime.facts[target_i, 31:43],
                                   rtol=0, atol=1e-12, equal_nan=True):
                    raise ValueError("raw baseline native facts do not replay")
                updated[31:43] = native_values(target, edited_counts,
                                               tuple(new_fractions))
                remaining = [runtime.index[uid] for uid in neighbors if uid != peer]
                if remaining:
                    values = published[remaining]
                    support = np.isfinite(values).sum(axis=0).astype(np.int32)
                    mean = np.nansum(values, axis=0) / np.maximum(support, 1)
                    mean[support == 0] = np.nan
                    updated[43:74] = mean.astype(np.float32)
                else:
                    support = np.zeros(31, dtype=np.int32)
                    updated[43:74] = np.nan
                if not np.array_equal(runtime.facts[target_i, :31], updated[:31],
                                      equal_nan=True):
                    raise ValueError("pointer redirection changed profile/tweet facts")
                alternative_facts.append(updated)
                alternative_supports.append(support)
                runs.append({
                    "target_account_id": target,
                    "neighbor_account_id": peer,
                    "target_neighbor_count_before": len(neighbors),
                    "target_neighbor_count_after": len(remaining),
                    "raw_relation_pointer_count": len(locators),
                    "raw_relation_pointers": sorted(locators, key=lambda x: (
                        x["source_account_id"], x["source_line"], x["tweet_index"],
                        x["relation"])),
                    "target_probability_before": original_by_id[target],
                    "changed_target_facts": [
                        {"name": runtime.reference.names[j],
                         "before": float(runtime.facts[target_i, j])
                            if np.isfinite(runtime.facts[target_i, j]) else None,
                         "after": float(updated[j]) if np.isfinite(updated[j]) else None}
                        for j in range(31, 74)
                        if not np.array_equal(runtime.facts[target_i, j:j+1],
                                              updated[j:j+1], equal_nan=True)],
                })
        alternative_facts = np.asarray(alternative_facts, dtype=float)
        alternative_supports = np.asarray(alternative_supports, dtype=np.int32)
        after = expit(np.asarray(runtime.model.booster_.predict(
            encoded(runtime, alternative_facts, alternative_supports),
            raw_score=True), dtype=float))
        for row, p in zip(runs, after):
            before = float(row["target_probability_before"])
            row["target_probability_after"] = float(p)
            row["target_class_before"] = int(before >= threshold)
            row["target_class_after"] = int(p >= threshold)
            row["target_class_changed"] = row["target_class_before"] != row["target_class_after"]
            row["target_probability_changed"] = abs(float(p)-before) > 1e-10
        by_target = defaultdict(list)
        for row in runs:
            by_target[row["target_account_id"]].append(row)
        output_runs.append({
            "seed": seed,
            "model_sha256": sha256(runtime.run_dir / "model.joblib"),
            "threshold": threshold,
            "group_pivotal_test_accounts": len(targets),
            "source_neighbor_pairs_tested": len(runs),
            "targets_with_any_one_pair_class_change": sum(
                any(r["target_class_changed"] for r in rows)
                for rows in by_target.values()),
            "pairs_with_class_change": sum(r["target_class_changed"] for r in runs),
            "pairs_with_probability_change": sum(r["target_probability_changed"] for r in runs),
            "target_counts": [{
                "target_account_id": uid,
                "neighbor_count": len(by_target[uid]),
                "one_pair_class_changes": sum(r["target_class_changed"] for r in by_target[uid]),
                "one_pair_probability_changes": sum(
                    r["target_probability_changed"] for r in by_target[uid]),
            } for uid in targets],
            "pair_operations": runs,
        })
        print(json.dumps({k: output_runs[-1][k] for k in (
            "seed", "group_pivotal_test_accounts", "source_neighbor_pairs_tested",
            "targets_with_any_one_pair_class_change", "pairs_with_class_change")}),
            flush=True)
    result = {
        "status": "all_group_pivotal_fox8_targets_all_native_neighbor_pairs_raw_pointer_replayed",
        "selection": "all fixed-test judgments whose class changes on full native-neighbor encoded group mask, with no label or source-response selection",
        "operation": "redirect all raw reply/retweet target-account pointers between one pair to a synthetic out-of-population account; keep tweet content, reply/retweet status and profile; rebuild typed events, direct native facts, neighbor means/support and same frozen model",
        "scope_limit": "hypothetical source-pair sensitivity, not a known source error or a measured natural correction; pair removal differs from encoded group masking",
        "fixed_test_labels_read": False,
        "raw_source_sha256": sha256(ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"),
        "published_feature_sha256": sha256(runtimes[0].features_path),
        "event_index_sha256": sha256(EVENTS),
        "group_effect_certificate_sha256": sha256(GROUP),
        "raw_pointer_locators_verified": len(locator_checked),
        "runs": output_runs,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
