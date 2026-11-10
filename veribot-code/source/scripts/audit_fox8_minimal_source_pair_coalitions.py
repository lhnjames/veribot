"""Exact minimal relation-source coalitions for small group-pivotal Fox8 ego graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_group_flip_source_pairs import encoded
from audit_fox8_interaction_all_tweet_deletions import pointers
from audit_fox8_interaction_raw_tweet_deletions import selected_raw_rows, sha256
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT / "research/certificates/fox8_group_flip_source_pairs.json"
JOINT = ROOT / "research/certificates/fox8_group_flip_joint_source_pairs.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_minimal_source_pair_coalitions.json"
MAX_EXACT_NEIGHBORS = 12


def run() -> dict:
    pair = json.loads(PAIR.read_text(encoding="utf-8"))
    joint = json.loads(JOINT.read_text(encoding="utf-8"))
    if joint["pair_certificate_sha256"] != sha256(PAIR):
        raise ValueError("joint and individual source certificates differ")
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    typed = Counter({
        (str(e["source_id"]), str(e["target_id"]), str(e["relation"])):
            int(e["count"]) for e in event_index})
    target_ids = {x["target_account_id"] for run in joint["runs"]
                  for x in run["target_operations"]
                  if x["native_neighbor_count"] <= MAX_EXACT_NEIGHBORS}
    raw_rows = selected_raw_rows(target_ids)
    result_runs = []
    for pair_run, joint_run in zip(pair["runs"], joint["runs"]):
        if pair_run["seed"] != joint_run["seed"]:
            raise ValueError("source seed order differs")
        seed = pair_run["seed"]
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        table = pd.read_csv(runtime.features_path)
        if table["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("source table order differs")
        published = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        by_target = defaultdict(dict)
        for row in pair_run["pair_operations"]:
            by_target[row["target_account_id"]][row["neighbor_account_id"]] = row
        joint_by_target = {r["target_account_id"]: r
                           for r in joint_run["target_operations"]}
        rows = []
        for target in sorted(target_ids & joint_by_target.keys()):
            idx = runtime.index[target]
            neighbors = runtime.neighbors[idx]
            n = len(neighbors)
            if n > MAX_EXACT_NEIGHBORS or n < 1:
                raise ValueError("exact-subset population changed")
            if set(neighbors) != set(by_target[target]):
                raise ValueError("not all neighbor pairs have source operations")
            _, _, record = raw_rows[target]
            denom = Counter(kind for tweet in record["user_tweets"]
                            for kind, _, _ in pointers(tweet, target,
                                                       set(runtime.ids)))
            masks = np.arange(1 << n, dtype=np.int32)
            removed = (masks[:, None] & (1 << np.arange(n))) != 0
            keep = (~removed).astype(np.int32)
            facts = np.repeat(runtime.facts[idx:idx+1], len(masks), axis=0)
            for kind_index, kind in enumerate(("reply", "retweet")):
                out = np.asarray([typed[(target, uid, kind)] for uid in neighbors])
                incoming = np.asarray([typed[(uid, target, kind)] for uid in neighbors])
                out_events = keep @ out
                in_events = keep @ incoming
                out_peers = keep @ (out > 0).astype(np.int32)
                in_peers = keep @ (incoming > 0).astype(np.int32)
                reciprocal = keep @ ((out > 0) & (incoming > 0)).astype(np.int32)
                j = 31 + 4 * kind_index
                facts[:, j:j+4] = np.stack((
                    np.log1p(out_events), np.log1p(in_events),
                    np.log1p(out_peers), np.log1p(in_peers)), axis=1)
                facts[:, 39+kind_index] = np.log1p(reciprocal)
                facts[:, 41+kind_index] = (
                    out_events / denom[kind] if denom[kind] else np.nan)
            values = published[[runtime.index[uid] for uid in neighbors]]
            finite = np.isfinite(values)
            support = keep @ finite.astype(np.int32)
            sums = keep @ np.where(finite, values, 0.0)
            means = sums / np.maximum(support, 1)
            means[support == 0] = np.nan
            facts[:, 43:74] = means.astype(np.float32)
            design = encoded(runtime, facts, support)
            probabilities = expit(np.asarray(runtime.model.booster_.predict(
                design, raw_score=True), dtype=float))
            original = joint_by_target[target]["probability_before"]
            all_removed = joint_by_target[target]["probability_after_joint_source"]
            if (abs(float(probabilities[0])-original) > 1e-12 or
                    abs(float(probabilities[-1])-all_removed) > 1e-12):
                raise ValueError("empty/full coalition does not replay source certificate")
            for j, uid in enumerate(neighbors):
                expected = by_target[target][uid]["target_probability_after"]
                if abs(float(probabilities[1 << j])-expected) > 1e-12:
                    raise ValueError("single-pair coalition does not replay source certificate")
            flips = ((probabilities >= runtime.threshold) !=
                     (probabilities[0] >= runtime.threshold))
            if not flips[-1] or np.any(flips[1 << np.arange(n)]):
                raise ValueError("coalition endpoints contradict pair/joint certificates")
            sizes = removed.sum(axis=1)
            minimum = int(np.min(sizes[flips]))
            minimals = masks[flips & (sizes == minimum)].tolist()
            first = int(minimals[0])
            first_neighbors = [uid for j, uid in enumerate(neighbors)
                               if first & (1 << j)]
            raw_pointers = [loc for uid in first_neighbors
                            for loc in by_target[target][uid]["raw_relation_pointers"]]
            rows.append({
                "target_account_id": target,
                "native_neighbor_count": n,
                "subsets_evaluated": len(masks),
                "subsets_changing_class": int(np.count_nonzero(flips)),
                "minimum_source_pairs_changing_class": minimum,
                "minimum_size_coalition_count": len(minimals),
                "lowest_bitmask_minimal_neighbor_ids": first_neighbors,
                "first_minimal_raw_pointer_count": len(raw_pointers),
                "first_minimal_raw_pointers": raw_pointers,
                "probability_before": float(probabilities[0]),
                "probability_after_first_minimal": float(probabilities[first]),
                "probability_after_all_pairs": float(probabilities[-1]),
                "all_subset_probability_sha256": hashlib.sha256(
                    probabilities.astype("<f8").tobytes()).hexdigest(),
            })
        result_runs.append({
            "seed": seed,
            "exact_target_count": len(rows),
            "subsets_evaluated": sum(r["subsets_evaluated"] for r in rows),
            "min_source_pairs_over_targets": min(
                r["minimum_source_pairs_changing_class"] for r in rows),
            "max_source_pairs_over_targets": max(
                r["minimum_source_pairs_changing_class"] for r in rows),
            "targets": rows,
        })
        print(json.dumps({k: result_runs[-1][k] for k in (
            "seed", "exact_target_count", "subsets_evaluated",
            "min_source_pairs_over_targets", "max_source_pairs_over_targets")}),
            flush=True)
    result = {
        "status": "fox8_group_pivotal_small_ego_graph_exact_source_pair_coalitions",
        "selection": "all preselected neighbor-group class-flip judgments with at most 12 native neighbors",
        "operation": "for each subset, redirect all original reply/retweet target pointers between the target and selected neighbors to external IDs; preserve tweet/profile fields, rebuild native direct and neighbor facts, use same frozen model",
        "scope_limit": "minimality is within this source-pair redirection family and selected at-most-12-neighbor judgments; no claim of unique causal or semantic bot reason",
        "fixed_test_labels_read": False,
        "pair_certificate_sha256": sha256(PAIR),
        "joint_certificate_sha256": sha256(JOINT),
        "event_index_sha256": sha256(EVENTS),
        "max_exact_neighbors": MAX_EXACT_NEIGHBORS,
        "runs": result_runs,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
