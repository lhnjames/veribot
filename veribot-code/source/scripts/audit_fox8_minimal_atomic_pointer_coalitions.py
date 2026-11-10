"""Exact raw-pointer coalition search for small group-pivotal Fox8 judgments."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
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
MIN_PAIR = ROOT / "research/certificates/fox8_minimal_source_pair_coalitions.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_minimal_atomic_pointer_coalitions.json"
ALTERNATIVES = ROOT / "research/certificates/fox8_minimal_atomic_pointer_alternatives.json"
MAX_POINTERS = 14


def run() -> dict:
    pair = json.loads(PAIR.read_text(encoding="utf-8"))
    joint = json.loads(JOINT.read_text(encoding="utf-8"))
    min_pair = json.loads(MIN_PAIR.read_text(encoding="utf-8"))
    if (joint["pair_certificate_sha256"] != sha256(PAIR) or
            min_pair["pair_certificate_sha256"] != sha256(PAIR) or
            min_pair["joint_certificate_sha256"] != sha256(JOINT)):
        raise ValueError("source-pair certificate chain differs")
    target_ids = {x["target_account_id"] for r in min_pair["runs"]
                  for x in r["targets"]}
    raw_rows = selected_raw_rows(target_ids)
    known = None
    output_runs = []
    alternative_runs = []
    for seed in (0, 1, 2):
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        known = set(runtime.ids)
        frame = pd.read_csv(runtime.features_path)
        if frame["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("source account order changed")
        published = frame[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        pair_run = next(r for r in pair["runs"] if r["seed"] == seed)
        joint_run = next(r for r in joint["runs"] if r["seed"] == seed)
        min_run = next(r for r in min_pair["runs"] if r["seed"] == seed)
        by_target = defaultdict(dict)
        for item in pair_run["pair_operations"]:
            by_target[item["target_account_id"]][item["neighbor_account_id"]] = item
        joint_by_target = {x["target_account_id"]: x
                           for x in joint_run["target_operations"]}
        rows = []
        alternative_rows = []
        for old in min_run["targets"]:
            target = old["target_account_id"]
            target_i = runtime.index[target]
            neighbors = runtime.neighbors[target_i]
            n = len(neighbors)
            raw_pointers = []
            pair_bits = []
            effects = []
            initial = np.zeros((n, 4), dtype=np.int32)
            for neighbor_index, uid in enumerate(neighbors):
                item = by_target[target][uid]
                pointers_for_pair = item["raw_relation_pointers"]
                group_bits = []
                for pointer in pointers_for_pair:
                    if {pointer["source_account_id"],
                        pointer["old_target_account_id"]} != {target, uid}:
                        raise ValueError("raw pointer belongs to another pair")
                    relation = pointer["relation"]
                    kind_index = 0 if relation == "reply" else 2
                    if relation not in ("reply", "retweet"):
                        raise ValueError("unknown event type")
                    direction = 0 if pointer["source_account_id"] == target else 1
                    column = kind_index + direction
                    initial[neighbor_index, column] += 1
                    effect = np.zeros((n, 4), dtype=np.int32)
                    effect[neighbor_index, column] = 1
                    effects.append(effect.reshape(-1))
                    group_bits.append(len(raw_pointers))
                    raw_pointers.append(pointer)
                pair_bits.append(group_bits)
            m = len(raw_pointers)
            if m > MAX_POINTERS or m < n:
                raise ValueError("atomic-pointer exact scope changed")
            keys = [(p["source_account_id"], p["source_line"],
                     p["tweet_index"], p["relation"], p["old_target_account_id"])
                    for p in raw_pointers]
            if len(keys) != len(set(keys)):
                raise ValueError("raw pointer reused across atomic units")
            _, _, record = raw_rows[target]
            denominator = Counter(kind for tweet in record["user_tweets"]
                                  for kind, _, _ in pointers(tweet, target, known))
            masks = np.arange(1 << m, dtype=np.int32)
            removed = ((masks[:, None] & (1 << np.arange(m))) != 0).astype(np.int32)
            remaining = (initial.reshape(1, -1) -
                         removed @ np.asarray(effects, dtype=np.int32)).reshape(-1, n, 4)
            if np.any(remaining < 0):
                raise ValueError("removed more raw pointers than published")
            adjacency = (remaining.sum(axis=2) > 0).astype(np.int32)
            facts = np.repeat(runtime.facts[target_i:target_i+1], len(masks), axis=0)
            for kind_index, column_base in ((0, 0), (1, 2)):
                out = remaining[:, :, column_base]
                incoming = remaining[:, :, column_base+1]
                out_events = out.sum(axis=1)
                in_events = incoming.sum(axis=1)
                out_peers = (out > 0).sum(axis=1)
                in_peers = (incoming > 0).sum(axis=1)
                reciprocal = ((out > 0) & (incoming > 0)).sum(axis=1)
                j = 31 + 4 * kind_index
                facts[:, j:j+4] = np.stack((
                    np.log1p(out_events), np.log1p(in_events),
                    np.log1p(out_peers), np.log1p(in_peers)), axis=1)
                facts[:, 39+kind_index] = np.log1p(reciprocal)
                relation = "reply" if kind_index == 0 else "retweet"
                facts[:, 41+kind_index] = (
                    out_events / denominator[relation]
                    if denominator[relation] else np.nan)
            values = published[[runtime.index[uid] for uid in neighbors]]
            finite = np.isfinite(values)
            supports = adjacency @ finite.astype(np.int32)
            means = ((adjacency @ np.where(finite, values, 0.0)) /
                     np.maximum(supports, 1))
            means[supports == 0] = np.nan
            facts[:, 43:74] = means.astype(np.float32)
            matrix = encoded(runtime, facts, supports)
            probs = expit(np.asarray(runtime.model.booster_.predict(
                matrix, raw_score=True), dtype=float))
            joint_row = joint_by_target[target]
            if (abs(float(probs[0])-joint_row["probability_before"]) > 1e-12 or
                    abs(float(probs[-1])-
                        joint_row["probability_after_joint_source"]) > 1e-12):
                raise ValueError("atomic subset endpoints differ from source certificates")
            for neighbor_index, uid in enumerate(neighbors):
                full_pair_mask = sum(1 << bit for bit in pair_bits[neighbor_index])
                expected = by_target[target][uid]["target_probability_after"]
                if abs(float(probs[full_pair_mask])-expected) > 1e-12:
                    raise ValueError("whole-pair atomic subset does not replay")
            original_minimal_mask = 0
            for neighbor_index, uid in enumerate(neighbors):
                if uid in old["lowest_bitmask_minimal_neighbor_ids"]:
                    original_minimal_mask |= sum(1 << bit for bit in pair_bits[neighbor_index])
            if abs(float(probs[original_minimal_mask])-
                   old["probability_after_first_minimal"]) > 1e-12:
                raise ValueError("previous minimal pair coalition does not replay")
            flips = (probs >= runtime.threshold) != (probs[0] >= runtime.threshold)
            sizes = removed.sum(axis=1)
            if not flips[-1]:
                raise ValueError("full atomic removal should change class")
            min_size = int(np.min(sizes[flips]))
            minimals = masks[flips & (sizes == min_size)].tolist()
            first = int(minimals[0])
            equal_size_nonflips = masks[(~flips) & (sizes == min_size)].tolist()
            matched_nonflip = (min(equal_size_nonflips,
                                   key=lambda mask: ((int(mask) ^ first).bit_count(),
                                                     int(mask)))
                               if equal_size_nonflips else None)
            first_pointers = [raw_pointers[k] for k in range(m)
                              if first & (1 << k)]
            first_pair_count = sum(any(first & (1 << k) for k in bits)
                                   for bits in pair_bits)
            first_removed_pair_count = sum(all(first & (1 << k) for k in bits)
                                           for bits in pair_bits)
            intersection = (1 << m) - 1
            union = 0
            partial_pair_minimals = 0
            for mask in minimals:
                intersection &= mask
                union |= mask
                if any(any(mask & (1 << bit) for bit in bits) and
                       not all(mask & (1 << bit) for bit in bits)
                       for bits in pair_bits):
                    partial_pair_minimals += 1
            alternative_rows.append({
                "target_account_id": target,
                "raw_pointer_count": m,
                "minimum_raw_pointers_changing_class": min_size,
                "same_size_candidate_coalitions": math.comb(m, min_size),
                "minimum_size_class_changing_coalitions": len(minimals),
                "same_size_nonchanging_coalitions": len(equal_size_nonflips),
                "closest_same_size_nonchanging_bitmask":
                    int(matched_nonflip) if matched_nonflip is not None else None,
                "closest_same_size_nonchanging_probability":
                    float(probs[matched_nonflip]) if matched_nonflip is not None else None,
                "pointer_swaps_from_first_changing_coalition":
                    ((int(matched_nonflip) ^ first).bit_count() // 2
                     if matched_nonflip is not None else None),
                "raw_pointer_locators_in_mask_order": raw_pointers,
                "all_minimal_bitmasks": minimals,
                "all_minimal_probabilities": [float(probs[mask]) for mask in minimals],
                "pointers_in_every_minimal_coalition": [
                    raw_pointers[k] for k in range(m) if intersection & (1 << k)],
                "pointers_in_some_minimal_coalition": [
                    raw_pointers[k] for k in range(m) if union & (1 << k)],
                "minimal_coalitions_leaving_a_touched_pair_active":
                    partial_pair_minimals,
                "probability_before": float(probs[0]),
                "threshold": float(runtime.threshold),
            })
            rows.append({
                "target_account_id": target,
                "native_neighbor_count": n,
                "raw_pointer_count": m,
                "pointer_subsets_evaluated": len(masks),
                "pointer_subsets_changing_class": int(np.count_nonzero(flips)),
                "single_pointer_class_changes": int(np.count_nonzero(
                    flips[1 << np.arange(m)])),
                "minimum_raw_pointers_changing_class": min_size,
                "minimum_size_pointer_coalition_count": len(minimals),
                "first_minimal_pointer_locators": first_pointers,
                "first_minimal_touched_neighbor_pairs": first_pair_count,
                "first_minimal_fully_removed_neighbor_pairs": first_removed_pair_count,
                "first_minimal_probability": float(probs[first]),
                "probability_before": float(probs[0]),
                "probability_all_pointers_redirected": float(probs[-1]),
                "previous_minimum_pair_count": old["minimum_source_pairs_changing_class"],
            })
        output_runs.append({
            "seed": seed,
            "targets": len(rows),
            "pointer_subsets_evaluated": sum(x["pointer_subsets_evaluated"] for x in rows),
            "targets_with_single_pointer_class_change": sum(
                x["single_pointer_class_changes"] > 0 for x in rows),
            "targets_where_minimal_pointer_set_leaves_some_touched_pair_active": sum(
                x["first_minimal_touched_neighbor_pairs"] >
                x["first_minimal_fully_removed_neighbor_pairs"] for x in rows),
            "rows": rows,
        })
        alternative_runs.append({"seed": seed, "rows": alternative_rows})
        print(json.dumps({k: output_runs[-1][k] for k in (
            "seed", "targets", "pointer_subsets_evaluated",
            "targets_with_single_pointer_class_change",
            "targets_where_minimal_pointer_set_leaves_some_touched_pair_active")}),
            flush=True)
    result = {
        "status": "fox8_group_pivotal_small_ego_graph_exact_raw_pointer_coalitions",
        "selection": "same 69 at-most-12-neighbor judgments as exact pair-coalition certificate; each has at most 14 raw source pointers",
        "operation": "redirect selected individual raw reply/retweet target pointers to an external ID, preserving tweet/profile fields and frozen reference/model; recompute typed counts, graph adjacency, direct and neighbor facts",
        "scope_limit": "minimality is within this synthetic raw-pointer redirection family and selected judgments; no known-error, semantic or general-population claim",
        "fixed_test_labels_read": False,
        "pair_certificate_sha256": sha256(PAIR),
        "joint_certificate_sha256": sha256(JOINT),
        "minimal_pair_certificate_sha256": sha256(MIN_PAIR),
        "event_index_sha256": sha256(EVENTS),
        "maximum_raw_pointers": MAX_POINTERS,
        "runs": output_runs,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    ALTERNATIVES.write_text(json.dumps({
        "status": "all_minimum_size_raw_pointer_coalitions_for_exact_small_fox8_scope",
        "selection": result["selection"],
        "operation": result["operation"],
        "scope_limit": result["scope_limit"],
        "base_certificate_sha256": sha256(OUTPUT),
        "event_index_sha256": sha256(EVENTS),
        "fixed_test_labels_read": False,
        "runs": alternative_runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
