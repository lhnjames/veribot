"""Independently replay every minimum-size Fox8 raw-pointer alternative."""

from __future__ import annotations

from collections import Counter
import json
import math
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
ALTERNATIVES = ROOT / "research/certificates/fox8_minimal_atomic_pointer_alternatives.json"
BASE = ROOT / "research/certificates/fox8_minimal_atomic_pointer_coalitions.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_minimal_atomic_pointer_alternatives_readback.json"


def locator_key(pointer: dict) -> tuple:
    return (pointer["source_account_id"], pointer["old_target_account_id"],
            pointer["relation"], pointer["source_line"], pointer["tweet_index"])


def run() -> dict:
    alt = json.loads(ALTERNATIVES.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    if (alt["base_certificate_sha256"] != sha256(BASE) or
            alt["event_index_sha256"] != sha256(EVENTS) or
            alt["fixed_test_labels_read"] is not False):
        raise ValueError("Alternative certificate binding changed")
    indexed = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    original_counts = Counter()
    event_by_locator = {}
    for event in indexed:
        key = (str(event["source_id"]), str(event["target_id"]),
               str(event["relation"]))
        original_counts[key] += len(event["locators"])
        for item in event["locators"]:
            loc = (key[0], key[1], key[2], int(item["line"]),
                   int(item["tweet_index"]))
            if loc in event_by_locator:
                raise ValueError("Duplicate original event locator")
            event_by_locator[loc] = key
    account_ids = {row["target_account_id"] for seed in alt["runs"]
                   for row in seed["rows"]}
    account_ids.update(p["source_account_id"] for seed in alt["runs"]
                       for row in seed["rows"]
                       for p in row["raw_pointer_locators_in_mask_order"])
    raw_rows = selected_raw_rows(account_ids)
    results = []
    for alt_run, base_run in zip(alt["runs"], base["runs"], strict=True):
        if alt_run["seed"] != base_run["seed"]:
            raise ValueError("Seed sequence changed")
        runtime = Fox8NeighborAtomicRuntime(ROOT, alt_run["seed"])
        table = pd.read_csv(runtime.features_path)
        if table["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("Feature source population changed")
        published = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        known = set(runtime.ids)
        max_error = 0.0
        verified = 0
        controls_verified = 0
        checked_unique_locators = set()
        multiple = 0
        no_mandatory = 0
        partial_pair_minimals = 0
        for row, first in zip(alt_run["rows"], base_run["rows"], strict=True):
            target = row["target_account_id"]
            if target != first["target_account_id"]:
                raise ValueError("Target sequence changed")
            inventory = row["raw_pointer_locators_in_mask_order"]
            m = len(inventory)
            if m != row["raw_pointer_count"] or len(row["all_minimal_bitmasks"]) != \
                    row["minimum_size_class_changing_coalitions"]:
                raise ValueError("Alternative inventory size changed")
            if (row["same_size_candidate_coalitions"] != math.comb(
                    m, row["minimum_raw_pointers_changing_class"]) or
                    row["same_size_nonchanging_coalitions"] !=
                    row["same_size_candidate_coalitions"] -
                    row["minimum_size_class_changing_coalitions"]):
                raise ValueError("Equal-size candidate partition differs")
            keys = [locator_key(p) for p in inventory]
            if len(set(keys)) != m or any(key not in event_by_locator for key in keys):
                raise ValueError("Alternative raw event locator missing or reused")
            for p, key in zip(inventory, keys, strict=True):
                line, line_hash, source = raw_rows[p["source_account_id"]]
                if line != p["source_line"] or line_hash != p["source_line_sha256"]:
                    raise ValueError("Raw source line changed")
                tweet = source["user_tweets"][p["tweet_index"]]
                if (p["relation"], p["old_target_account_id"], True) not in pointers(
                        tweet, p["source_account_id"], known):
                    raise ValueError("Raw tweet does not contain pointer")
                checked_unique_locators.add(key)
            _, _, source_target = raw_rows[target]
            denominator = Counter(kind for tweet in source_target["user_tweets"]
                                  for kind, _, _ in pointers(tweet, target, known))
            idx = runtime.index[target]
            original_class = row["probability_before"] >= runtime.threshold
            if row["threshold"] != runtime.threshold or abs(
                    row["probability_before"]-first["probability_before"]) > 1e-12:
                raise ValueError("Baseline model response differs")
            intersection = (1 << m) - 1
            union = 0
            partial = 0
            group_bits = {}
            for bit, p in enumerate(inventory):
                peer = (p["old_target_account_id"] if p["source_account_id"] == target
                        else p["source_account_id"])
                group_bits.setdefault(peer, []).append(bit)
            if len(group_bits) != first["native_neighbor_count"]:
                raise ValueError("Raw pointer groups differ from graph neighbors")
            if len(set(row["all_minimal_bitmasks"])) != len(row["all_minimal_bitmasks"]):
                raise ValueError("Minimal masks repeat")
            def replay(mask: int) -> float:
                edited = original_counts.copy()
                for bit, key in enumerate(keys):
                    if mask & (1 << bit):
                        edited[event_by_locator[key]] -= 1
                        if edited[event_by_locator[key]] < 0:
                            raise ValueError("Edited event count negative")
                fractions = tuple(
                    (sum(n for (source, _, kind), n in edited.items()
                         if source == target and kind == relation) /
                     denominator[relation]) if denominator[relation] else np.nan
                    for relation in ("reply", "retweet"))
                facts = runtime.facts[idx].copy()
                facts[31:43] = native_values(target, edited, fractions)
                remaining_ids = {
                    peer for (source, dest, _), n in edited.items() if n > 0
                    for peer in ([dest] if source == target else
                                 [source] if dest == target else [])}
                if not remaining_ids <= set(runtime.neighbors[idx]):
                    raise ValueError("Edited graph gained an unexpected neighbor")
                remaining = [runtime.index[uid] for uid in runtime.neighbors[idx]
                             if uid in remaining_ids]
                if remaining:
                    values = published[remaining]
                    supports = np.isfinite(values).sum(axis=0).astype(np.int32)
                    means = np.nansum(values, axis=0) / np.maximum(supports, 1)
                    means[supports == 0] = np.nan
                    facts[43:74] = means.astype(np.float32)
                else:
                    supports = np.zeros(31, dtype=np.int32)
                    facts[43:74] = np.nan
                matrix = encoded(runtime, facts.reshape(1, -1),
                                 supports.reshape(1, -1))
                return float(expit(runtime.model.booster_.predict(
                    matrix, raw_score=True)[0]))

            for mask, expected in zip(row["all_minimal_bitmasks"],
                                      row["all_minimal_probabilities"], strict=True):
                if mask <= 0 or mask >= (1 << m) or mask.bit_count() != \
                        row["minimum_raw_pointers_changing_class"]:
                    raise ValueError("Nonminimal mask size")
                intersection &= mask
                union |= mask
                if any(any(mask & (1 << bit) for bit in bits) and
                       not all(mask & (1 << bit) for bit in bits)
                       for bits in group_bits.values()):
                    partial += 1
                probability = replay(mask)
                max_error = max(max_error, abs(probability-expected))
                if abs(probability-expected) > 1e-12 or \
                        (probability >= runtime.threshold) == original_class:
                    raise ValueError("Independent alternative response differs")
                verified += 1
            control = row["closest_same_size_nonchanging_bitmask"]
            if control is None:
                if (row["same_size_nonchanging_coalitions"] != 0 or
                        row["closest_same_size_nonchanging_probability"] is not None or
                        row["pointer_swaps_from_first_changing_coalition"] is not None):
                    raise ValueError("Missing matched control fields disagree")
            else:
                if (control in row["all_minimal_bitmasks"] or
                        control.bit_count() != row["minimum_raw_pointers_changing_class"] or
                        ((control ^ row["all_minimal_bitmasks"][0]).bit_count() // 2) !=
                        row["pointer_swaps_from_first_changing_coalition"]):
                    raise ValueError("Matched control does not have the same edit budget")
                probability = replay(control)
                expected = row["closest_same_size_nonchanging_probability"]
                max_error = max(max_error, abs(probability-expected))
                if abs(probability-expected) > 1e-12 or \
                        (probability >= runtime.threshold) != original_class:
                    raise ValueError("Independent matched-size control differs")
                controls_verified += 1
            if [inventory[bit] for bit in range(m)
                if row["all_minimal_bitmasks"][0] & (1 << bit)] != \
                    first["first_minimal_pointer_locators"]:
                raise ValueError("First alternative differs from base certificate")
            if [inventory[bit] for bit in range(m) if intersection & (1 << bit)] != \
                    row["pointers_in_every_minimal_coalition"] or \
                    [inventory[bit] for bit in range(m) if union & (1 << bit)] != \
                    row["pointers_in_some_minimal_coalition"] or \
                    partial != row["minimal_coalitions_leaving_a_touched_pair_active"]:
                raise ValueError("Alternative intersection or pair status differs")
            multiple += len(row["all_minimal_bitmasks"]) > 1
            no_mandatory += intersection == 0
            partial_pair_minimals += partial
        results.append({"seed": alt_run["seed"],
                        "targets": len(alt_run["rows"]),
                        "minimum_coalitions_replayed": verified,
                        "matched_size_nonchanging_controls_replayed": controls_verified,
                        "unique_raw_locators_checked": len(checked_unique_locators),
                        "targets_with_multiple_minimum_coalitions": multiple,
                        "targets_without_mandatory_raw_pointer": no_mandatory,
                        "minimum_coalitions_with_partial_touched_pair":
                            partial_pair_minimals,
                        "maximum_probability_difference": max_error})
        print(results[-1], flush=True)
    report = {"status": "all_minimum_size_fox8_pointer_alternatives_independently_rebuilt",
              "source_certificate_sha256": sha256(ALTERNATIVES),
              "base_certificate_sha256": sha256(BASE),
              "event_index_sha256": sha256(EVENTS),
              "fixed_test_labels_read": False,
              "runs": results}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    run()
