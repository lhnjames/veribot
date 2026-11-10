"""Replay all incoming raw-tweet sources for one nonadditive Fox8 decision.

The target was fixed by the prior additive-class disagreement, not selected
after inspecting tweet responses. Each of 74 incident raw tweet locators is
deleted once while its separate embedded profile snapshot is retained.
"""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_interaction_all_tweet_deletions import (
    EVENTS, GROUPS, INTERACTION_CERT, RAW, ROOT, interaction, native_values,
    pointers, selected_raw_rows, sha256, subset_probabilities,
)
from audit_fox8_interaction_raw_tweet_deletions import design
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, TWEET_FIELDS, tweet_features
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


OUTPUT = ROOT / "research/certificates/fox8_interaction_incoming_tweet_deletions.json"


def run() -> dict:
    earlier = json.loads(INTERACTION_CERT.read_text(encoding="utf-8"))
    selected = [(r["seed"], c["id"]) for r in earlier["runs"]
                if r["dataset"] == "fox8"
                for c in r["additive_approximation_disagreement_cases"]]
    selected = [(seed, uid) for seed, uid in selected if seed == 2]
    if len(selected) != 1:
        raise ValueError("seed2 nonadditive case changed")
    seed, target_uid = selected[0]
    runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
    target_index = runtime.index[target_uid]
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))
    event_count = Counter({
        (e["source_id"], e["target_id"], e["relation"]): e["count"]
        for e in event_index["events"]})
    incoming = [e for e in event_index["events"] if e["target_id"] == target_uid]
    operations = sorted({
        (e["source_id"], int(loc["line"]), int(loc["tweet_index"]))
        for e in incoming for loc in e["locators"]})
    if len(operations) != 74 or len(incoming) != 39:
        raise ValueError("incident source population changed")
    raw_rows = selected_raw_rows({source for source, _, _ in operations})
    table = pd.read_csv(runtime.features_path)
    if table["user_id"].astype(str).tolist() != runtime.ids:
        raise ValueError("published feature order changed")
    published = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    for source_uid, _, _ in operations:
        _, _, source_row = raw_rows[source_uid]
        source_tweets = source_row.get("user_tweets") or []
        all_by_kind = Counter(kind for current in source_tweets
                              for kind, _, _ in pointers(current, source_uid,
                                                         set(runtime.ids)))
        fractions = []
        for kind in ("reply", "retweet"):
            internal = sum(count for (src, _, k), count in event_count.items()
                           if src == source_uid and k == kind)
            fractions.append(internal / all_by_kind[kind]
                             if all_by_kind[kind] else np.nan)
        observed_native = native_values(
            source_uid, event_count, (fractions[0], fractions[1]))
        if not np.allclose(observed_native,
                           runtime.facts[runtime.index[source_uid], 31:43],
                           rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError("source native facts do not rebuild before deletion")
    target_original = subset_probabilities(runtime, design(
        runtime, runtime.facts, np.asarray([target_index])))
    original_p = float(target_original[-1])
    if abs(original_p-runtime.predict(target_uid)["probability"]) > 1e-12:
        raise ValueError("frozen target did not replay")
    known = set(runtime.ids)
    rows = []
    for source_uid, line_number, position in operations:
        source_index = runtime.index[source_uid]
        raw_line, line_sha, record = raw_rows[source_uid]
        if raw_line != line_number:
            raise ValueError("event locator does not point to first source record")
        tweets = record.get("user_tweets") or []
        tweet = tweets[position]
        all_pointers = pointers(tweet, source_uid, known)
        if not any(peer == target_uid and internal
                   for _, peer, internal in all_pointers):
            raise ValueError("incident tweet lost its target pointer")
        old_tweet_features = tweet_features(tweets)
        if not np.allclose(
                [old_tweet_features[name] for name in TWEET_FIELDS],
                published[source_index, 23:31], rtol=1e-6, atol=1e-6):
            raise ValueError("raw source tweet aggregates differ")
        updated_count = event_count.copy()
        touched = {source_uid}
        internal_removed = []
        for kind, peer, internal in all_pointers:
            if not internal:
                continue
            key = (source_uid, peer, kind)
            event = next((e for e in event_index["events"]
                          if (e["source_id"], e["target_id"], e["relation"]) == key), None)
            if (event is None or not any(int(loc["line"]) == line_number and
                                          int(loc["tweet_index"]) == position
                                          for loc in event["locators"])
                    or updated_count[key] <= 0):
                raise ValueError("raw tweet pointer absent from native event index")
            updated_count[key] -= 1
            touched.add(peer)
            internal_removed.append({"relation": kind, "target_id": peer,
                                     "remaining_typed_pair_count": updated_count[key]})
        replay_counts = Counter({
            (event["source_id"], event["target_id"], event["relation"]):
                sum(not (int(locator["line"]) == line_number and
                         int(locator["tweet_index"]) == position)
                    for locator in event["locators"])
            for event in event_index["events"]})
        if replay_counts != updated_count:
            raise ValueError("incremental typed events differ from locator-filtered full index")
        updated = runtime.facts.copy()
        updated_support = runtime.supports.copy()
        remaining = tweets[:position] + tweets[position+1:]
        new_tweet_features = tweet_features(remaining)
        updated[source_index, 23:31] = np.asarray(
            [new_tweet_features[name] for name in TWEET_FIELDS],
            dtype=np.float32)
        updated_published = published.copy()
        updated_published[source_index, 23:31] = updated[source_index, 23:31]
        source_targets = Counter(kind for current in tweets
                                 for kind, _, _ in pointers(current, source_uid, known))
        source_targets.subtract(kind for kind, _, _ in all_pointers)
        for uid in touched:
            idx = runtime.index[uid]
            if uid == source_uid:
                fractions = []
                for kind in ("reply", "retweet"):
                    internal = sum(count for (src, _, k), count in updated_count.items()
                                   if src == uid and k == kind)
                    fractions.append(internal / source_targets[kind]
                                     if source_targets[kind] else np.nan)
            else:
                fractions = runtime.facts[idx, 41:43]
            updated[idx, 31:43] = native_values(
                uid, updated_count, (fractions[0], fractions[1]))
        neighbor_sets = {uid: set(runtime.neighbors[runtime.index[uid]])
                         for uid in touched}
        removed_peers = []
        for pointer in internal_removed:
            peer = pointer["target_id"]
            if not any(count > 0 and {src, dst} == {source_uid, peer}
                       for (src, dst, _), count in updated_count.items()):
                neighbor_sets[source_uid].discard(peer)
                neighbor_sets[peer].discard(source_uid)
                removed_peers.append(peer)
        affected = {source_uid, *runtime.neighbors[source_index], *touched}
        for uid in affected:
            idx = runtime.index[uid]
            peers = neighbor_sets.get(uid, set(runtime.neighbors[idx]))
            if peers:
                values = updated_published[[runtime.index[peer] for peer in peers]]
                counts = np.isfinite(values).sum(axis=0)
                means = np.nansum(values, axis=0) / np.maximum(counts, 1)
                means[counts == 0] = np.nan
                updated[idx, 43:74] = means.astype(np.float32)
                updated_support[idx] = counts
            else:
                updated[idx, 43:74] = np.nan
                updated_support[idx] = 0
        affected_ids = sorted(affected)
        indices = np.asarray([runtime.index[uid] for uid in affected_ids])
        before_matrix = design(runtime, runtime.facts, indices)
        before_p = expit(np.asarray(runtime.model.booster_.predict(
            before_matrix, raw_score=True), dtype=float))
        evidence = runtime.reference.transform(updated[indices])
        reliability = evidence.reliability.copy()
        reliability[:, 43:74] *= updated_support[indices] / (
            updated_support[indices] + 5.0)
        after_matrix = np.column_stack((np.log1p(evidence.upper)*reliability,
                                        np.log1p(evidence.lower)*reliability))
        after_p = expit(np.asarray(runtime.model.booster_.predict(
            after_matrix, raw_score=True), dtype=float))
        target_position = affected_ids.index(target_uid)
        after_subsets = subset_probabilities(
            runtime, after_matrix[target_position:target_position+1])
        if (abs(float(before_p[target_position])-original_p) > 1e-12 or
                abs(float(after_p[target_position])-float(after_subsets[-1])) > 1e-12):
            raise ValueError("source-edited target score did not replay")
        others = np.arange(len(affected_ids)) != target_position
        rows.append({
            "source_account_id": source_uid,
            "source_line_sha256": line_sha,
            "raw_tweet_locator": f"{RAW}#line={line_number};tweet={position}",
            "tweet_id": str(tweet.get("id_str") or tweet.get("id") or position),
            "removed_targets": [{"relation": kind, "target_id": peer,
                                 "internal": internal}
                                for kind, peer, internal in all_pointers],
            "removed_native_graph_peers": removed_peers,
            "changed_target_model_facts": [
                {"fact": runtime.reference.names[j],
                 "before": float(runtime.facts[target_index, j])
                    if np.isfinite(runtime.facts[target_index, j]) else None,
                 "after": float(updated[target_index, j])
                    if np.isfinite(updated[target_index, j]) else None}
                for j in range(74)
                if not np.array_equal(runtime.facts[target_index, j:j+1],
                                      updated[target_index, j:j+1],
                                      equal_nan=True)],
            "changed_target_neighbor_support_fields": [
                {"fact": runtime.reference.names[43+j],
                 "before": int(runtime.supports[target_index, j]),
                 "after": int(updated_support[target_index, j])}
                for j in range(31)
                if runtime.supports[target_index, j] !=
                   updated_support[target_index, j]],
            "affected_account_count": len(affected),
            "target_probability_before": original_p,
            "target_probability_after": float(after_p[target_position]),
            "target_class_changes": bool((after_p[target_position] >= runtime.threshold) !=
                                         (original_p >= runtime.threshold)),
            "target_interaction_logit_before": interaction(target_original),
            "target_interaction_logit_after": interaction(after_subsets),
            "other_account_probability_change_count": int(np.count_nonzero(
                np.abs(before_p[others]-after_p[others]) > 1e-12)),
            "other_account_class_change_count": int(np.count_nonzero(
                (before_p[others] >= runtime.threshold) !=
                (after_p[others] >= runtime.threshold))),
        })
    result = {
        "status": "all_incident_raw_tweet_deletions_replayed_for_nonadditive_fox8_case",
        "selection": "seed2 account whose naive group-additive approximation reverses frozen class",
        "operation": "delete an incoming raw sampled tweet; preserve separate profile snapshot; rebuild source tweet aggregates, native typed events and graph, neighbor means/support, same frozen classifier",
        "scope_limit": "source deletion is hypothetical, not a known-error correction; 74 tweets from one preselected target do not estimate population-wide behavior",
        "fixed_test_labels_read": False,
        "seed": seed, "target_account_id": target_uid,
        "original_target_probability": original_p,
        "incoming_typed_pairs": len(incoming),
        "unique_incoming_raw_tweets": len(rows),
        "source_accounts": len({row["source_account_id"] for row in rows}),
        "target_probability_change_count": sum(
            abs(row["target_probability_after"]-original_p) > 1e-12 for row in rows),
        "target_class_change_count": sum(row["target_class_changes"] for row in rows),
        "native_graph_peer_removal_events": sum(
            len(row["removed_native_graph_peers"]) for row in rows),
        "other_account_class_change_events": sum(
            row["other_account_class_change_count"] for row in rows),
        "raw_archive_sha256": sha256(RAW),
        "native_event_index_sha256": sha256(EVENTS),
        "prior_interaction_certificate_sha256": sha256(INTERACTION_CERT),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                      encoding="utf-8")
    print(seed, target_uid, len(rows), result["target_probability_change_count"],
          result["target_class_change_count"],
          result["native_graph_peer_removal_events"], flush=True)
    return result


if __name__ == "__main__":
    run()
