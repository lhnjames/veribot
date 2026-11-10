"""Enumerate every nonfirst raw tweet deletion for two nonadditive Fox8 cases.

Rebuild tweet aggregates, native typed-event facts, the induced undirected
native-neighbor set, neighbor means/support, and frozen raw_neighbor_74 scores.
Only the two cases identified by the prior class-reversing additive error are
examined; this is source sensitivity, not known-error correction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_interaction_raw_tweet_deletions import (
    INTERACTION_CERT, RAW, ROOT, design, selected_raw_rows, sha256,
)
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, TWEET_FIELDS, tweet_features
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_interaction_all_tweet_deletions.json"
GROUPS = ((0, 23), (23, 31), (31, 43), (43, 74))


def pointers(tweet: dict, source: str, known: set[str]) -> list[tuple[str, str, bool]]:
    result = []
    reply = tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
    if reply is not None:
        target = str(reply)
        result.append(("reply", target, target in known and target != source))
    retweet = tweet.get("retweeted_status")
    if isinstance(retweet, dict):
        author = retweet.get("user") or {}
        target = author.get("id_str") or author.get("id")
        if target is not None:
            target = str(target)
            result.append(("retweet", target, target in known and target != source))
    return result


def native_values(uid: str, counts: Counter, fractions: tuple[float, float]) -> np.ndarray:
    out_peers = defaultdict(set)
    in_peers = defaultdict(set)
    out_events = Counter()
    in_events = Counter()
    for (source, target, kind), count in counts.items():
        if count <= 0:
            continue
        if source == uid:
            out_peers[kind].add(target)
            out_events[kind] += count
        if target == uid:
            in_peers[kind].add(source)
            in_events[kind] += count
    return np.asarray([
        np.log1p(out_events["reply"]), np.log1p(in_events["reply"]),
        np.log1p(len(out_peers["reply"])), np.log1p(len(in_peers["reply"])),
        np.log1p(out_events["retweet"]), np.log1p(in_events["retweet"]),
        np.log1p(len(out_peers["retweet"])), np.log1p(len(in_peers["retweet"])),
        np.log1p(len(out_peers["reply"] & in_peers["reply"])),
        np.log1p(len(out_peers["retweet"] & in_peers["retweet"])),
        fractions[0], fractions[1],
    ], dtype=float)


def subset_probabilities(runtime: Fox8NeighborAtomicRuntime,
                         matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (1, 148):
        raise ValueError("target evidence width changed")
    batch = np.repeat(matrix, 16, axis=0)
    for mask in range(16):
        for j, (start, stop) in enumerate(GROUPS):
            if not mask & (1 << j):
                batch[mask, start:stop] = 0
                batch[mask, start+74:stop+74] = 0
    return expit(np.asarray(runtime.model.booster_.predict(
        batch, raw_score=True), dtype=float))


def interaction(p: np.ndarray) -> float:
    scores = logit(p)
    return float(scores[-1] - scores[0] - sum(
        scores[1 << j]-scores[0] for j in range(4)))


def run() -> dict:
    earlier = json.loads(INTERACTION_CERT.read_text(encoding="utf-8"))
    selected = [(r["seed"], c["id"]) for r in earlier["runs"]
                if r["dataset"] == "fox8"
                for c in r["additive_approximation_disagreement_cases"]]
    if len(selected) != 2:
        raise ValueError("preselected cases changed")
    raw_rows = selected_raw_rows({uid for _, uid in selected})
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))
    event_count = Counter({
        (e["source_id"], e["target_id"], e["relation"]): e["count"]
        for e in event_index["events"]})
    locator_lookup = {
        (e["source_id"], e["target_id"], e["relation"],
         int(locator["line"]), int(locator["tweet_index"]))
        for e in event_index["events"] for locator in e["locators"]}
    cases = []
    for seed, uid in selected:
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        line, line_sha, record = raw_rows[uid]
        tweets = record.get("user_tweets") or []
        original_tweet_feature_values = tweet_features(tweets)
        known = set(runtime.ids)
        target = runtime.index[uid]
        table = pd.read_csv(runtime.features_path)
        if table["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("source feature population changed")
        base_features = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        if not np.allclose(
                np.asarray([tweet_features(tweets)[name] for name in TWEET_FIELDS]),
                base_features[target, 23:31], rtol=1e-6, atol=1e-6):
            raise ValueError("raw tweet aggregates differ from model source")
        for check_uid in [uid, *runtime.neighbors[target]]:
            raw_fraction = []
            if check_uid == uid:
                check_tweets = tweets
                for kind in ("reply", "retweet"):
                    candidates = [p for tweet in check_tweets
                                  for p in pointers(tweet, check_uid, known)
                                  if p[0] == kind]
                    internal = sum(p[2] for p in candidates)
                    raw_fraction.append(internal / len(candidates)
                                        if candidates else np.nan)
            else:
                check_idx = runtime.index[check_uid]
                raw_fraction = runtime.facts[check_idx, 41:43]
            native = native_values(check_uid, event_count,
                                   (raw_fraction[0], raw_fraction[1]))
            check_idx = runtime.index[check_uid]
            if not np.allclose(native, runtime.facts[check_idx, 31:43],
                               rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError("baseline native facts do not rebuild")
        original_subsets = subset_probabilities(runtime, design(
            runtime, runtime.facts, np.asarray([target])))
        original_score = runtime.predict(uid)["probability"]
        if abs(float(original_subsets[-1])-original_score) > 1e-12:
            raise ValueError("original frozen decision did not replay")
        source_all = Counter()
        for tweet in tweets:
            source_all.update(kind for kind, _, _ in pointers(tweet, uid, known))
        rows = []
        for position in range(1, len(tweets)):
            tweet = tweets[position]
            removed = pointers(tweet, uid, known)
            changed_counts = event_count.copy()
            touched = {uid}
            internal_pointers = []
            for kind, peer, internal in removed:
                if not internal:
                    continue
                key = (uid, peer, kind)
                if ((uid, peer, kind, line, position) not in locator_lookup
                        or changed_counts[key] <= 0):
                    raise ValueError("raw tweet pointer absent from event index")
                changed_counts[key] -= 1
                touched.add(peer)
                internal_pointers.append({"relation": kind, "target_id": peer,
                                          "remaining_pair_events": changed_counts[key]})
            changed_all = source_all.copy()
            changed_all.subtract(kind for kind, _, _ in removed)
            updated = runtime.facts.copy()
            updated_support = runtime.supports.copy()
            after_tweets = tweets[:position] + tweets[position+1:]
            new_tweet_features = tweet_features(after_tweets)
            updated[target, 23:31] = np.asarray(
                [new_tweet_features[name] for name in TWEET_FIELDS],
                dtype=np.float32)
            updated_base = base_features.copy()
            updated_base[target, 23:31] = updated[target, 23:31]
            for changed_uid in touched:
                idx = runtime.index[changed_uid]
                if changed_uid == uid:
                    fractions = []
                    for kind in ("reply", "retweet"):
                        internal_count = sum(
                            count for (src, _, k), count in changed_counts.items()
                            if src == uid and k == kind)
                        fractions.append(internal_count / changed_all[kind]
                                         if changed_all[kind] else np.nan)
                else:
                    fractions = runtime.facts[idx, 41:43]
                updated[idx, 31:43] = native_values(
                    changed_uid, changed_counts,
                    (fractions[0], fractions[1]))
            neighbors = {account: set(runtime.neighbors[runtime.index[account]])
                         for account in touched}
            removed_pairs = []
            for pointer in internal_pointers:
                peer = pointer["target_id"]
                any_remaining = any(count > 0 and {src, dst} == {uid, peer}
                                    for (src, dst, _), count in changed_counts.items())
                if not any_remaining:
                    neighbors[uid].discard(peer)
                    neighbors[peer].discard(uid)
                    removed_pairs.append(peer)
            affected = {uid, *runtime.neighbors[target], *touched}
            for account in affected:
                idx = runtime.index[account]
                peers = neighbors.get(account, set(runtime.neighbors[idx]))
                peer_indices = [runtime.index[peer] for peer in peers]
                if peer_indices:
                    values = updated_base[peer_indices]
                    counts = np.isfinite(values).sum(axis=0)
                    means = np.nansum(values, axis=0) / np.maximum(counts, 1)
                    means[counts == 0] = np.nan
                    updated[idx, 43:74] = means.astype(np.float32)
                    updated_support[idx, :] = counts
                else:
                    updated[idx, 43:74] = np.nan
                    updated_support[idx, :] = 0
            affected_indices = np.asarray([runtime.index[account]
                                           for account in sorted(affected)], dtype=int)
            # The reference remains frozen; support may change if the edge vanishes.
            evidence = runtime.reference.transform(updated[affected_indices])
            reliability = evidence.reliability.copy()
            reliability[:, 43:74] *= updated_support[affected_indices] / (
                updated_support[affected_indices] + 5.0)
            matrix = np.column_stack((np.log1p(evidence.upper)*reliability,
                                      np.log1p(evidence.lower)*reliability))
            after_p = expit(np.asarray(runtime.model.booster_.predict(
                matrix, raw_score=True), dtype=float))
            baseline_matrix = design(runtime, runtime.facts, affected_indices)
            before_p = expit(np.asarray(runtime.model.booster_.predict(
                baseline_matrix, raw_score=True), dtype=float))
            target_position = sorted(affected).index(uid)
            target_after_subsets = subset_probabilities(
                runtime, matrix[target_position:target_position+1])
            if abs(float(after_p[target_position])-float(target_after_subsets[-1])) > 1e-12:
                raise ValueError("full source-edited model differs from subset replay")
            outside = np.arange(len(affected_indices)) != target_position
            rows.append({
                "raw_tweet_locator": f"{RAW}#line={line};tweet={position}",
                "tweet_id": str(tweet.get("id_str") or tweet.get("id") or position),
                "removed_tweet_position": position,
                "removed_targets": [{"relation": kind, "target_id": peer,
                                     "internal": internal}
                                    for kind, peer, internal in removed],
                "removed_native_graph_peers": removed_pairs,
                "changed_source_tweet_aggregates": {
                    name: {"before": float(original_tweet_feature_values[name]),
                           "after": float(new_tweet_features[name])}
                    for name in TWEET_FIELDS
                    if original_tweet_feature_values[name] != new_tweet_features[name]},
                "changed_target_model_facts": [
                    {"fact": runtime.reference.names[j],
                     "before": float(runtime.facts[target, j])
                         if np.isfinite(runtime.facts[target, j]) else None,
                     "after": float(updated[target, j])
                         if np.isfinite(updated[target, j]) else None}
                    for j in range(74)
                    if not np.array_equal(runtime.facts[target, j:j+1],
                                          updated[target, j:j+1], equal_nan=True)],
                "affected_account_count": len(affected),
                "target_probability_before": float(before_p[target_position]),
                "target_probability_after": float(after_p[target_position]),
                "target_class_changes": bool((before_p[target_position] >= runtime.threshold) !=
                                             (after_p[target_position] >= runtime.threshold)),
                "target_interaction_logit_before": interaction(original_subsets),
                "target_interaction_logit_after": interaction(target_after_subsets),
                "other_account_probability_change_count": int(np.count_nonzero(
                    np.abs(before_p[outside]-after_p[outside]) > 1e-12)),
                "other_account_class_change_count": int(np.count_nonzero(
                    (before_p[outside] >= runtime.threshold) !=
                    (after_p[outside] >= runtime.threshold))),
            })
        cases.append({
            "seed": seed, "account_id": uid,
            "raw_record_line": line, "raw_record_line_sha256": line_sha,
            "sampled_tweet_count": len(tweets),
            "nonfirst_tweets_evaluated": len(rows),
            "target_probability_change_count": sum(
                abs(r["target_probability_after"]-r["target_probability_before"]) > 1e-12
                for r in rows),
            "target_class_change_count": sum(r["target_class_changes"] for r in rows),
            "other_account_class_change_events": sum(
                r["other_account_class_change_count"] for r in rows),
            "native_graph_peer_removal_events": sum(
                len(r["removed_native_graph_peers"]) for r in rows),
            "rows": rows,
        })
        print(seed, uid, len(rows), cases[-1]["target_class_change_count"],
              cases[-1]["other_account_class_change_events"], flush=True)
    result = {
        "status": "all_nonfirst_sampled_tweet_deletions_replayed_for_two_fox8_interaction_cases",
        "selection": "both fixed-test accounts where a naive additive explanation reverses the frozen class",
        "operation": "delete one sampled raw tweet; preserve first embedded profile snapshot; rebuild tweet aggregates, typed native facts, graph neighbor sets, neighbor means/support and same frozen model",
        "scope_limit": "defined source-deletion sensitivity, not known-error correction; two selected accounts cannot estimate population-wide source-dependence rates",
        "fixed_test_labels_read": False,
        "raw_archive_sha256": sha256(RAW),
        "native_event_index_sha256": sha256(EVENTS),
        "prior_interaction_certificate_sha256": sha256(INTERACTION_CERT),
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
