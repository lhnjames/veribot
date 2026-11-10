"""Trace two nonadditive Fox8 decisions through actual sampled-tweet removal.

The two accounts were selected by the pre-existing additivity disagreement
certificate. Every eligible nonfirst tweet without a native reply/retweet
target is removed once, and all linked neighbor means are rebuilt. This is a
defined source-deletion sensitivity experiment, not a known-error correction.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, TWEET_FIELDS, tweet_features
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
INTERACTION_CERT = ROOT / "research/certificates/best_group_additivity_scope.json"
OUTPUT = ROOT / "research/certificates/fox8_interaction_raw_tweet_deletions.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_raw_rows(ids: set[str]) -> dict[str, tuple[int, str, dict]]:
    found = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            account_id = str(row["user_id"])
            if account_id in ids and account_id not in found:
                found[account_id] = (line_number, hashlib.sha256(line.encode()).hexdigest(), row)
            if len(found) == len(ids):
                break
    if set(found) != ids:
        raise ValueError("raw account record missing")
    return found


def design(runtime: Fox8NeighborAtomicRuntime, facts: np.ndarray,
           indices: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts[indices])
    reliability = batch.reliability.copy()
    reliability[:, 43:74] *= runtime.supports[indices] / (
        runtime.supports[indices] + 5.0)
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def predict(runtime: Fox8NeighborAtomicRuntime, facts: np.ndarray,
            indices: np.ndarray) -> np.ndarray:
    return expit(np.asarray(runtime.model.booster_.predict(
        design(runtime, facts, indices), raw_score=True), dtype=float))


def run() -> dict:
    earlier = json.loads(INTERACTION_CERT.read_text(encoding="utf-8"))
    selected = [(r["seed"], c["id"]) for r in earlier["runs"]
                if r["dataset"] == "fox8"
                for c in r["additive_approximation_disagreement_cases"]]
    if len(selected) != 2:
        raise ValueError("preselected two class-disagreement cases changed")
    raw_rows = selected_raw_rows({account_id for _, account_id in selected})
    cases = []
    for seed, account_id in selected:
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        target = runtime.index[account_id]
        line_number, raw_line_sha, row = raw_rows[account_id]
        tweets = row.get("user_tweets") or []
        original_raw_tweets = np.asarray(
            [tweet_features(tweets)[name] for name in TWEET_FIELDS], dtype=np.float32)
        if not np.allclose(original_raw_tweets, runtime.facts[target, 23:31],
                           rtol=1e-6, atol=1e-6):
            raise ValueError("original sampled tweets do not reconstruct model facts")
        frame = pd.read_csv(runtime.features_path)
        if frame["user_id"].astype(str).tolist() != runtime.ids:
            raise ValueError("published feature order changed")
        published = frame[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        adjacency = [runtime.index[uid] for uid in runtime.neighbors[target]]
        impacted = np.asarray([target, *adjacency], dtype=int)
        for adjacent in adjacency:
            peers = [runtime.index[uid] for uid in runtime.neighbors[adjacent]]
            values = published[peers]
            counts = np.isfinite(values).sum(axis=0)
            means = np.nansum(values, axis=0) / np.maximum(counts, 1)
            means[counts == 0] = np.nan
            if (not np.array_equal(counts, runtime.supports[adjacent]) or
                    not np.array_equal(means.astype(np.float32),
                                       runtime.facts[adjacent, 43:74],
                                       equal_nan=True)):
                raise ValueError("original native neighbor means did not rebuild")
        baseline_p = predict(runtime, runtime.facts, impacted)
        saved = runtime.predict(account_id)["probability"]
        if abs(float(baseline_p[0]) - saved) > 1e-12:
            raise ValueError("original frozen target score did not replay")
        mask_scores = np.empty(16, dtype=float)
        original_target_design = design(runtime, runtime.facts,
                                        np.asarray([target]))
        for mask in range(16):
            modified = original_target_design.copy()
            for j, (start, stop) in enumerate(((0, 23), (23, 31),
                                                (31, 43), (43, 74))):
                if not mask & (1 << j):
                    modified[:, start:stop] = 0
                    modified[:, start+74:stop+74] = 0
            mask_scores[mask] = float(expit(runtime.model.booster_.predict(
                modified, raw_score=True)[0]))
        records = []
        for position in range(1, len(tweets)):
            tweet = tweets[position]
            if (tweet.get("in_reply_to_user_id_str") or
                    tweet.get("in_reply_to_user_id") or
                    isinstance(tweet.get("retweeted_status"), dict)):
                continue
            remaining = tweets[:position] + tweets[position+1:]
            new_values = np.asarray([tweet_features(remaining)[name]
                                     for name in TWEET_FIELDS], dtype=np.float32)
            updated = runtime.facts.copy()
            updated[target, 23:31] = new_values
            published_updated = published.copy()
            published_updated[target, 23:31] = new_values.astype(float)
            for adjacent in adjacency:
                peers = [runtime.index[uid] for uid in runtime.neighbors[adjacent]]
                values = published_updated[peers]
                counts = np.isfinite(values).sum(axis=0)
                if not np.array_equal(counts, runtime.supports[adjacent]):
                    raise ValueError("native neighbor support changed unexpectedly")
                means = np.nansum(values, axis=0) / np.maximum(counts, 1)
                means[counts == 0] = np.nan
                updated[adjacent, 43:74] = means.astype(np.float32)
            after_p = predict(runtime, updated, impacted)
            target_matrix = design(runtime, updated, np.asarray([target]))
            after_subsets = np.empty(16, dtype=float)
            for mask in range(16):
                modified = target_matrix.copy()
                for j, (start, stop) in enumerate(((0, 23), (23, 31),
                                                    (31, 43), (43, 74))):
                    if not mask & (1 << j):
                        modified[:, start:stop] = 0
                        modified[:, start+74:stop+74] = 0
                after_subsets[mask] = float(expit(runtime.model.booster_.predict(
                    modified, raw_score=True)[0]))
            if not np.array_equal(after_subsets[[m for m in range(16) if not m & 2]],
                                  mask_scores[[m for m in range(16) if not m & 2]]):
                raise ValueError("nonactivity subsets changed despite only tweet-aggregate edit")
            original_logit = logit(mask_scores)
            after_logit = logit(after_subsets)
            interaction_before = float(original_logit[-1] - original_logit[0] -
                                       sum(original_logit[1 << j]-original_logit[0]
                                           for j in range(4)))
            interaction_after = float(after_logit[-1] - after_logit[0] -
                                      sum(after_logit[1 << j]-after_logit[0]
                                          for j in range(4)))
            records.append({
                "raw_tweet_locator": f"{RAW}#line={line_number};tweet={position}",
                "tweet_id": str(tweet.get("id_str") or tweet.get("id") or position),
                "removed_tweet_position": position,
                "changed_aggregate_fields": {
                    TWEET_FIELDS[j]: {"before": float(original_raw_tweets[j]),
                                      "after": float(new_values[j])}
                    for j in range(8) if original_raw_tweets[j] != new_values[j]},
                "target_bot_probability_before": float(baseline_p[0]),
                "target_bot_probability_after": float(after_p[0]),
                "target_class_changes": bool((after_p[0] >= runtime.threshold) !=
                                             (baseline_p[0] >= runtime.threshold)),
                "target_interaction_logit_before": interaction_before,
                "target_interaction_logit_after": interaction_after,
                "affected_neighbor_accounts": len(adjacency),
                "affected_neighbor_probability_changes": int(np.count_nonzero(
                    np.abs(after_p[1:] - baseline_p[1:]) > 1e-12)),
                "affected_neighbor_class_changes": int(np.count_nonzero(
                    (after_p[1:] >= runtime.threshold) !=
                    (baseline_p[1:] >= runtime.threshold))),
            })
        cases.append({
            "seed": seed, "account_id": account_id,
            "raw_record_line": line_number,
            "raw_record_line_sha256": raw_line_sha,
            "sampled_tweet_count": len(tweets),
            "eligible_nonfirst_no_native_target_tweets": len(records),
            "target_class_changes": sum(x["target_class_changes"] for x in records),
            "target_nonzero_probability_responses": sum(
                abs(x["target_bot_probability_after"] -
                    x["target_bot_probability_before"]) > 1e-12 for x in records),
            "neighbor_class_change_events": sum(
                x["affected_neighbor_class_changes"] for x in records),
            "neighbor_count": len(adjacency),
            "records": records,
        })
        print(seed, account_id, len(records), cases[-1]["target_class_changes"],
              cases[-1]["neighbor_class_change_events"], flush=True)
    result = {
        "status": "raw_sampled_tweet_deletion_replayed_through_same_frozen_fox8_model",
        "selection": "both saved Fox8 cases where additive group approximation reverses the frozen class",
        "operation": "remove every eligible nonfirst sampled tweet with no native reply/retweet target, rebuild own eight aggregates and all incident accounts' neighbor means",
        "scope_limit": "source deletion is a defined hypothetical edit, not a known correction; tweets with native targets require a separate graph rebuild",
        "fixed_test_labels_read": False,
        "raw_archive_sha256": sha256(RAW),
        "prior_interaction_certificate_sha256": sha256(INTERACTION_CERT),
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
