"""Known raw tweet-aggregate join repair through frozen best Fox8 model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.fox8_adapter import (FEATURE_FIELDS, index_sources,
                                         raw_feature_matrix)
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/certificates/fox8_best_source_join_correction.json"
FOLDER = ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity"
TWEET_COLUMNS = tuple(range(23, 31))
NEIGHBOR_TWEET_COLUMNS = tuple(range(43 + 23, 43 + 31))
RECIPIENT_SALT = "veribot-fox8-best-tweet-recipient-v1"
DONOR_SALT = "veribot-fox8-best-tweet-donor-v1"


def order(ids: list[str], salt: str) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"{salt}:{ids[i]}".encode()).digest())


def neighbor_tweet_means(runtime: Fox8NeighborAtomicRuntime,
                         raw: np.ndarray) -> np.ndarray:
    means = np.full((len(runtime.ids), len(TWEET_COLUMNS)), np.nan,
                    dtype=np.float32)
    for i, adjacent_ids in enumerate(runtime.neighbors):
        if not adjacent_ids:
            continue
        selected = raw[[runtime.index[uid] for uid in adjacent_ids],
                       23:31]
        valid = np.isfinite(selected)
        counts = valid.sum(axis=0)
        values = np.nansum(selected, axis=0) / np.maximum(counts, 1)
        values[counts == 0] = np.nan
        means[i] = values.astype(np.float32)
    return means


def facts_from_join(runtime: Fox8NeighborAtomicRuntime,
                    raw: np.ndarray) -> np.ndarray:
    facts = runtime.facts.copy()
    facts[:, TWEET_COLUMNS] = raw[:, 23:31].astype(np.float32)
    facts[:, NEIGHBOR_TWEET_COLUMNS] = neighbor_tweet_means(runtime, raw)
    return facts


def predict_facts(runtime: Fox8NeighborAtomicRuntime,
                  facts: np.ndarray, test: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts[test])
    reliability = batch.reliability.copy()
    reliability[:, 43:74] *= runtime.supports[test] / (
        runtime.supports[test] + 5.0)
    matrix = np.column_stack((np.log1p(batch.upper) * reliability,
                              np.log1p(batch.lower) * reliability))
    return runtime.model.predict_proba(matrix)[:, 1]


def evaluate_seed(seed: int) -> dict:
    runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
    _, labels, ids, _, split, source = load(
        "fox8", runtime.data_dir, None, seed)
    saved_path = FOLDER / f"seed{seed}/test_predictions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    manifest = saved["manifest"]
    if (ids != runtime.ids or
            source["split_sha256"] !=
            runtime.lock["split_sha256_by_seed"][str(seed)] or
            manifest["split_sha256"] != source["split_sha256"] or
            manifest["threshold"] != runtime.threshold or
            manifest["reference_version"] != runtime.reference.version or
            manifest["model_sha256"] != sha256_file(
                runtime.run_dir / "model.joblib")):
        raise ValueError("Fox8 frozen source, split, or model changed")
    if tuple(FEATURE_FIELDS[j] for j in TWEET_COLUMNS) != (
            "n_tweets", "retweet_ratio", "reply_ratio", "avg_tweet_len",
            "avg_favorites", "avg_retweets", "avg_score",
            "unique_lang_ratio"):
        raise ValueError("sampled tweet fact vocabulary differs")
    table = pd.read_csv(runtime.features_path)
    raw = table[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    if (table["user_id"].astype(str).tolist() != runtime.ids or
            not np.array_equal(raw[:, 23:31].astype(np.float32),
                               runtime.facts[:, 23:31], equal_nan=True)):
        raise ValueError("published Fox8 tweet aggregate facts differ")
    clean_facts = facts_from_join(runtime, raw)
    if not np.array_equal(clean_facts, runtime.facts, equal_nan=True):
        raise ValueError("native neighbor means did not reconstruct clean facts")
    test = split["test"]
    test_ids = [ids[int(i)] for i in test]
    recorded = saved["predictions"]
    if ([row["account_id"] for row in recorded] != test_ids or
            [row["label"] for row in recorded] != labels[test].tolist()):
        raise ValueError("saved Fox8 test rows differ")
    clean_p = predict_facts(runtime, clean_facts, test)
    if np.max(np.abs(clean_p - np.asarray([
            row["probability"] for row in recorded]))) > 1e-12:
        raise ValueError("clean Fox8 facts did not replay frozen probabilities")
    recipient_order = order(test_ids, RECIPIENT_SALT)
    donor_order = order(test_ids, DONOR_SALT)
    recipients = recipient_order[:round(len(test) * 0.25)]
    corrupted_raw = raw.copy()
    donor_for: dict[int, int] = {}
    for rank, recipient in enumerate(recipients):
        donor = next((candidate for candidate in
                      donor_order[rank:] + donor_order[:rank]
                      if candidate != recipient and not np.array_equal(
                          raw[test[candidate], 23:31],
                          raw[test[recipient], 23:31], equal_nan=True)), None)
        if donor is None:
            raise ValueError("no distinct raw tweet donor")
        corrupted_raw[test[recipient], 23:31] = raw[test[donor], 23:31]
        donor_for[recipient] = donor
    selected_ids = sorted({test_ids[i] for i in recipients} |
                          {test_ids[i] for i in donor_for.values()})
    source_records = index_sources(runtime.raw_path, selected_ids)
    raw_rebuilt = raw_feature_matrix(source_records, selected_ids)
    for i, uid in enumerate(selected_ids):
        if not np.allclose(raw_rebuilt[i, 23:31],
                           raw[runtime.index[uid], 23:31].astype(np.float32),
                           rtol=1e-6, atol=1e-6, equal_nan=True):
            raise ValueError("sampled raw tweets do not match derived facts")
    locators = {}
    for record in source_records.values():
        if record.kind == "tweet" and record.source_id.endswith(":0"):
            locators[record.account_id] = record.raw_location
    if set(locators) != set(selected_ids):
        raise ValueError("sampled raw account locator is missing")
    corrupt_facts = facts_from_join(runtime, corrupted_raw)
    corrupt_p = predict_facts(runtime, corrupt_facts, test)
    neighbor_cell_change = ~np.isclose(
        corrupt_facts[:, NEIGHBOR_TWEET_COLUMNS],
        clean_facts[:, NEIGHBOR_TWEET_COLUMNS],
        rtol=0, atol=0, equal_nan=True)
    repaired_raw = corrupted_raw.copy()
    repaired_raw[test[recipients], 23:31] = raw[test[recipients], 23:31]
    if not np.array_equal(repaired_raw, raw, equal_nan=True):
        raise ValueError("raw source repair did not restore feature table")
    repaired_facts = facts_from_join(runtime, repaired_raw)
    if not np.array_equal(repaired_facts, clean_facts, equal_nan=True):
        raise ValueError("source repair did not restore linked neighbor facts")
    repaired_p = predict_facts(runtime, repaired_facts, test)
    if np.max(np.abs(repaired_p - clean_p)) > 1e-12:
        raise ValueError("source repair did not restore frozen probabilities")
    threshold = runtime.threshold
    truth = labels[test]
    clean_error = (clean_p >= threshold) != truth
    corrupt_error = (corrupt_p >= threshold) != truth
    repaired_error = (repaired_p >= threshold) != truth
    selected_set = set(recipients)
    unselected_test = np.asarray(
        [i not in selected_set for i in range(len(test))])
    changed_unselected = int(np.count_nonzero(
        ((clean_p >= threshold) != (corrupt_p >= threshold)) &
        unselected_test))
    strategies = {
        "hash_random": sorted(recipients, key=lambda i: hashlib.sha256(
            f"veribot-fox8-best-repair-order-v1:{test_ids[i]}".encode()).digest()),
        "model_uncertainty": sorted(recipients, key=lambda i: (
            abs(corrupt_p[i] - threshold), test_ids[i])),
    }
    budgets = sorted(set((0, 10, 20, 40, 60, len(recipients))))
    curves = {}
    for name, queue in strategies.items():
        points = []
        for budget in budgets:
            partial = corrupted_raw.copy()
            checked = queue[:budget]
            partial[test[checked], 23:31] = raw[test[checked], 23:31]
            partial_p = predict_facts(runtime, facts_from_join(runtime, partial),
                                      test)
            source_errors = (partial_p >= threshold) != truth
            direct_errors = corrupt_error.copy()
            direct_errors[checked] = False
            points.append({
                "account_actions": budget,
                "source_corrected_total_errors": int(source_errors.sum()),
                "direct_account_labeled_total_errors": int(direct_errors.sum()),
            })
        if points[-1]["source_corrected_total_errors"] != int(clean_error.sum()):
            raise ValueError("full source repair did not restore errors")
        curves[name] = points
    rows = [{
        "account_id": test_ids[recipient],
        "donor_account_id": test_ids[donor_for[recipient]],
        "raw_account_locator": locators[test_ids[recipient]],
        "donor_raw_account_locator": locators[test_ids[donor_for[recipient]]],
        "clean_probability": float(clean_p[recipient]),
        "corrupted_probability": float(corrupt_p[recipient]),
        "repaired_probability": float(repaired_p[recipient]),
    } for recipient in recipients]
    return {
        "seed": seed, "split_sha256": source["split_sha256"],
        "model_sha256": manifest["model_sha256"],
        "saved_predictions_sha256": sha256_file(saved_path),
        "test_accounts": len(test), "injected_accounts": len(recipients),
        "clean_errors": int(clean_error.sum()),
        "corrupted_errors": int(corrupt_error.sum()),
        "repaired_errors": int(repaired_error.sum()),
        "changed_classes": int(np.count_nonzero(
            (clean_p >= threshold) != (corrupt_p >= threshold))),
        "changed_unselected_test_accounts": changed_unselected,
        "changed_neighbor_fact_cells_all_accounts": int(
            np.count_nonzero(neighbor_cell_change)),
        "accounts_with_changed_neighbor_facts": int(np.count_nonzero(
            np.any(neighbor_cell_change, axis=1))),
        "unselected_test_accounts_with_probability_change": int(
            np.count_nonzero((np.abs(clean_p - corrupt_p) > 1e-12) &
                             unselected_test)),
        "new_errors": int(np.sum(corrupt_error & ~clean_error)),
        "removed_errors": int(np.sum(clean_error & ~corrupt_error)),
        "raw_account_records_checked": len(selected_ids),
        "curves": curves, "rows": rows,
    }


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        result = evaluate_seed(seed)
        runs.append(result)
        print({key: result[key] for key in (
            "seed", "clean_errors", "corrupted_errors", "repaired_errors",
            "changed_classes", "changed_unselected_test_accounts")}, flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "status": "frozen_best_fox8_known_tweet_aggregate_row_join_repair",
        "model": "fox8_raw_neighbor_atomic_74",
        "injection_and_account_selection_use_labels": False,
        "repair_uses_labels": False,
        "source_truth": "sampled_raw_tweets_rebuilt_to_eight_account_aggregates",
        "native_interaction_graph_changed": False,
        "linked_neighbor_mean_facts_recomputed": True,
        "comparison_unit": "account actions, not human review time",
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
