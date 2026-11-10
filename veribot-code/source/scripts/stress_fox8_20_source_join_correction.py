"""Test source repair of a known Fox8 sampled-tweet bundle join error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_fox8_group_tail_20 import source_cards
from gate0_shallow_tree import load, sha256_file
from stress_botsim_12_source_join_correction import hash_order
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def matches(value: float | None, expected: float) -> bool:
    return (value is None and np.isnan(expected)) or (
        value is not None and np.isfinite(expected) and abs(value - expected) <= 1e-12)


def run(args: argparse.Namespace, seed: int) -> dict:
    full_dir = args.full_model_dir / f"seed{seed}"
    compact_dir = args.compact_model_dir / f"seed{seed}"
    full = json.loads((full_dir / "report.json").read_text(encoding="utf-8"))
    report = json.loads((compact_dir / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((compact_dir / "test_predictions.json").read_text(encoding="utf-8"))
    x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
    with np.load(args.native_facts, allow_pickle=False) as cache:
        if cache["ids"].astype(str).tolist() != ids:
            raise ValueError("native interaction account population differs")
        x = np.column_stack((x, cache["values"].astype(float)))
        names += cache["names"].astype(str).tolist()
    if (source["split_sha256"] != report["manifest"]["split_sha256"]
            or source["files_sha256"] != report["manifest"]["source_files_sha256"]
            or sha256_file(args.native_facts) != report["manifest"]["native_facts_sha256"]
            or saved["manifest"] != report["manifest"]):
        raise ValueError("frozen Fox8 source or compact model differs")
    test = split["test"]
    test_ids = [ids[int(i)] for i in test]
    cards, source_card_hash = source_cards(args.source_cards_dir, seed, test_ids)
    reference_path = full_dir / "reference.json.gz"
    if sha256_file(reference_path) != full["manifest"]["reference_artifact_sha256"]:
        raise ValueError("source reference artifact changed")
    reference = LabelBlindReference.load(reference_path)
    groups = json.loads(args.groups_json.read_text(encoding="utf-8"))
    model = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    original = x[test].copy()
    clean_concepts, _ = collapse_group_tails(reference.transform(original), groups)
    clean_probability = model.predict_proba(clean_concepts)
    recorded = saved["predictions"]
    if (model.reference_version != clean_concepts.reference_version
            or [row["account_id"] for row in recorded] != test_ids
            or np.max(np.abs(clean_probability - np.asarray(
                [row["probability"] for row in recorded]))) > 1e-12):
        raise ValueError("clean source does not reproduce frozen decisions")
    tweet_indices = list(range(names.index("n_tweets"), 31)) + list(range(31, len(names)))
    if len(tweet_indices) != 20 or len(names) != 43:
        raise ValueError("sampled tweet and interaction source vocabulary differs")
    recipient_order = hash_order(test_ids, "veribot-fox8-tweet-join-recipient-v1")
    donor_order = hash_order(test_ids, "veribot-fox8-tweet-join-donor-v1")
    selected = recipient_order[:round(len(test) * 0.25)]
    corrupted = original.copy()
    donors = {}
    for rank, recipient in enumerate(selected):
        donor = next((candidate for candidate in donor_order[rank:] + donor_order[:rank]
                      if candidate != recipient and not np.array_equal(
                          original[candidate, tweet_indices],
                          original[recipient, tweet_indices], equal_nan=True)), None)
        if donor is None:
            raise ValueError("no different donor sampled-tweet bundle")
        for j in tweet_indices:
            if (cards[recipient]["terms"][j]["name"] != names[j]
                    or cards[donor]["terms"][j]["name"] != names[j]
                    or not matches(cards[recipient]["terms"][j]["source_value"],
                                   original[recipient, j])
                    or not matches(cards[donor]["terms"][j]["source_value"],
                                   original[donor, j])):
                raise ValueError("audited raw sampled-tweet bundle differs")
        corrupted[recipient, tweet_indices] = original[donor, tweet_indices]
        donors[recipient] = donor
    bad_concepts, _ = collapse_group_tails(reference.transform(corrupted), groups)
    bad_probability = model.predict_proba(bad_concepts)
    repaired = corrupted.copy()
    repaired[np.ix_(selected, tweet_indices)] = original[np.ix_(selected, tweet_indices)]
    if not np.array_equal(repaired, original, equal_nan=True):
        raise ValueError("repair did not restore the raw sampled-tweet facts")
    repaired_probability = model.predict_proba(
        collapse_group_tails(reference.transform(repaired), groups)[0])
    if np.max(np.abs(repaired_probability - clean_probability)) > 1e-12:
        raise ValueError("source repair did not restore fixed model probabilities")
    threshold = float(report["manifest"]["threshold"])
    labels = y[test]
    clean_error = (clean_probability >= threshold) != labels
    bad_error = (bad_probability >= threshold) != labels
    repaired_error = (repaired_probability >= threshold) != labels
    selected = np.asarray(selected, dtype=int)
    strategies = {
        "hash_random": sorted(selected, key=lambda i: hashlib.sha256(
            f"veribot-fox8-correction-order-v1:{test_ids[i]}".encode()).digest()),
        "model_uncertainty": sorted(selected,
                                    key=lambda i: (abs(bad_probability[i] - threshold),
                                                   test_ids[i])),
    }
    budgets = sorted(set((0, 10, 20, 40, 60, len(selected))))
    curves = {}
    for strategy, ordering in strategies.items():
        points = []
        for budget in budgets:
            checked = np.asarray(ordering[:budget], dtype=int)
            after_source = bad_error.copy()
            after_source[checked] = clean_error[checked]
            after_label = bad_error.copy()
            after_label[checked] = False
            points.append({
                "account_actions": budget,
                "source_correction_total_errors": int(after_source.sum()),
                "direct_account_label_total_errors": int(after_label.sum()),
                "source_correction_net_errors_removed": int(bad_error.sum() - after_source.sum()),
                "direct_account_label_net_errors_removed": int(bad_error.sum() - after_label.sum()),
            })
        if points[-1]["source_correction_total_errors"] != int(clean_error.sum()):
            raise ValueError("full source correction does not return to clean error count")
        curves[strategy] = points
    rows = []
    for recipient in selected:
        donor = donors[int(recipient)]
        rows.append({
            "account_id": test_ids[recipient], "donor_account_id": test_ids[donor],
            "source_locator": {"raw_account_key": test_ids[recipient],
                               "operation": "sampled_tweet_bundle_and_native_targets"},
            "source_card_sha256": source_card_hash,
            "source_fact_values": {names[j]: (float(original[recipient, j])
                                             if np.isfinite(original[recipient, j]) else None)
                                   for j in tweet_indices},
            "wrong_join_fact_values": {names[j]: (float(corrupted[recipient, j])
                                                 if np.isfinite(corrupted[recipient, j]) else None)
                                       for j in tweet_indices},
            "clean_probability": float(clean_probability[recipient]),
            "corrupted_probability": float(bad_probability[recipient]),
            "repaired_probability": float(repaired_probability[recipient]),
            "label_for_evaluation": int(labels[recipient]),
        })
    return {
        "seed": seed, "split_sha256": source["split_sha256"],
        "test_accounts": len(test), "selected_accounts": len(selected),
        "clean_errors": int(clean_error.sum()),
        "corrupted_errors": int(bad_error.sum()),
        "source_corrected_errors": int(repaired_error.sum()),
        "corruption_changed_predictions": int(np.count_nonzero(
            (bad_probability >= threshold) != (clean_probability >= threshold))),
        "corruption_new_errors": int(np.sum(bad_error & ~clean_error)),
        "corruption_removed_errors": int(np.sum(clean_error & ~bad_error)),
        "max_repair_probability_difference": float(np.max(
            np.abs(repaired_probability - clean_probability))),
        "threshold": threshold, "curves": curves, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--native-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--compact-model-dir", required=True, type=Path)
    parser.add_argument("--source-cards-dir", required=True, type=Path)
    parser.add_argument("--groups-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = [run(args, seed) for seed in (0, 1, 2)]
    result = {
        "dataset": "fox8", "model": "source_backed_twenty_group_tail_concepts",
        "scope": "known_sampled_tweet_bundle_join_error_injection_and_repair",
        "injection_uses_bot_human_labels": False,
        "account_selection_uses_bot_human_labels": False,
        "source_correction_uses_bot_human_labels": False,
        "comparison_unit": "account actions; reviewer time and natural error frequency not measured",
        "source_truth": "previously independently verified raw sampled tweets and target events",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{key: row[key] for key in (
        "seed", "test_accounts", "selected_accounts", "clean_errors",
        "corrupted_errors", "source_corrected_errors",
        "corruption_changed_predictions", "corruption_new_errors",
        "corruption_removed_errors")}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
