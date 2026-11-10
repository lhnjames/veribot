"""Test source-backed repair of a known, label-blind BotSim profile join error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_botsim_group_tail_12 import read_verified_source_cards
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def hash_order(ids: list[str], salt: str) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"{salt}:{ids[i]}".encode()).digest())


def run(args: argparse.Namespace, seed: int) -> dict:
    compact_dir = args.compact_model_dir / f"seed{seed}"
    full_dir = args.full_model_dir / f"seed{seed}"
    report = json.loads((compact_dir / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((compact_dir / "test_predictions.json")
                       .read_text(encoding="utf-8"))
    full = json.loads((full_dir / "report.json").read_text(encoding="utf-8"))
    x, y, ids, names, split, source = load(
        "botsim", args.data_dir, args.account_file, seed, args.native_graph_facts)
    if (source["split_sha256"] != report["manifest"]["split_sha256"]
            or source["files_sha256"] != report["manifest"]["source_files_sha256"]
            or saved["manifest"] != report["manifest"]):
        raise ValueError("fixed BotSim source, split or compact model differs")
    test = split["test"]
    test_ids = [ids[int(i)] for i in test]
    cards = read_verified_source_cards(args.source_cards_dir, seed, test_ids)
    reference_path = full_dir / "reference.json.gz"
    if sha256_file(reference_path) != full["manifest"]["reference_artifact_sha256"]:
        raise ValueError("label-blind source reference changed")
    reference = LabelBlindReference.load(reference_path)
    groups = full["reasoner"]["groups"]
    model = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    original = x[test].copy()
    clean_concepts, _ = collapse_group_tails(reference.transform(original), groups)
    clean_probability = model.predict_proba(clean_concepts)
    recorded = saved["predictions"]
    if (model.reference_version != clean_concepts.reference_version
            or [row["account_id"] for row in recorded] != test_ids
            or np.max(np.abs(clean_probability - np.asarray(
                [row["probability"] for row in recorded]))) > 1e-12):
        raise ValueError("clean source does not reproduce frozen test predictions")
    profile_indices = [names.index(name) for name in groups["user_presentation"]]
    if [names[j] for j in profile_indices] != ["has_description", "description_len"]:
        raise ValueError("profile source group differs")
    recipient_order = hash_order(test_ids, "veribot-botsim-profile-join-recipient-v1")
    donor_order = hash_order(test_ids, "veribot-botsim-profile-join-donor-v1")
    selected = recipient_order[:round(len(test) * 0.25)]
    corrupted = original.copy()
    donors = {}
    for rank, recipient in enumerate(selected):
        recipient_uid = test_ids[recipient]
        donor = next((candidate for candidate in donor_order[rank:] + donor_order[:rank]
                      if candidate != recipient and not np.array_equal(
                          original[candidate, profile_indices],
                          original[recipient, profile_indices])), None)
        if donor is None:
            raise ValueError("no different, label-blind source row for profile join")
        donor_uid = test_ids[donor]
        for j in profile_indices:
            recipient_term = cards[recipient_uid]["terms"][j]
            donor_term = cards[donor_uid]["terms"][j]
            if (recipient_term["name"] != names[j] or donor_term["name"] != names[j]
                    or abs(recipient_term["source_value"] - original[recipient, j]) > 1e-12
                    or abs(donor_term["source_value"] - original[donor, j]) > 1e-12):
                raise ValueError("raw-source profile join fields differ")
        corrupted[recipient, profile_indices] = original[donor, profile_indices]
        donors[recipient] = donor
    bad_concepts, _ = collapse_group_tails(reference.transform(corrupted), groups)
    bad_probability = model.predict_proba(bad_concepts)
    repaired = corrupted.copy()
    repaired[np.ix_(selected, profile_indices)] = original[np.ix_(selected, profile_indices)]
    if not np.array_equal(repaired, original):
        raise ValueError("source correction did not restore original facts")
    repaired_probability = model.predict_proba(
        collapse_group_tails(reference.transform(repaired), groups)[0])
    if np.max(np.abs(repaired_probability - clean_probability)) > 1e-12:
        raise ValueError("source repair did not restore frozen model outputs")
    threshold = float(report["manifest"]["threshold"])
    labels = y[test]
    clean_error = (clean_probability >= threshold) != labels
    bad_error = (bad_probability >= threshold) != labels
    repaired_error = (repaired_probability >= threshold) != labels
    if not np.array_equal(clean_error, repaired_error):
        raise ValueError("corrected decisions differ from clean source")
    selected = np.asarray(selected, dtype=int)
    strategies = {
        "hash_random": sorted(selected, key=lambda i: hashlib.sha256(
            f"veribot-botsim-correction-order-v1:{test_ids[i]}".encode()).digest()),
        "model_uncertainty": sorted(selected,
                                    key=lambda i: (abs(bad_probability[i] - threshold),
                                                   test_ids[i])),
    }
    budgets = sorted(set((0, 10, 20, 50, 75, len(selected))))
    curves = {}
    for strategy, ordering in strategies.items():
        points = []
        for budget in budgets:
            checked = np.asarray(ordering[:budget], dtype=int)
            source_corrected = bad_error.copy()
            source_corrected[checked] = clean_error[checked]
            direct_labeled = bad_error.copy()
            direct_labeled[checked] = False
            points.append({
                "account_actions": budget,
                "source_correction_total_errors": int(source_corrected.sum()),
                "direct_account_label_total_errors": int(direct_labeled.sum()),
                "source_correction_net_errors_removed": int(
                    bad_error.sum() - source_corrected.sum()),
                "direct_account_label_net_errors_removed": int(
                    bad_error.sum() - direct_labeled.sum()),
            })
        if (points[-1]["source_correction_total_errors"] != int(clean_error.sum())
                or points[-1]["direct_account_label_total_errors"] !=
                int(np.sum(bad_error & ~np.isin(np.arange(len(test)), selected)))):
            raise ValueError("full-budget source/label endpoints differ")
        curves[strategy] = points
    rows = []
    for recipient in selected:
        donor = donors[int(recipient)]
        rows.append({
            "account_id": test_ids[recipient], "donor_account_id": test_ids[donor],
            "source_locator": {"users_csv_row": int(test[recipient]) + 2,
                               "activity_account_key": test_ids[recipient]},
            "source_values": {names[j]: float(original[recipient, j])
                              for j in profile_indices},
            "wrong_join_values": {names[j]: float(corrupted[recipient, j])
                                  for j in profile_indices},
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
    parser.add_argument("--account-file", required=True, type=Path)
    parser.add_argument("--native-graph-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--compact-model-dir", required=True, type=Path)
    parser.add_argument("--source-cards-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = [run(args, seed) for seed in (0, 1, 2)]
    result = {
        "dataset": "botsim", "model": "source_backed_twelve_group_tail_concepts",
        "scope": "known_profile_source_row_join_error_injection_and_repair",
        "injection_uses_bot_human_labels": False,
        "account_selection_uses_bot_human_labels": False,
        "source_correction_uses_bot_human_labels": False,
        "comparison_unit": "account actions; reviewer time and natural error frequency not measured",
        "source_truth": "previously independently verified raw Users.csv and activity records",
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
