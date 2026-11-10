"""Discover and repair raw-profile joins in the locked neutral Fox8 model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from stress_fox8_targeted_twenty_source_repair import order, score
from verifiable_bot.fox8_profile_integrity import (
    detect_statuses_discrepancies, read_first_raw_statuses)
from verifiable_bot.fox8_twenty_neutral_anchor_runtime import (
    Fox8TwentyNeutralAnchorRuntime)


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_twenty_neutral_anchor"
OUT = FOLDER / "known_source_repair.json"
SOURCE_AUDIT = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"


def main() -> None:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    fixed = json.loads((FOLDER / "fixed_test_report.json")
                       .read_text(encoding="utf-8"))
    if audit["status"] != "all_model_used_fox8_twenty_facts_independently_recomputed_from_raw":
        raise ValueError("Fox8 independent raw source audit missing")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TwentyNeutralAnchorRuntime(ROOT, seed)
        if sha256_file(runtime.source.raw_path) != audit["raw_source_sha256"]:
            raise ValueError("Fox8 raw profile source changed")
        snapshot = read_first_raw_statuses(runtime.source.raw_path,
                                           runtime.ids)
        truth = snapshot.values
        col = runtime.reference.names.index("statuses_count")
        if not np.array_equal(
                truth.astype(np.float32),
                runtime.source.facts[:, col].astype(np.float32)):
            raise ValueError("Clean source differs from raw record")
        if detect_statuses_discrepancies(
                runtime.ids, runtime.source.facts[:, col], snapshot):
            raise ValueError("Clean Fox8 materialized source has a discrepancy")
        saved = fixed["runs"][seed]
        with np.load(saved["prediction_matrix_path"],
                     allow_pickle=False) as z:
            test_ids = z["account_ids"].astype(str).tolist()
            saved_p = z["probability"].astype(float)
        original_predictions = json.loads((runtime.run_dir /
            "test_predictions.json").read_text(encoding="utf-8"))["predictions"]
        if [x["account_id"] for x in original_predictions] != test_ids:
            raise ValueError("Fox8 test account IDs differ")
        labels = np.asarray([x["label"] for x in original_predictions], dtype=int)
        target_rows = np.asarray([runtime.index[uid] for uid in test_ids], dtype=int)
        clean = score(runtime, truth, target_rows)
        clean_error = (clean >= runtime.threshold) != labels
        clean_replay = float(np.max(np.abs(clean-saved_p)))
        if clean_replay > 1e-12:
            raise ValueError("Clean neutral model probability differs")
        ambiguous = set(snapshot.ambiguous_account_ids)
        candidate_ids = [uid for uid in runtime.ids
                         if uid not in ambiguous and
                         runtime.source.neighbors[runtime.index[uid]]]
        recipients = order(candidate_ids,
                           "Fox8-targeted-twenty-join-recipients-v1")
        selected = recipients[:round(.10 * len(recipients))]
        donors = order(runtime.ids, "Fox8-targeted-twenty-join-donors-v1")
        corrupted = truth.copy()
        source_rows = []
        for rank, uid in enumerate(selected):
            donor = next(item for item in donors[rank:] + donors[:rank]
                         if item != uid and truth[runtime.index[item]] !=
                         truth[runtime.index[uid]])
            corrupted[runtime.index[uid]] = truth[runtime.index[donor]]
            source_rows.append({
                "source_account_id": uid,
                "wrong_join_donor_id": donor,
                "raw_source_value": float(truth[runtime.index[uid]]),
                "wrong_join_value": float(truth[runtime.index[donor]])})
        detected = detect_statuses_discrepancies(
            runtime.ids, corrupted, snapshot)
        if set(detected) != set(selected) or len(detected) != len(selected):
            raise ValueError("Label-free raw record comparison missed a join error")
        bad = score(runtime, corrupted, target_rows)
        bad_error = (bad >= runtime.threshold) != labels
        target_index = {uid: i for i, uid in enumerate(test_ids)}
        uncertainty_order = sorted(test_ids, key=lambda uid: (
            abs(float(bad[target_index[uid]])-runtime.threshold), uid))
        budgets = sorted(set((0, 1, 10, 25, 50, 100, len(selected))))
        curve = []
        for budget in budgets:
            repaired = corrupted.copy()
            for uid in selected[:budget]:
                repaired[runtime.index[uid]] = truth[runtime.index[uid]]
            remaining = detect_statuses_discrepancies(
                runtime.ids, repaired, snapshot)
            if set(remaining) != set(selected[budget:]):
                raise ValueError("Source repair leaves unexpected discrepancies")
            after = score(runtime, repaired, target_rows)
            direct = bad_error.copy()
            direct[[target_index[uid] for uid in
                    uncertainty_order[:budget]]] = False
            curve.append({
                "actions": budget,
                "source_corrected_errors": int(np.count_nonzero(
                    (after >= runtime.threshold) != labels)),
                "direct_account_labeled_errors": int(direct.sum()),
                "source_remaining_mismatches": len(remaining),
                "source_correction_changed_classes": int(np.count_nonzero(
                    (after >= runtime.threshold) !=
                    (bad >= runtime.threshold)))})
        fully = corrupted.copy()
        fully[[runtime.index[uid] for uid in selected]] = truth[
            [runtime.index[uid] for uid in selected]]
        repair_error = float(np.max(np.abs(
            score(runtime, fully, target_rows) - clean)))
        if (curve[-1]["source_corrected_errors"] != int(clean_error.sum()) or
                repair_error > 1e-12):
            raise ValueError("Full source correction failed to restore model")
        runs.append({
            "seed": seed, "test_accounts": len(test_ids),
            "wrong_join_sources": len(selected),
            "ambiguous_raw_source_accounts_excluded": len(ambiguous),
            "wrong_joins_discovered_without_labels": len(detected),
            "false_discoveries_in_clean_source": 0,
            "clean_errors": int(clean_error.sum()),
            "corrupted_errors": int(bad_error.sum()),
            "fully_corrected_errors": int(clean_error.sum()),
            "corruption_changed_classes": int(np.count_nonzero(
                (bad >= runtime.threshold) !=
                (clean >= runtime.threshold))),
            "maximum_clean_probability_replay_error": clean_replay,
            "maximum_full_repair_probability_error": repair_error,
            "budget_curve": curve,
            "source_rows": source_rows,
            "neutral_model_sha256": runtime.neutral_model_sha256,
            "fixed_test_prediction_matrix_sha256": saved[
                "prediction_matrix_sha256"]})
        print({"seed": seed, "wrong_sources": len(selected),
               "clean_errors": int(clean_error.sum()),
               "corrupted_errors": int(bad_error.sum())}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "neutral_Fox8_20_concept_known_profile_join_error_detection_and_repair",
        "independent_raw_truth_sha256": sha256_file(SOURCE_AUDIT),
        "error_selection_discovery_and_repair_use_bot_human_labels": False,
        "same_budget_action_unit": "one_source_record_correction_or_one_directly_labeled_test_account",
        "natural_error_frequency_or_human_minutes_measured": False,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
