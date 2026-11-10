"""Conditioned one-source repair utility for the neutral Fox8 classifier."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from stress_fox8_targeted_twenty_source_repair import score
from verifiable_bot.fox8_profile_integrity import (
    detect_statuses_discrepancies, read_first_raw_statuses)
from verifiable_bot.fox8_twenty_neutral_anchor_runtime import (
    Fox8TwentyNeutralAnchorRuntime)


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_twenty_neutral_anchor"
OUT = FOLDER / "top_source_donor_sweep.json"
RAW_AUDIT = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"
DONOR_COUNT = 32


def main() -> None:
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    fixed = json.loads((FOLDER / "fixed_test_report.json")
                       .read_text(encoding="utf-8"))
    if raw_audit["status"] != (
            "all_model_used_fox8_twenty_facts_independently_recomputed_from_raw"):
        raise ValueError("Fox8 raw source audit differs")
    runs = []
    first_source = None
    first_donors = None
    for seed in (0, 1, 2):
        runtime = Fox8TwentyNeutralAnchorRuntime(ROOT, seed)
        if sha256_file(runtime.source.raw_path) != raw_audit["raw_source_sha256"]:
            raise ValueError("Fox8 raw release changed")
        snapshot = read_first_raw_statuses(runtime.source.raw_path,
                                           runtime.ids)
        truth = snapshot.values
        eligible = [uid for uid in runtime.ids
                    if uid not in snapshot.ambiguous_account_ids]
        selected = sorted(eligible, key=lambda uid: (
            -len(runtime.source.neighbors[runtime.index[uid]]), uid))[0]
        candidates = [uid for uid in eligible
                      if uid != selected and
                      truth[runtime.index[uid]] != truth[runtime.index[selected]]]
        donors = sorted(candidates, key=lambda uid: hashlib.sha256(
            f"Fox8-high-fanout-donor-sweep-v1:{uid}".encode()).digest())[
                :DONOR_COUNT]
        if (first_source is not None and selected != first_source or
                first_donors is not None and donors != first_donors):
            raise ValueError("Label-blind source and donor selection differs")
        first_source, first_donors = selected, donors
        saved = fixed["runs"][seed]
        with np.load(saved["prediction_matrix_path"],
                     allow_pickle=False) as z:
            ids = z["account_ids"].astype(str).tolist()
            expected_p = z["probability"].astype(float)
        original = json.loads((runtime.run_dir / "test_predictions.json")
                              .read_text(encoding="utf-8"))["predictions"]
        if [row["account_id"] for row in original] != ids:
            raise ValueError("Official test account order differs")
        labels = np.asarray([row["label"] for row in original], dtype=int)
        rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        clean = score(runtime, truth, rows)
        if np.max(np.abs(clean-expected_p)) > 1e-12:
            raise ValueError("Clean source does not replay neutral prediction")
        clean_error = (clean >= runtime.threshold) != labels
        results = []
        for donor in donors:
            corrupted = truth.copy()
            corrupted[runtime.index[selected]] = truth[runtime.index[donor]]
            detected = detect_statuses_discrepancies(
                runtime.ids, corrupted, snapshot)
            if detected != [selected]:
                raise ValueError("One wrong join was not independently discovered")
            bad = score(runtime, corrupted, rows)
            bad_error = (bad >= runtime.threshold) != labels
            after_one_repair = clean_error.sum()
            oracle_one_label = max(0, int(bad_error.sum()) - 1)
            results.append({
                "donor_account_id": donor,
                "donor_raw_statuses_count": float(truth[runtime.index[donor]]),
                "source_raw_statuses_count": float(truth[runtime.index[selected]]),
                "corrupted_errors": int(bad_error.sum()),
                "new_errors_from_wrong_join": int(np.count_nonzero(
                    bad_error & ~clean_error)),
                "wrong_join_changed_classes": int(np.count_nonzero(
                    (bad >= runtime.threshold) !=
                    (clean >= runtime.threshold))),
                "one_source_repair_remaining_errors": int(after_one_repair),
                "one_oracle_direct_label_remaining_errors": oracle_one_label,
                "source_repair_beats_one_oracle_label": bool(
                    after_one_repair < oracle_one_label),
            })
        runs.append({
            "seed": seed, "test_accounts": len(ids),
            "clean_errors": int(clean_error.sum()),
            "top_source_account_id": selected,
            "top_source_native_neighbor_count": len(
                runtime.source.neighbors[runtime.index[selected]]),
            "top_source_test_recipients": sum(
                selected == uid or selected in
                runtime.source.neighbors[runtime.index[uid]] for uid in ids),
            "donor_count": len(donors),
            "donors_with_one_source_repair_advantage": sum(
                x["source_repair_beats_one_oracle_label"] for x in results),
            "donors": results,
            "neutral_model_sha256": runtime.neutral_model_sha256,
            "fixed_test_prediction_matrix_sha256": saved[
                "prediction_matrix_sha256"]})
        print({"seed": seed,
               "advantage_donors": runs[-1][
                   "donors_with_one_source_repair_advantage"],
               "clean_errors": runs[-1]["clean_errors"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "neutral_Fox8_one_graph_central_source_known_wrong_join_donor_sensitivity",
        "source_and_donor_selection_use_bot_human_labels": False,
        "wrong_join_discovery_uses_bot_human_labels": False,
        "same_budget_unit": "one_source_record_correction_or_one_oracle_direct_account_label",
        "natural_error_frequency_or_human_cost_measured": False,
        "prior_test_population_and_donor_list_previously_explored": True,
        "independent_raw_truth_sha256": sha256_file(RAW_AUDIT),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
