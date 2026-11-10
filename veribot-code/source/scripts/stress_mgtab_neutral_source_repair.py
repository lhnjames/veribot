"""Known published content-row joins and repair in the neutral MGTAB model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from stress_mgtab_best_source_join_correction import (
    DONOR_SALT, RECIPIENT_SALT, evidence_design, order, project_content)
from verifiable_bot.mgtab_neutral_anchor_runtime import MGTABNeutralAnchorRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_150_neutral_anchor"
OUT = FOLDER / "known_content_row_source_repair.json"


def main() -> None:
    pointer = json.loads((ROOT / "data/raw/mgtab_source.json")
                         .read_text(encoding="utf-8"))
    published_path = Path(pointer["path"]) / "features.pt"
    published = torch.load(published_path, map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("MGTAB published feature shape differs")
    lock = json.loads((FOLDER / "selection_lock.json")
                      .read_text(encoding="utf-8"))
    if sha256_file(published_path) != lock["published_source_features_sha256"]:
        raise ValueError("MGTAB published content source differs")
    fixed = json.loads((FOLDER / "fixed_test_report.json")
                       .read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABNeutralAnchorRuntime(ROOT, seed)
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        test = split["test"]
        saved = fixed["runs"][seed]
        with np.load(saved["predictions_path"], allow_pickle=False) as z:
            stored_ids = z["account_ids"].astype(str).tolist()
            stored_p = z["probability"].astype(float)
        if (ids != runtime.ids or
                source["split_sha256"] != saved["split_sha256"] or
                [ids[int(i)] for i in test] != stored_ids):
            raise ValueError("Official MGTAB test population changed")
        projected = project_content(published, runtime.projection_path)
        original = runtime.facts[test].copy()
        projection_error = float(np.max(np.abs(
            projected[test] - original[:, 86:150])))
        if projection_error > 1e-10:
            raise ValueError("Published content projection does not replay")
        clean = runtime.model.predict_proba(
            evidence_design(runtime, original, test))[:, 1]
        clean_error = float(np.max(np.abs(clean-stored_p)))
        if clean_error > 1e-12:
            raise ValueError("Clean neutral score differs from saved test")
        test_ids = [ids[int(i)] for i in test]
        recipient_order = order(test_ids, RECIPIENT_SALT)
        donor_order = order(test_ids, DONOR_SALT)
        recipients = recipient_order[:round(len(test)*.25)]
        corrupted = original.copy()
        donors = {}
        for rank, recipient in enumerate(recipients):
            donor = next(candidate for candidate in
                         donor_order[rank:] + donor_order[:rank]
                         if candidate != recipient and not np.array_equal(
                             published[test[candidate], 20:],
                             published[test[recipient], 20:]))
            corrupted[recipient, 86:150] = projected[test[donor]]
            donors[recipient] = donor
        detected = [i for i in range(len(test))
                    if not np.array_equal(corrupted[i, 86:150],
                                          projected[test[i]])]
        if set(detected) != set(recipients):
            raise ValueError("Published-row source comparison missed a wrong join")
        bad = runtime.model.predict_proba(
            evidence_design(runtime, corrupted, test))[:, 1]
        threshold = runtime.threshold
        truth = labels[test]
        clean_wrong = (clean >= threshold) != truth
        bad_wrong = (bad >= threshold) != truth
        repair_order = sorted(recipients, key=lambda i: (
            abs(float(bad[i])-threshold), test_ids[i]))
        budgets = sorted(set((0, 1, 20, 50, 100, 150, len(recipients))))
        curve = []
        for budget in budgets:
            selected = repair_order[:budget]
            repaired = corrupted.copy()
            repaired[selected, 86:150] = projected[test[selected]]
            remaining = [i for i in range(len(test))
                         if not np.array_equal(repaired[i, 86:150],
                                               projected[test[i]])]
            if set(remaining) != set(recipients) - set(selected):
                raise ValueError("MGTAB source repair leaves unexpected wrong join")
            after = runtime.model.predict_proba(
                evidence_design(runtime, repaired, test))[:, 1]
            direct = bad_wrong.copy()
            direct[selected] = False
            curve.append({
                "actions": budget,
                "source_corrected_errors": int(np.count_nonzero(
                    (after >= threshold) != truth)),
                "direct_account_labeled_errors": int(direct.sum()),
                "remaining_detected_wrong_joins": len(remaining),
            })
        fully = corrupted.copy()
        fully[recipients, 86:150] = projected[test[recipients]]
        repair_error = float(np.max(np.abs(
            runtime.model.predict_proba(
                evidence_design(runtime, fully, test))[:, 1] - clean)))
        if (curve[-1]["source_corrected_errors"] !=
                int(clean_wrong.sum()) or repair_error > 1e-12):
            raise ValueError("Full MGTAB source repair did not restore model")
        runs.append({
            "seed": seed, "test_accounts": len(test),
            "wrong_join_rows": len(recipients),
            "wrong_joins_detected_without_labels": len(detected),
            "clean_errors": int(clean_wrong.sum()),
            "corrupted_errors": int(bad_wrong.sum()),
            "fully_corrected_errors": int(clean_wrong.sum()),
            "corruption_changed_classes": int(np.count_nonzero(
                (bad >= threshold) != (clean >= threshold))),
            "maximum_projection_replay_error": projection_error,
            "maximum_clean_probability_replay_error": clean_error,
            "maximum_full_repair_probability_error": repair_error,
            "budget_curve": curve,
            "wrong_join_rows_with_donors": [
                {"account_id": test_ids[i],
                 "wrong_join_donor_id": test_ids[donors[i]],
                 "published_source_row": int(test[i]),
                 "donor_source_row": int(test[donors[i]])}
                for i in recipients],
            "neutral_model_sha256": runtime.neutral_anchor_model_sha256,
            "fixed_test_prediction_sha256": saved["predictions_sha256"],
        })
        print({"seed": seed, "clean_errors": runs[-1]["clean_errors"],
               "corrupted_errors": runs[-1]["corrupted_errors"]},
              flush=True)
    OUT.write_text(json.dumps({
        "scope": "neutral_MGTAB_known_published_content_row_join_and_repair",
        "published_source_features_sha256": sha256_file(published_path),
        "source_selection_discovery_and_repair_use_bot_human_labels": False,
        "direct_label_budget_unit": "one_source_row_correction_or_one_directly_labeled_test_account",
        "individual_tweet_text_truth_available": False,
        "natural_error_frequency_or_human_minutes_measured": False,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
