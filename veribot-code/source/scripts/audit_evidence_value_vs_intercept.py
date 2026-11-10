"""Audit fixed-test accuracy gained from evidence over the frozen intercept."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PREDICTIONS = {
    "botsim": "output/v3_computable/botsim_group_tail_12/seed{seed}/test_predictions.json",
    "fox8": "output/v3_fox8_native_spline_l2/group_tail_20_selected/seed{seed}/test_predictions.json",
    "mgtab": "output/v3_mgtab_ab_spline_content/direct_selected_20/seed{seed}/test_predictions.json",
    "twibot20": "output/v3_twibot20_sparse_text_evidence/selected/test_predictions.json",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dependency_path = root / "research/three_dataset_compact_decision_dependence_2026-11-03.json"
    mask_path = root / "research/all_evidence_mask_dependency_2026-11-03.json"
    dependency = read(dependency_path)
    mask = read(mask_path)
    runs = []
    for mask_run in mask["rows"]:
        dataset, seed = mask_run["dataset"], mask_run["seed"]
        relative = PREDICTIONS[dataset]
        prediction_path = root / (relative.format(seed=seed) if seed is not None else relative)
        saved_predictions = read(prediction_path)
        predictions = saved_predictions["predictions"]
        threshold = float(saved_predictions["manifest"]["threshold"])
        if not 0 < threshold < 1:
            raise ValueError("invalid frozen threshold")
        if len(predictions) != mask_run["test_n"]:
            raise ValueError(f"test population changed: {dataset}/{seed}")
        if dataset != "twibot20":
            audit = next(item for item in dependency["datasets"]
                         if item["dataset"] == dataset)
            detail = next(item for item in audit["runs"] if item["seed"] == seed)
            accounts = detail["accounts"]
            if len(accounts) != len(predictions):
                raise ValueError("dependency population changed")
            for saved, account in zip(predictions, accounts):
                if (saved["account_id"] != account["account_id"] or
                        int(saved["probability"] >= threshold) !=
                        int(account["prediction"]) or
                        abs(saved["probability"] - account["probability"]) > 1e-11 or
                        account["intercept_only_prediction"] !=
                        mask_run["all_evidence_zero_mask_prediction"]):
                    raise ValueError("saved prediction or intercept does not replay")
            masked_class = accounts[0]["intercept_only_prediction"]
        else:
            masked_class = mask_run["all_evidence_zero_mask_prediction"]
        if masked_class not in (0, 1):
            raise ValueError("invalid intercept-only class")
        correct_full = correct_masked = rescued = harmed = 0
        positive_predictions = 0
        for saved in predictions:
            label = int(saved["label"])
            prediction = int(saved["probability"] >= threshold)
            if label not in (0, 1) or prediction not in (0, 1):
                raise ValueError("invalid prediction or label")
            full_ok = prediction == label
            masked_ok = masked_class == label
            positive_predictions += prediction
            correct_full += full_ok
            correct_masked += masked_ok
            rescued += full_ok and not masked_ok
            harmed += masked_ok and not full_ok
        if correct_full - correct_masked != rescued - harmed:
            raise ValueError("accuracy change does not balance")
        if (rescued + harmed != mask_run["decisions_changed_by_all_evidence_mask"]
                or positive_predictions != mask_run["positive_predictions"]):
            raise ValueError("saved predictions do not match prior mask audit")
        runs.append({
            "dataset": dataset, "seed": seed, "test_n": len(predictions),
            "predictions_sha256": sha256(prediction_path),
            "masked_class": masked_class,
            "correct_with_all_evidence": correct_full,
            "correct_with_all_evidence_masked": correct_masked,
            "acc_with_all_evidence": correct_full / len(predictions),
            "acc_with_all_evidence_masked": correct_masked / len(predictions),
            "accounts_rescued_by_evidence": rescued,
            "accounts_harmed_by_evidence": harmed,
            "net_extra_correct_with_evidence": rescued - harmed,
            "interpretation": "zero-mask inside the fixed model, not an account intervention",
        })
    output = {
        "scope": "fixed_test_full_evidence_vs_intercept_only_no_model_selection",
        "decision_dependence_sha256": sha256(dependency_path),
        "all_evidence_mask_audit_sha256": sha256(mask_path),
        "runs": runs,
    }
    destination = root / "research/four_dataset_evidence_value_vs_intercept_2026-11-03.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: run[key] for key in (
        "dataset", "seed", "test_n", "correct_with_all_evidence",
        "correct_with_all_evidence_masked", "accounts_rescued_by_evidence",
        "accounts_harmed_by_evidence", "net_extra_correct_with_evidence")}
        for run in runs], indent=2))


if __name__ == "__main__":
    main()
