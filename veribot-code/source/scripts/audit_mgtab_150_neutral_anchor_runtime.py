"""Read back every fixed-test score through label-free source-bound runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_neutral_anchor_runtime import MGTABNeutralAnchorRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_150_neutral_anchor"
CERT = ROOT / "research/certificates/mgtab_150_neutral_anchor_runtime.json"


def main() -> None:
    fixed_path = FOLDER / "fixed_test_report.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    if (fixed["scope"] != "dev_locked_MGTAB_150_fact_neutral_anchor_fixed_test" or
            not fixed["all_seeds_above_hypbot"] or
            not fixed["all_seeds_above_91_2_percent"]):
        raise ValueError("Locked fixed-test gate differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABNeutralAnchorRuntime(ROOT, seed)
        saved = fixed["runs"][seed]
        path = Path(saved["predictions_path"])
        if sha256_file(path) != saved["predictions_sha256"]:
            raise ValueError("Saved fixed-test predictions changed")
        with np.load(path, allow_pickle=False) as z:
            ids = z["account_ids"].astype(str)
            expected_p = z["probability"].astype(float)
            expected_y = z["predicted_bot"].astype(bool)
            threshold = float(z["threshold"])
        if threshold != runtime.threshold or len(ids) != 1020:
            raise ValueError("Neutral threshold or official population differs")
        rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        x = evidence_design(runtime, rows)
        raw = np.asarray(runtime.model.booster_.predict(
            x, raw_score=True), dtype=float)
        full_p = expit(raw)
        full_error = float(np.max(np.abs(full_p - expected_p)))
        if full_error > 1e-12 or not np.array_equal(
                full_p >= threshold, expected_y):
            raise ValueError("Source-bound account inference differs")
        zero_raw = float(runtime.model.booster_.predict(
            np.zeros((1, x.shape[1])), raw_score=True)[0])
        if abs(float(expit(zero_raw)) - threshold) > 1e-12:
            raise ValueError("Neutral anchor differs in raw score")
        group_delta = []
        for indices in runtime.group_fact_indices.values():
            masked = x.copy()
            masked[:, indices] = 0.0
            masked[:, np.asarray(indices) + 150] = 0.0
            group_delta.append(raw - np.asarray(
                runtime.model.booster_.predict(masked, raw_score=True),
                dtype=float))
        reconstructed = zero_raw + np.sum(group_delta, axis=0)
        group_error = float(np.max(np.abs(reconstructed - raw)))
        if group_error > 1e-9:
            raise ValueError("Group terms do not reconstruct decision score")
        signed = np.where(expected_y, raw > zero_raw + 1e-10,
                          raw < zero_raw - 1e-10)
        if not signed.all():
            raise ValueError("An account lacks evidence direction support")
        sample_error = 0.0
        for uid in ids[[0, len(ids)//2, len(ids)-1]]:
            card = runtime.predict(uid)
            j = int(np.flatnonzero(ids == uid)[0])
            sample_error = max(sample_error,
                               abs(card["probability"] - full_p[j]))
            if (not card["source_group_decision_certificate"][
                    "evidence_strengthens_predicted_class"] or
                    card["group_logit_additivity_error"] > 1e-9):
                raise ValueError("Single-account source card differs")
        if sample_error > 1e-12:
            raise ValueError("Single-account source path differs from batch")
        runs.append({
            "seed": seed, "accounts": len(ids),
            "bot_predictions": int(expected_y.sum()),
            "human_predictions": int((~expected_y).sum()),
            "all_accounts_have_signed_movement_from_neutral": True,
            "maximum_probability_replay_error": full_error,
            "maximum_source_group_logit_reconstruction_error": group_error,
            "maximum_sample_card_probability_error": sample_error,
            "neutral_probability": threshold,
            "model_sha256": runtime.neutral_anchor_model_sha256,
            "selection_lock_sha256": runtime.neutral_anchor_lock_sha256,
            "prediction_matrix_sha256": saved["predictions_sha256"],
        })
        print({"seed": seed, "accounts": len(ids),
               "probability_error": full_error,
               "group_error": group_error}, flush=True)
    CERT.write_text(json.dumps({
        "status": "all_3060_fixed_test_accounts_replayed_from_source_bound_neutral_runtime",
        "fixed_test_report_sha256": sha256_file(fixed_path),
        "test_labels_read_by_this_audit": False,
        "source_scope": "released normalized account features, typed relation tensors, and averaged tweet tensor projection",
        "external_semantic_truth_claimed": False,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
