"""Replay all Fox8 neutral-anchor test accounts from audited raw sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_twenty_neutral_anchor_runtime import (
    Fox8TwentyNeutralAnchorRuntime)


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_twenty_neutral_anchor"
CERT = ROOT / "research/certificates/fox8_twenty_neutral_anchor_runtime.json"


def main() -> None:
    report_path = FOLDER / "fixed_test_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not (report["all_seeds_above_hypbot"] and
            report["all_seeds_above_99_4_percent"]):
        raise ValueError("Fox8 neutral model failed fixed-test gate")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TwentyNeutralAnchorRuntime(ROOT, seed)
        saved = report["runs"][seed]
        path = Path(saved["prediction_matrix_path"])
        if sha256_file(path) != saved["prediction_matrix_sha256"]:
            raise ValueError("Fox8 saved test predictions changed")
        with np.load(path, allow_pickle=False) as z:
            ids = z["account_ids"].astype(str)
            probability = z["probability"].astype(float)
            pred = z["predicted_bot"].astype(bool)
            threshold = float(z["threshold"])
        cards = runtime.predict_many(ids.tolist())
        actual = np.asarray([x["probability"] for x in cards])
        actual_class = np.asarray([x["prediction"] for x in cards], dtype=bool)
        error = float(np.max(np.abs(actual-probability)))
        group_error = float(max(x["group_additivity_error"] for x in cards))
        if (len(ids) != 343 or abs(threshold-runtime.threshold) > 1e-12 or
                error > 1e-12 or group_error > 1e-9 or
                not np.array_equal(actual_class, pred) or
                not all(x["evidence_strengthens_predicted_class"]
                        for x in cards)):
            raise ValueError("Fox8 raw-source neutral prediction differs")
        runs.append({
            "seed": seed, "accounts": len(ids),
            "bot_predictions": int(pred.sum()),
            "human_predictions": int((~pred).sum()),
            "maximum_probability_replay_error": error,
            "maximum_group_logit_additivity_error": group_error,
            "all_accounts_have_signed_evidence_movement": True,
            "neutral_threshold": threshold,
            "model_sha256": runtime.neutral_model_sha256,
            "neutral_selection_lock_sha256": runtime.neutral_lock_sha256,
            "prediction_matrix_sha256": saved["prediction_matrix_sha256"],
        })
        print({"seed": seed, "accounts": len(ids),
               "probability_error": error,
               "group_error": group_error}, flush=True)
    CERT.write_text(json.dumps({
        "status": "all_1029_Fox8_test_decisions_replayed_from_raw_source_neutral_runtime",
        "fixed_test_report_sha256": sha256_file(report_path),
        "test_labels_read_by_this_audit": False,
        "semantic_or_causal_ground_truth_claimed": False,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
