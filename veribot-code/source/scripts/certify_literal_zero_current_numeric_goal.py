"""Bind the three-dataset numeric gate to the corrected BotSim source model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "research/certificates/three_dataset_current_numeric_goal.json"
OUT = ROOT / "research/certificates/three_dataset_literal_zero_numeric_goal.json"
MODEL = ROOT / "output/v3_botsim_literal_upvote_zero_12"
HYPBOT = ROOT / "output/hypbot_botsim_paired"


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(old))
    botsim = next(row for row in result["datasets"] if row["dataset"] == "botsim")
    botsim["candidate_model"] = "output/v3_botsim_literal_upvote_zero_12"
    for seed, row in enumerate(botsim["runs"]):
        candidate_path = MODEL / f"seed{seed}/test_predictions.json"
        hypbot_path = HYPBOT / f"seed{seed}/test_predictions.json"
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))["predictions"]
        hypbot = json.loads(hypbot_path.read_text(encoding="utf-8"))["predictions"]
        if ([r["account_id"] for r in candidate] !=
                [r["account_id"] for r in hypbot] or
                [r["label"] for r in candidate] != [r["label"] for r in hypbot]):
            raise ValueError("literal-zero and HypBot account/label pairs differ")
        y = np.asarray([r["label"] for r in candidate], dtype=int)
        c = np.asarray([r["prediction"] for r in candidate], dtype=int)
        threshold = float(json.loads(hypbot_path.read_text(encoding="utf-8"))
                          ["manifest"]["threshold"])
        h = np.asarray([r["probability"] >= threshold for r in hypbot], dtype=int)
        row.update({
            "test_accounts": len(y),
            "candidate_correct": int(np.count_nonzero(c == y)),
            "hypbot_correct": int(np.count_nonzero(h == y)),
            "candidate_acc": float(np.mean(c == y)),
            "hypbot_acc": float(np.mean(h == y)),
            "candidate_only_correct": int(np.count_nonzero((c == y) & (h != y))),
            "hypbot_only_correct": int(np.count_nonzero((h == y) & (c != y))),
            "candidate_predictions_sha256": sha256_file(candidate_path),
            "hypbot_predictions_sha256": sha256_file(hypbot_path),
        })
        row["candidate_strictly_above_hypbot"] = row["candidate_acc"] > row["hypbot_acc"]
        row["candidate_above_user_threshold"] = row["candidate_acc"] > .994
        if (not row["candidate_strictly_above_hypbot"] or
                not row["candidate_above_user_threshold"]):
            raise ValueError("corrected BotSim model misses numeric gate")
    botsim["candidate_mean_acc"] = float(np.mean([r["candidate_acc"] for r in botsim["runs"]]))
    botsim["hypbot_mean_acc"] = float(np.mean([r["hypbot_acc"] for r in botsim["runs"]]))
    botsim["candidate_above_official_hypbot_each_seed"] = True
    botsim["candidate_above_user_threshold_each_seed"] = True
    result["scope"] = "three_dataset_literal_zero_source_corrected_paired_numeric_goal"
    result["source_to_decision_audit_sha256"] = sha256_file(
        ROOT / "research/certificates/botsim_literal_zero_source_to_decision.json")
    result["all_nine_batch_replay_sha256"] = sha256_file(
        ROOT / "research/certificates/best_evidence_batch_literal_zero_full_test.json")
    result["all_three_datasets_above_official_hypbot_each_seed"] = all(
        item["candidate_above_official_hypbot_each_seed"] for item in result["datasets"])
    result["all_three_datasets_above_user_acc_threshold_each_seed"] = all(
        item["candidate_above_user_threshold_each_seed"] for item in result["datasets"])
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
