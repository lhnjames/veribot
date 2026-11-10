"""Evaluate the dev-locked BotSim shortcut ablation on paired fixed test splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from screen_botsim_observed_means_dev import ACTIVITY, DATA, NATIVE, ROOT, USERS
from verifiable_bot.botsim_literal_zero_runtime import BotSimShortcutAblatedRuntime


FOLDER = ROOT / "output/v3_botsim_shortcut_ablated_10"
DEV = FOLDER / "dev_only.json"
OUT = FOLDER / "fixed_test_report.json"
HYPBOT = ROOT / "output/hypbot_botsim_paired"
CURRENT = ROOT / "output/v3_botsim_observed_means_12"


def main() -> None:
    lock = json.loads(DEV.read_text(encoding="utf-8"))
    if (lock["test_predictions_or_metrics_accessed"] or len(lock["runs"]) != 3 or
            any(row["validation"]["errors"] > 2 for row in lock["runs"])):
        raise ValueError("candidate was not locked on dev")
    results = []
    for seed, plan in enumerate(lock["runs"]):
        _, labels, ids, _, split, source = load("botsim", DATA, USERS, seed, NATIVE)
        if (source["split_sha256"] != plan["split_sha256"] or
                sha256_file(ACTIVITY) != plan["activity_sha256"]):
            raise ValueError("source or split differs from dev lock")
        runtime = BotSimShortcutAblatedRuntime(ROOT, seed)
        test_ids = [ids[int(i)] for i in split["test"]]
        decisions = runtime.predict_many(test_ids)
        y = labels[split["test"]]
        probability = np.asarray([row["probability"] for row in decisions])
        predicted = np.asarray([row["prediction"] for row in decisions])
        metrics = evaluate(y, probability, runtime.threshold)
        hyp_path = HYPBOT / f"seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        prior_path = CURRENT / f"seed{seed}/test_predictions.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))["predictions"]
        if ([row["account_id"] for row in hyp["predictions"]] != test_ids or
                [row["label"] for row in hyp["predictions"]] != y.tolist() or
                [row["account_id"] for row in prior] != test_ids):
            raise ValueError("paired test population differs")
        hyp_threshold = float(hyp["manifest"]["threshold"])
        hyp_prediction = np.asarray([row["probability"] >= hyp_threshold
                                     for row in hyp["predictions"]], dtype=int)
        prior_prediction = np.asarray([row["prediction"] for row in prior])
        path = FOLDER / f"seed{seed}/test_predictions.json"
        path.write_text(json.dumps({
            "seed": seed, "dev_lock_sha256": sha256_file(DEV),
            "predictions": [{"account_id": uid, "label": int(label),
                             "probability": float(p), "prediction": int(pred)}
                            for uid, label, p, pred in zip(
                                test_ids, y, probability, predicted, strict=True)]
        }, indent=2) + "\n", encoding="utf-8")
        hyp_correct = int(np.count_nonzero(hyp_prediction == y))
        passed = metrics["acc"] > .994 and metrics["acc"] > hyp_correct / len(y)
        results.append({
            "seed": seed, "test_accounts": len(y), "split_sha256": source["split_sha256"],
            "threshold": runtime.threshold, "validation": plan["validation"], "test": metrics,
            "hypbot_correct": hyp_correct,
            "current_model_correct": int(np.count_nonzero(prior_prediction == y)),
            "shortcut_ablated_correct": int(np.count_nonzero(predicted == y)),
            "changed_decisions_from_current": int(np.count_nonzero(predicted != prior_prediction)),
            "strict_numeric_gate_pass": bool(passed),
            "test_predictions_sha256": sha256_file(path),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
            "current_predictions_sha256": sha256_file(prior_path),
        })
        print({"seed": seed, "correct": results[-1]["shortcut_ablated_correct"],
               "current": results[-1]["current_model_correct"],
               "hypbot": hyp_correct, "pass": passed}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_dev_selected_shortcut_ablation_fixed_test",
        "dev_lock_sha256": sha256_file(DEV),
        "all_seeds_strict_numeric_gate_pass": all(row["strict_numeric_gate_pass"] for row in results),
        "prior_test_population_exploration": True,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
