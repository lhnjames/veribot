"""Paired fixed-test diagnostic of the dev-locked two-cue BotSim removal."""

from __future__ import annotations

import json

import numpy as np

from gate0_shallow_tree import evaluate, load, sha256_file
from screen_botsim_observed_means_dev import ACTIVITY, DATA, NATIVE, ROOT, USERS
from verifiable_bot.botsim_literal_zero_runtime import (
    BotSimTwoEngagementCuesRemovedRuntime)


FOLDER = ROOT / "output/v3_botsim_engagement_shortcuts/selected"
DEV = FOLDER / "dev_only.json"
OUT = FOLDER / "fixed_test_report.json"
HYPBOT = ROOT / "output/hypbot_botsim_paired"
CURRENT = ROOT / "output/v3_botsim_observed_means_12"


def main() -> None:
    lock = json.loads(DEV.read_text(encoding="utf-8"))
    screen = ROOT / "output/v3_botsim_engagement_shortcuts/dev_only.json"
    if (lock["scope"] != "BotSim_two_engagement_cues_removed_dev_locked" or
            lock["screen_sha256"] != sha256_file(screen) or
            lock["test_predictions_or_metrics_accessed"] is not False or
            set(lock["excluded_source_facts"]) != {
                "avg_upvote_ratio", "avg_comment_score"} or
            any(row["validation"]["errors"] > 2 for row in lock["runs"])):
        raise ValueError("Two-cue model was not locked on dev")
    results = []
    for seed, plan in enumerate(lock["runs"]):
        _, labels, ids, _, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        if (source["split_sha256"] != plan["split_sha256"] or
                sha256_file(ACTIVITY) != plan["activity_sha256"]):
            raise ValueError("Source or official split differs")
        runtime = BotSimTwoEngagementCuesRemovedRuntime(ROOT, seed)
        test_ids = [ids[int(i)] for i in split["test"]]
        decisions = runtime.predict_many(test_ids)
        y = labels[split["test"]]
        p = np.asarray([row["probability"] for row in decisions])
        pred = np.asarray([row["prediction"] for row in decisions])
        metrics = evaluate(y, p, runtime.threshold)
        hyp_path = HYPBOT / f"seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        prior_path = CURRENT / f"seed{seed}/test_predictions.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))["predictions"]
        if ([row["account_id"] for row in hyp["predictions"]] != test_ids or
                [row["label"] for row in hyp["predictions"]] != y.tolist() or
                [row["account_id"] for row in prior] != test_ids):
            raise ValueError("Paired fixed population differs")
        hyp_threshold = float(hyp["manifest"]["threshold"])
        hyp_pred = np.asarray([row["probability"] >= hyp_threshold
                               for row in hyp["predictions"]], dtype=int)
        prior_pred = np.asarray([row["prediction"] for row in prior])
        path = FOLDER / f"seed{seed}/test_predictions.json"
        path.write_text(json.dumps({
            "seed": seed, "dev_lock_sha256": sha256_file(DEV),
            "predictions": [{"account_id": uid, "label": int(label),
                             "probability": float(value), "prediction": int(cls)}
                            for uid, label, value, cls in zip(
                                test_ids, y, p, pred, strict=True)]
        }, indent=2) + "\n", encoding="utf-8")
        correct = int(np.count_nonzero(pred == y))
        hyp_correct = int(np.count_nonzero(hyp_pred == y))
        passed = bool(metrics["acc"] > .994 and correct > hyp_correct)
        row = {
            "seed": seed, "test_accounts": len(y),
            "split_sha256": source["split_sha256"],
            "threshold": runtime.threshold,
            "validation": plan["validation"], "test": metrics,
            "hypbot_correct": hyp_correct,
            "current_model_correct": int(np.count_nonzero(prior_pred == y)),
            "two_cue_removed_correct": correct,
            "changed_decisions_from_current": int(np.count_nonzero(
                pred != prior_pred)),
            "strict_numeric_gate_pass": passed,
            "test_predictions_sha256": sha256_file(path),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
            "current_predictions_sha256": sha256_file(prior_path),
        }
        results.append(row)
        print({"seed": seed, "correct": correct, "hypbot": hyp_correct,
               "pass": passed}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "BotSim_two_engagement_cues_removed_fixed_test_diagnostic",
        "dev_lock_sha256": sha256_file(DEV),
        "all_seeds_strict_numeric_gate_pass": all(
            row["strict_numeric_gate_pass"] for row in results),
        "prior_test_population_exploration": True,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
