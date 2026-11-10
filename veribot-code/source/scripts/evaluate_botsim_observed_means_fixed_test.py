"""Evaluate dev-locked BotSim model with undefined source averages unavailable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from screen_botsim_observed_means_dev import ACTIVITY, DATA, FOLDER, NATIVE, OUT as DEV, ROOT, USERS
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime


OUT = FOLDER / "fixed_test_report.json"
HYPBOT = ROOT / "output/hypbot_botsim_paired"
PREVIOUS = ROOT / "output/v3_botsim_literal_upvote_zero_12"


def main() -> None:
    locked = json.loads(DEV.read_text(encoding="utf-8"))
    if (locked["test_predictions_or_metrics_accessed"] is not False or
            locked["fixed_l2_from_prior_validation_lock"] != .01 or
            len(locked["runs"]) != 3 or
            any(run["validation"]["errors"] != 0 for run in locked["runs"])):
        raise ValueError("observed-mean model was not development locked")
    runs = []
    for seed, plan in enumerate(locked["runs"]):
        x, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        if (plan["seed"] != seed or plan["split_sha256"] != source["split_sha256"] or
                plan["activity_sha256"] != sha256_file(ACTIVITY)):
            raise ValueError("official source or split differs from development lock")
        runtime = BotSimObservedMeanRuntime(ROOT, seed)
        test_ids = [ids[int(i)] for i in split["test"]]
        decisions = runtime.predict_many(test_ids)
        probability = np.asarray([row["probability"] for row in decisions], dtype=float)
        prediction = np.asarray([row["prediction"] for row in decisions], dtype=int)
        y = labels[split["test"]]
        if abs(runtime.threshold - plan["threshold"]) > 1e-12:
            raise ValueError("frozen threshold differs")
        metrics = evaluate(y, probability, runtime.threshold)
        hyp_path = HYPBOT / f"seed{seed}/test_predictions.json"
        old_path = PREVIOUS / f"seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        old = json.loads(old_path.read_text(encoding="utf-8"))["predictions"]
        if ([row["account_id"] for row in hyp["predictions"]] != test_ids or
                [row["label"] for row in hyp["predictions"]] != y.tolist() or
                [row["account_id"] for row in old] != test_ids):
            raise ValueError("paired fixed test account IDs or labels differ")
        hyp_threshold = float(hyp["manifest"]["threshold"])
        hyp_class = np.asarray([row["probability"] >= hyp_threshold
                                for row in hyp["predictions"]], dtype=int)
        old_class = np.asarray([row["prediction"] for row in old], dtype=int)
        passed = bool(metrics["acc"] > .994 and metrics["acc"] >
                      float(np.mean(hyp_class == y)))
        saved = [{"account_id": uid, "label": int(label),
                  "probability": float(p), "prediction": int(pred)}
                 for uid, label, p, pred in zip(test_ids, y, probability,
                                                prediction, strict=True)]
        path = FOLDER / f"seed{seed}/test_predictions.json"
        path.write_text(json.dumps({"seed": seed,
                                    "dev_lock_sha256": sha256_file(DEV),
                                    "predictions": saved}, indent=2) + "\n",
                        encoding="utf-8")
        runs.append({
            "seed": seed, "test_accounts": len(test_ids),
            "split_sha256": source["split_sha256"],
            "threshold": runtime.threshold,
            "validation": plan["validation"], "test": metrics,
            "hypbot_correct": int(np.count_nonzero(hyp_class == y)),
            "previous_correct": int(np.count_nonzero(old_class == y)),
            "observed_mean_correct": int(np.count_nonzero(prediction == y)),
            "previous_vs_observed_mean_changed_classes": int(np.count_nonzero(
                old_class != prediction)),
            "strictly_above_hypbot_and_99_4_percent": passed,
            "test_predictions_sha256": sha256_file(path),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
            "previous_predictions_sha256": sha256_file(old_path),
        })
        print({"seed": seed, "test_acc": metrics["acc"],
               "hypbot_correct": runs[-1]["hypbot_correct"],
               "previous_correct": runs[-1]["previous_correct"],
               "new_correct": runs[-1]["observed_mean_correct"],
               "passed": passed}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_observed_means_twelve_concepts_fixed_test_after_dev_lock",
        "dev_lock_sha256": sha256_file(DEV),
        "all_seeds_strict_numeric_gate_pass": all(
            row["strictly_above_hypbot_and_99_4_percent"] for row in runs),
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
