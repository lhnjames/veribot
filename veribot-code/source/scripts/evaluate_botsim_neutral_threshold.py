"""Lock a source-bound neutral threshold on dev, then evaluate BotSim test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_upvote_zero_dev import DATA, NATIVE, USERS
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "output/v3_fox8_botsim_neutral_threshold/dev_only.json"
FOLDER = ROOT / "output/v3_botsim_neutral_threshold"
LOCK = FOLDER / "selection_lock.json"
BASE = ROOT / "research/certificates/three_dataset_observed_means_numeric_goal.json"


def lock() -> None:
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    if (dev["test_predictions_or_metrics_computed"] is not False or
            not dev["gates"]["botsim"]["passes_predeclared_gate"]):
        raise ValueError("BotSim did not pass predeclared dev gate")
    FOLDER.mkdir(parents=True, exist_ok=True)
    rows = dev["runs"]["botsim"]
    LOCK.write_text(json.dumps({
        "scope": "BotSim_12_concept_existing_reasoner_neutral_threshold_dev_locked",
        "threshold_rule": "sigmoid(frozen_reasoner_intercept_for_all_zero_evidence)",
        "dev_screen_sha256": sha256_file(DEV),
        "test_predictions_or_metrics_used_for_selection": False,
        "runs": [{"seed": x["seed"],
                  "split_sha256": x["source_split_sha256"],
                  "reasoner_sha256": x["reasoner_sha256"],
                  "neutral_threshold": x["zero_evidence_probability"],
                  "dev_validation": x["neutral_validation"]} for x in rows]},
        indent=2) + "\n", encoding="utf-8")
    print(LOCK)


def test() -> None:
    selection = json.loads(LOCK.read_text(encoding="utf-8"))
    if (selection["scope"] !=
            "BotSim_12_concept_existing_reasoner_neutral_threshold_dev_locked" or
            selection["dev_screen_sha256"] != sha256_file(DEV) or
            selection["test_predictions_or_metrics_used_for_selection"] is not False):
        raise ValueError("BotSim neutral selection lock differs")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    baseline = next(x for x in base["datasets"] if x["dataset"] == "botsim")
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimObservedMeanRuntime(ROOT, seed)
        _, labels, ids, _, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        row = selection["runs"][seed]
        if (ids != runtime.ids or
                row["split_sha256"] != source["split_sha256"] or
                row["reasoner_sha256"] !=
                runtime.source_binding["reasoner_sha256"]):
            raise ValueError("BotSim source or reasoner differs")
        threshold = float(expit(runtime.reasoner.intercept))
        if abs(threshold-row["neutral_threshold"]) > 1e-12:
            raise ValueError("BotSim neutral threshold differs")
        test_rows = split["test"]
        concepts, _ = collapse_group_tails(runtime.reference.transform(
            runtime.facts[test_rows]), runtime.atomic_groups)
        p = runtime.reasoner.predict_proba(concepts)
        if np.any(np.abs(p-threshold) <= 1e-10):
            raise ValueError("Real account has no signed evidence movement")
        pred = p >= threshold
        metrics = evaluate(labels[test_rows], p, threshold)
        correct = int(np.count_nonzero(pred == labels[test_rows]))
        hypbot = int(baseline["runs"][seed]["hypbot_correct"])
        path = FOLDER / f"seed{seed}_fixed_test_predictions.npz"
        np.savez_compressed(path,
                            account_ids=np.asarray(ids)[test_rows],
                            probability=p, predicted_bot=pred,
                            threshold=np.asarray(threshold))
        runs.append({
            "seed": seed, "accounts": len(test_rows),
            "correct": correct, "metrics": metrics,
            "hypbot_correct_on_same_population": hypbot,
            "strictly_above_hypbot": correct > hypbot,
            "strictly_above_99_4_percent_acc": metrics["acc"] > .994,
            "score_direction_supported_by_nonzero_input": int(len(test_rows)),
            "neutral_threshold": threshold,
            "prediction_matrix_path": str(path),
            "prediction_matrix_sha256": sha256_file(path),
            "source_split_sha256": source["split_sha256"],
            "reasoner_sha256": row["reasoner_sha256"]})
        print({"seed": seed, "correct": correct, "hypbot": hypbot,
               "acc": metrics["acc"]}, flush=True)
    (FOLDER / "fixed_test_report.json").write_text(json.dumps({
        "scope": "dev_locked_BotSim_12_concept_neutral_threshold_fixed_test",
        "selection_lock_sha256": sha256_file(LOCK),
        "prior_numeric_certificate_sha256": sha256_file(BASE),
        "test_population_previously_explored_by_other_candidates": True,
        "all_seeds_above_hypbot": all(r["strictly_above_hypbot"] for r in runs),
        "all_seeds_above_99_4_percent": all(
            r["strictly_above_99_4_percent_acc"] for r in runs),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(FOLDER / "fixed_test_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("lock", "test"))
    stage = parser.parse_args().stage
    lock() if stage == "lock" else test()


if __name__ == "__main__":
    main()
