"""Evaluate dev-locked 20-fact MGTAB models on paired official fixed tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_twenty_source_facts/selected"
LOCK = FOLDER / "dev_lock.json"
OUT = FOLDER / "fixed_test_report.json"


def scores(model, design: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(model.booster_.predict(design, raw_score=True))
    named = design.copy()
    named[:, 12:20] = 0.0
    named[:, 32:40] = 0.0
    named_logit = np.asarray(model.booster_.predict(named, raw_score=True))
    residual = full - named_logit
    return expit(named_logit + np.clip(residual, -cap, cap)), residual


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock["test_predictions_or_metrics_accessed"] or
            lock["selected_quota"] != [4, 8, 8] or len(lock["runs"]) != 3):
        raise ValueError("20-fact candidate is not dev locked")
    results = []
    for seed, plan in enumerate(lock["runs"]):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or source["split_sha256"] != plan["split_sha256"] or
                sha256_file(runtime.run_dir / "label_blind_reference.json.gz") !=
                plan["original_reference_sha256"]):
            raise ValueError("MGTAB source, split or train-only reference differs")
        model_path = FOLDER / f"seed{seed}/model.joblib"
        if sha256_file(model_path) != plan["model_sha256"]:
            raise ValueError("frozen 20-fact model differs")
        model = joblib.load(model_path)
        model.set_params(n_jobs=1)
        selected = np.asarray(plan["selected_fact_indices"], dtype=int)
        columns = np.r_[selected, selected + 150]
        dev = split["val"]
        dev_probability, _ = scores(
            model, evidence_design(runtime, dev)[:, columns],
            lock["content_output_absolute_logit_cap"])
        dev_metrics = evaluate(labels[dev], dev_probability, plan["threshold"])
        if dev_metrics != plan["validation"]:
            raise ValueError("saved 20-fact model fails development replay")
        test = split["test"]
        probability, residual = scores(
            model, evidence_design(runtime, test)[:, columns],
            lock["content_output_absolute_logit_cap"])
        metrics = evaluate(labels[test], probability, plan["threshold"])
        test_ids = [ids[int(i)] for i in test]
        hyp_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        if (hyp["manifest"]["split_sha256"] != source["split_sha256"] or
                [row["account_id"] for row in hyp["predictions"]] != test_ids or
                [row["label"] for row in hyp["predictions"]] != labels[test].tolist()):
            raise ValueError("paired HypBot account or label alignment differs")
        hyp_threshold = float(hyp["manifest"]["threshold"])
        hyp_prediction = np.asarray([
            row["probability"] >= hyp_threshold for row in hyp["predictions"]])
        predicted = probability >= plan["threshold"]
        correct = predicted == labels[test]
        hyp_correct = hyp_prediction == labels[test]
        prediction_path = FOLDER / f"seed{seed}/test_predictions.json"
        prediction_path.write_text(json.dumps({
            "manifest": {
                "seed": seed, "dev_lock_sha256": sha256_file(LOCK),
                "model_sha256": sha256_file(model_path),
                "split_sha256": source["split_sha256"],
                "threshold": plan["threshold"],
            },
            "predictions": [
                {"account_id": uid, "label": int(label),
                 "probability": float(p), "prediction": int(pred),
                 "content_group_raw_logit": float(r)}
                for uid, label, p, pred, r in zip(
                    test_ids, labels[test], probability, predicted, residual, strict=True)]
        }, indent=2) + "\n", encoding="utf-8")
        results.append({
            "seed": seed, "test_accounts": len(test),
            "validation": dev_metrics, "test": metrics,
            "hypbot_correct": int(hyp_correct.sum()),
            "twenty_fact_correct": int(correct.sum()),
            "only_twenty_fact_correct": int(np.count_nonzero(correct & ~hyp_correct)),
            "only_hypbot_correct": int(np.count_nonzero(~correct & hyp_correct)),
            "above_hypbot": bool(metrics["acc"] > hyp_correct.mean()),
            "above_91_2_percent": bool(metrics["acc"] > .912),
            "model_sha256": sha256_file(model_path),
            "test_predictions_sha256": sha256_file(prediction_path),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
        })
        print({"seed": seed, "correct": results[-1]["twenty_fact_correct"],
               "hypbot": results[-1]["hypbot_correct"],
               "pass": results[-1]["above_hypbot"] and
               results[-1]["above_91_2_percent"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_twenty_source_facts_fixed_test_after_dev_lock",
        "dev_lock_sha256": sha256_file(LOCK),
        "all_seeds_above_hypbot_and_91_2_percent": all(
            row["above_hypbot"] and row["above_91_2_percent"] for row in results),
        "prior_test_population_exploration": True,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
