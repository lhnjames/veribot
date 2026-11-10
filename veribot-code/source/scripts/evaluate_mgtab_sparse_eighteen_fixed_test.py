"""One paired fixed-test evaluation of the dev-locked sparse 18-summary model."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_sparse_A_weighted_group_summaries/selected"
LOCK = FOLDER / "dev_lock.json"
OUT = FOLDER / "fixed_test_report.json"


def scores(model, design: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(model.booster_.predict(design, raw_score=True))
    no_content = design.copy()
    no_content[:, -2:] = 0.0
    named = np.asarray(model.booster_.predict(no_content, raw_score=True))
    content = full - named
    return expit(named + np.clip(content, -cap, cap)), content


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock["test_prediction_or_metric_accessed"] or
            lock["selected_top_k_per_group_direction"] != 8 or
            len(lock["runs"]) != 3):
        raise ValueError("sparse candidate was not frozen on dev")
    results = []
    for seed, plan in enumerate(lock["runs"]):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir,
                                               None, seed)
        if (ids != runtime.ids or source["split_sha256"] != plan["split_sha256"]
                or runtime.reference.version != plan["reference_version"]
                or sha256_file(runtime.run_dir /
                    "label_blind_reference.json.gz") != plan[
                        "original_reference_sha256"]
                or runtime.lock["source_features_sha256"] != plan[
                    "source_features_sha256"]):
            raise ValueError("MGTAB official source or reference differs")
        folder = FOLDER / f"seed{seed}"
        model_path = folder / "model.joblib"
        weights_path = folder / "source_weights.npz"
        if (sha256_file(model_path) != plan["model_sha256"] or
                sha256_file(weights_path) != plan["weights_sha256"]):
            raise ValueError("locked model or source weights differ")
        model = joblib.load(model_path)
        model.set_params(n_jobs=1)
        with np.load(weights_path, allow_pickle=False) as data:
            weights = data["weights"].astype(float)
            names = data["fact_names"].astype(str).tolist()
        if (weights.shape != (18, 300) or names != runtime.names or
                int(np.count_nonzero(weights)) != plan[
                    "nonzero_fact_tail_slots"]):
            raise ValueError("locked source-summary inputs differ")
        dev = split["val"]
        dev_prob, _ = scores(model, evidence_design(runtime, dev) @ weights.T,
                             lock["content_logit_cap"])
        dev_metrics = evaluate(labels[dev], dev_prob, plan["threshold"])
        if dev_metrics != plan["validation"]:
            raise ValueError("locked candidate does not replay development data")
        test = split["test"]
        probability, content = scores(
            model, evidence_design(runtime, test) @ weights.T,
            lock["content_logit_cap"])
        metrics = evaluate(labels[test], probability, plan["threshold"])
        test_ids = [ids[int(row)] for row in test]
        hyp_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        if (hyp["manifest"]["split_sha256"] != source["split_sha256"] or
                [row["account_id"] for row in hyp["predictions"]] != test_ids or
                [row["label"] for row in hyp["predictions"]] != labels[test].tolist()):
            raise ValueError("paired HypBot account alignment differs")
        hyp_threshold = float(hyp["manifest"]["threshold"])
        hyp_prediction = np.asarray([row["probability"] >= hyp_threshold
                                     for row in hyp["predictions"]])
        prediction = probability >= plan["threshold"]
        correct = prediction == labels[test]
        hyp_correct = hyp_prediction == labels[test]
        path = folder / "test_predictions.json"
        path.write_text(json.dumps({
            "manifest": {
                "seed": seed, "dev_lock_sha256": sha256_file(LOCK),
                "model_sha256": plan["model_sha256"],
                "weights_sha256": plan["weights_sha256"],
                "split_sha256": source["split_sha256"],
                "threshold": plan["threshold"],
            },
            "predictions": [
                {"account_id": uid, "label": int(label),
                 "probability": float(p), "prediction": int(pred),
                 "content_group_raw_logit": float(effect)}
                for uid, label, p, pred, effect in zip(
                    test_ids, labels[test], probability, prediction,
                    content, strict=True)],
        }, indent=2) + "\n", encoding="utf-8")
        results.append({
            "seed": seed, "test_accounts": len(test),
            "validation": dev_metrics, "test": metrics,
            "hypbot_correct": int(hyp_correct.sum()),
            "sparse_18_correct": int(correct.sum()),
            "only_sparse_18_correct": int(np.count_nonzero(correct & ~hyp_correct)),
            "only_hypbot_correct": int(np.count_nonzero(~correct & hyp_correct)),
            "above_hypbot": bool(metrics["acc"] > hyp_correct.mean()),
            "above_91_2_percent": bool(metrics["acc"] > .912),
            "content_cap_active_test_accounts": int(np.count_nonzero(
                np.abs(content) > lock["content_logit_cap"])),
            "test_predictions_sha256": sha256_file(path),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
        })
        print({"seed": seed, "correct": results[-1]["sparse_18_correct"],
               "hypbot": results[-1]["hypbot_correct"],
               "strict_numeric_gate": results[-1]["above_hypbot"]
                   and results[-1]["above_91_2_percent"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "one_fixed_test_after_sparse_18_dev_lock",
        "dev_lock_sha256": sha256_file(LOCK),
        "all_seeds_above_hypbot_and_91_2_percent": all(
            row["above_hypbot"] and row["above_91_2_percent"]
            for row in results),
        "prior_test_population_exploration": True,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
