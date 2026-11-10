"""Evaluate frozen evidence-only Fox8/BotSim candidates on official test rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from lock_group_additive_schema_v2_candidates import _design, OUT as LOCK_DIR
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = LOCK_DIR / "selection_lock.json"
SCHEMA_PATH = ROOT / "research/evidence_schema_v2.json"
OUT = LOCK_DIR / "fixed_test_report.json"


def _test_context(facts: np.ndarray, names: list[str], run: dict) -> np.ndarray | None:
    if run["reference_context_rule"] == "pooled_unlabeled_training_reference":
        return None
    if run["reference_context_rule"] != \
            "unlabeled_training_tertiles_with_global_fallback":
        raise ValueError("unknown frozen reference context policy")
    cuts = np.asarray(run["train_only_context_quantiles"], dtype=float)
    if cuts.shape != (2,) or not np.isfinite(cuts).all():
        raise ValueError("invalid frozen context cuts")
    values = facts[:, names.index(run["reference_context_fact"])]
    return np.asarray([f"activity_{i}" for i in
                       np.searchsorted(cuts, values, side="right")])


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if (lock["status"] != "development_locked_candidate_not_independently_tested" or
            lock["schema_sha256"] != sha256_file(SCHEMA_PATH) or
            lock["test_predictions_or_metrics_accessed"] is not False):
        raise ValueError("candidate lock differs")
    baseline_certificate_path = ROOT / "research/certificates/three_dataset_observed_means_numeric_goal.json"
    baseline_certificate = json.loads(baseline_certificate_path.read_text(encoding="utf-8"))
    reports = {}
    for dataset in ("fox8", "botsim"):
        dataset_runs = []
        prior_baseline = next(item for item in baseline_certificate["datasets"]
                              if item["dataset"] == dataset)
        for seed in (0, 1, 2):
            frozen = lock["runs"][dataset][seed]
            raw, y, ids, names, split, source = _source_facts(dataset, seed)
            if (frozen["seed"] != seed or
                    frozen["split_sha256"] != source["split_sha256"] or
                    frozen["source_files_sha256"] != source["files_sha256"]):
                raise ValueError("official test population differs from dev lock")
            raw = _missing_means(dataset, raw, names, ids)
            selected_names = frozen["selected_fact_names"]
            facts = raw[:, [names.index(name) for name in selected_names]]
            test = split["test"]
            reference_path = ROOT / frozen["reference_path"]
            direction_path = ROOT / frozen["direction_path"]
            model_paths = [ROOT / value for value in frozen["model_paths"]]
            if (sha256_file(reference_path) != frozen["reference_sha256"] or
                    sha256_file(direction_path) != frozen["direction_sha256"] or
                    [sha256_file(path) for path in model_paths] !=
                    frozen["model_sha256"]):
                raise ValueError("frozen model/reference hash differs")
            reference = LabelBlindReference.load(reference_path)
            directions = json.loads(direction_path.read_text(encoding="utf-8"))
            if reference.version != frozen["reference_version"] or \
                    list(reference.names) != selected_names or len(directions) != 2:
                raise ValueError("frozen evidence reference or directions differ")
            context = _test_context(facts[test], selected_names, frozen)
            evidence = reference.transform(facts[test], context)
            logits = []
            for path, direction in zip(model_paths, directions, strict=True):
                if list(direction["directions"]) != selected_names or \
                        direction["reference_version"] != reference.version:
                    raise ValueError("frozen direction and evidence vocabulary differ")
                model = joblib.load(path)
                logits.append(np.asarray(model.booster_.predict(
                    _design(evidence, direction, selected_names), raw_score=True)))
            probability = expit((logits[0] + logits[1]) / 2)
            threshold = float(frozen["classification_threshold_selected_on_dev"])
            metric = evaluate(y[test], probability, threshold)

            hyp_path = ROOT / f"output/hypbot_{dataset}_paired/seed{seed}/test_predictions.json"
            hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
            h = hyp["manifest"]
            if (h["split_sha256"] != source["split_sha256"] or
                    h["auto_k"] is not False or h["num_sectors"] != 2 or
                    sha256_file(hyp_path.parent / "model.pt") != h["model_sha256"] or
                    [str(item["account_id"]) for item in hyp["predictions"]] !=
                    [ids[int(i)] for i in test] or
                    [int(item["label"]) for item in hyp["predictions"]] !=
                    y[test].tolist()):
                raise ValueError("official HypBot baseline population or model differs")
            hyp_probability = np.asarray([float(item["probability"])
                                          for item in hyp["predictions"]])
            hyp_pred = hyp_probability >= float(h["threshold"])
            hyp_correct = int(np.count_nonzero(hyp_pred == y[test]))
            baseline_run = prior_baseline["runs"][seed]
            if baseline_run["hypbot_correct"] != hyp_correct or \
                    baseline_run["split_sha256"] != source["split_sha256"]:
                raise ValueError("HypBot baseline certificate does not replay")
            candidate_pred = probability >= threshold
            paired = {
                "candidate_only_correct": int(np.count_nonzero(
                    (candidate_pred == y[test]) & (hyp_pred != y[test]))),
                "hypbot_only_correct": int(np.count_nonzero(
                    (candidate_pred != y[test]) & (hyp_pred == y[test]))),
                "both_correct": int(np.count_nonzero(
                    (candidate_pred == y[test]) & (hyp_pred == y[test]))),
                "both_wrong": int(np.count_nonzero(
                    (candidate_pred != y[test]) & (hyp_pred != y[test]))),
            }
            prediction_path = LOCK_DIR / dataset / f"seed{seed}" / "test_predictions.json"
            prediction_path.write_text(json.dumps({
                "dataset": dataset, "seed": seed,
                "selection_lock_sha256": sha256_file(LOCK_PATH),
                "split_sha256": source["split_sha256"],
                "threshold": threshold,
                "predictions": [
                    {"account_id": ids[int(i)], "label": int(y[int(i)]),
                     "probability": float(p), "class": int(p >= threshold)}
                    for i, p in zip(test, probability, strict=True)]
            }, indent=2) + "\n", encoding="utf-8")
            dataset_runs.append({
                "seed": seed, "split_sha256": source["split_sha256"],
                "test_accounts": len(test), "candidate_correct": len(test)-metric["errors"],
                "candidate_metric": metric, "hypbot_correct": hyp_correct,
                "strict_acc_threshold": .994,
                "passes_acc_threshold": metric["acc"] > .994,
                "strictly_above_official_hypbot": len(test)-metric["errors"] > hyp_correct,
                "paired": paired,
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "prediction_sha256": sha256_file(prediction_path),
                "hypbot_prediction_sha256": sha256_file(hyp_path),
            })
            print({"dataset": dataset, "seed": seed,
                   "candidate_correct": len(test)-metric["errors"],
                   "hypbot_correct": hyp_correct,
                   "acc_gt_99.4": metric["acc"] > .994}, flush=True)
        reports[dataset] = dataset_runs
    result = {
        "scope": "frozen_schema_v2_evidence_group_additive_official_fixed_test",
        "selection_lock_sha256": sha256_file(LOCK_PATH),
        "hypbot_baseline_certificate_sha256": sha256_file(baseline_certificate_path),
        "test_population_previously_explored_by_other_candidates": True,
        "independent_semantic_truth_validated": False,
        "all_six_strict_numeric_gates_pass": all(
            row["passes_acc_threshold"] and row["strictly_above_official_hypbot"]
            for rows in reports.values() for row in rows),
        "runs": reports,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
