"""Lock a dev-selected 16-coefficient residual, then audit its fixed test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_eight_projection_low_capacity_residual_dev import content_columns
from screen_mgtab_supervised_bounded_linear_residual_dev import CAP, fit, bounded
from verifiable_bot.mgtab_reduced_projection_runtime import MGTABReducedProjectionRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_supervised_bounded_linear_residual"
SCREEN = OUT / "dev_only.json"
PARENT = ROOT / "output/v3_mgtab_reduced_projection_ab_group/selection_lock.json"
LOCK = OUT / "selection_lock.json"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def selected() -> tuple[dict, float]:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (screen["scope"] != "mgtab_supervised_bounded_sixteen_coefficient_residual_dev_only"
            or screen["parent_selection_lock_sha256"] != sha256_file(PARENT)
            or screen["test_predictions_or_metrics_computed"] is not False
            or screen["test_labels_used_for_selection"] is not False
            or len(screen["runs"]) != 3 or not screen["gate_passing_candidates"]):
        raise ValueError("Supervised residual dev screen or gate differs")
    choice = max(screen["gate_passing_candidates"],
                 key=lambda row: (row["mean_dev_macro_f1"], row["l2"]))
    return screen, float(choice["l2"])


def lock() -> None:
    screen, l2 = selected()
    hashes = {}
    thresholds = {}
    for seed in (0, 1, 2):
        runtime = MGTABReducedProjectionRuntime(ROOT, seed)
        _, labels, _, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if source["split_sha256"] != screen["runs"][seed]["split_sha256"]:
            raise ValueError("Official split differs")
        contract_path = ROOT / ("output/v3_mgtab_ab_group_additive_tree/"
                                f"seed{seed}/ab_direction_contract.npz")
        with np.load(contract_path, allow_pickle=False) as contract:
            b = contract["b_indices"].astype(int)
        x_b = content_columns(runtime.evidence_matrix(b))
        named_b = runtime.predict_rows(b)["named_logit"]
        weights, _ = fit(x_b, named_b, labels[b], l2)
        val = split["val"]
        probability = expit(runtime.predict_rows(val)["named_logit"] +
                            bounded(content_columns(runtime.evidence_matrix(val)), weights))
        threshold = best_validation_threshold(probability, labels[val])
        metric = evaluate(labels[val], probability, threshold)
        expected = next(row for row in screen["runs"][seed]["candidates"]
                        if row["l2"] == l2)
        if (abs(threshold - expected["threshold"]) > 1e-12
                or any(abs(metric[k] - expected["validation"][k]) > 1e-12
                       for k in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Dev result did not replay")
        path = OUT / f"seed{seed}/weights.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=weights.astype(np.float64),
                            l2=np.asarray(l2), cap=np.asarray(CAP),
                            threshold=np.asarray(threshold))
        hashes[str(seed)] = sha256_file(path)
        thresholds[str(seed)] = threshold
    save_json(LOCK, {
        "scope": "mgtab_supervised_bounded_linear_residual_dev_lock",
        "selection_rule": "highest mean dev Macro-F1 among candidates passing the predeclared dev gate",
        "screen_sha256": sha256_file(SCREEN),
        "parent_selection_lock_sha256": sha256_file(PARENT),
        "selected_l2": l2,
        "trainable_residual_coefficients": 16,
        "residual_absolute_logit_cap": CAP,
        "threshold_by_seed": thresholds,
        "weights_sha256_by_seed": hashes,
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in screen["runs"]},
        "test_metric_or_prediction_used_to_create_lock": False,
        "prior_results_on_same_fixed_test_population_already_observed": True,
    })
    print({"selected_l2": l2, "selection_lock": str(LOCK)}, flush=True)


def test() -> None:
    screen, l2 = selected()
    selection = json.loads(LOCK.read_text(encoding="utf-8"))
    if (selection["scope"] != "mgtab_supervised_bounded_linear_residual_dev_lock"
            or selection["screen_sha256"] != sha256_file(SCREEN)
            or selection["parent_selection_lock_sha256"] != sha256_file(PARENT)
            or selection["selected_l2"] != l2
            or selection["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("Dev selection lock differs")
    results = []
    for seed in (0, 1, 2):
        runtime = MGTABReducedProjectionRuntime(ROOT, seed)
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if source["split_sha256"] != selection["split_sha256_by_seed"][str(seed)]:
            raise ValueError("Official split differs")
        weight_path = OUT / f"seed{seed}/weights.npz"
        if sha256_file(weight_path) != selection["weights_sha256_by_seed"][str(seed)]:
            raise ValueError("Frozen coefficient file differs")
        with np.load(weight_path, allow_pickle=False) as frozen:
            weights = frozen["weights"].astype(float)
            threshold = float(frozen["threshold"])
            if (weights.shape != (16,) or float(frozen["l2"]) != l2
                    or float(frozen["cap"]) != CAP
                    or threshold != selection["threshold_by_seed"][str(seed)]):
                raise ValueError("Frozen residual contract differs")
        rows = split["test"]
        named = runtime.predict_rows(rows)["named_logit"]
        residual = bounded(content_columns(runtime.evidence_matrix(rows)), weights)
        probability = expit(named + residual)
        metric = evaluate(labels[rows], probability, threshold)
        hyp = json.loads((ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json")
                         .read_text(encoding="utf-8"))
        if (hyp["manifest"]["split_sha256"] != source["split_sha256"]
                or [r["account_id"] for r in hyp["predictions"]] !=
                [ids[int(i)] for i in rows]
                or [r["label"] for r in hyp["predictions"]] !=
                labels[rows].astype(int).tolist()):
            raise ValueError("Paired HypBot accounts or labels differ")
        hyp_pred = np.asarray([r["probability"] >= hyp["manifest"]["threshold"]
                               for r in hyp["predictions"]], dtype=bool)
        pred = probability >= threshold
        named_pred = expit(named) >= threshold
        correct = pred == labels[rows]
        hyp_correct = hyp_pred == labels[rows]
        run_dir = OUT / f"seed{seed}"
        save_json(run_dir / "test_predictions.json", {
            "manifest": {"dataset": "mgtab", "seed": seed,
                         "selection_lock_sha256": sha256_file(LOCK),
                         "weights_sha256": sha256_file(weight_path),
                         "split_sha256": source["split_sha256"],
                         "threshold": threshold},
            "predictions": [{"account_id": ids[int(i)],
                             "label": int(labels[i]),
                             "probability": float(probability[j]),
                             "named_logit": float(named[j]),
                             "bounded_linear_residual": float(residual[j])}
                            for j, i in enumerate(rows)],
        })
        result = {
            "seed": seed, "test": metric,
            "dev": next(c["validation"] for c in screen["runs"][seed]["candidates"]
                        if c["l2"] == l2),
            "hypbot_acc": float(hyp_correct.mean()),
            "only_model_correct": int(np.count_nonzero(correct & ~hyp_correct)),
            "only_hypbot_correct": int(np.count_nonzero(~correct & hyp_correct)),
            "named_only_correct": int(np.count_nonzero(named_pred == labels[rows])),
            "residual_changes_class": int(np.count_nonzero(pred != named_pred)),
            "residual_rescues_named_only_errors": int(np.count_nonzero(
                correct & (named_pred != labels[rows]))),
            "residual_harms_named_only_correct": int(np.count_nonzero(
                ~correct & (named_pred == labels[rows]))),
            "above_hypbot": bool(metric["acc"] > hyp_correct.mean()),
            "above_user_91_2_percent": bool(metric["acc"] > .912),
            "weights_sha256": sha256_file(weight_path),
            "predictions_sha256": sha256_file(run_dir / "test_predictions.json"),
        }
        results.append(result)
        print(result, flush=True)
    save_json(OUT / "fixed_test_report.json", {
        "scope": "mgtab_supervised_bounded_linear_residual_fixed_test_after_dev_lock",
        "selection_lock_sha256": sha256_file(LOCK),
        "all_seeds_above_hypbot_and_91_2_percent": all(
            r["above_hypbot"] and r["above_user_91_2_percent"] for r in results),
        "prior_results_on_same_fixed_test_population_already_observed": True,
        "runs": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "test"), required=True)
    args = parser.parse_args()
    if args.stage == "lock":
        lock()
    else:
        test()


if __name__ == "__main__":
    main()
