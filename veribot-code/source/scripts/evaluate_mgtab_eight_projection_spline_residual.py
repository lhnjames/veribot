"""Exploratory fixed-test check of predeclared additive MGTAB content splines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_eight_projection_low_capacity_residual_dev import content_columns
from verifiable_bot.mgtab_reduced_projection_runtime import MGTABReducedProjectionRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_eight_projection_spline_residual"
SCREEN = ROOT / "output/v3_mgtab_eight_projection_low_capacity_residual/dev_only.json"
PARENT = ROOT / "output/v3_mgtab_reduced_projection_ab_group/selection_lock.json"
LOCK = OUT / "selection_lock.json"
FAMILY = "additive_quadratic_splines_three_knots"
CAP = 3.0


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def lock() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (screen["scope"] != "mgtab_eight_projection_low_capacity_residual_dev_only"
            or screen["test_predictions_or_metrics_computed"] is not False
            or screen["parent_selection_lock_sha256"] != sha256_file(PARENT)):
        raise ValueError("Low-capacity validation screen differs")
    passing = [row for row in screen["gate_passing_candidates"]
               if row["family"] == FAMILY and row["trainable_coefficients"] == 48]
    if not passing:
        raise ValueError("No additive 48-coefficient spline passed dev gate")
    chosen = max(passing, key=lambda row: (row["mean_dev_macro_f1"], -row["alpha"]))
    alpha = float(chosen["alpha"])
    rows = [next(row for row in run["candidates"]
                 if row["family"] == FAMILY and row["alpha"] == alpha)
            for run in screen["runs"]]
    save_json(LOCK, {
        "scope": "mgtab_eight_projection_additive_spline_exploratory_lock",
        "selection_rule": "largest mean dev Macro-F1 among predeclared 48-coefficient additive spline candidates passing dev gate",
        "screen_sha256": sha256_file(SCREEN),
        "parent_selection_lock_sha256": sha256_file(PARENT),
        "family": FAMILY, "selected_alpha": alpha,
        "trainable_residual_coefficients": 48,
        "residual_absolute_logit_cap": CAP,
        "threshold_by_seed": {str(run["seed"]): row["validation_threshold"]
                              for run, row in zip(screen["runs"], rows, strict=True)},
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in screen["runs"]},
        "test_metric_or_prediction_used_to_choose_alpha": False,
        "decision_to_examine_spline_after_linear_fixed_test": True,
        "prior_results_on_same_fixed_test_population_already_observed": True,
    })
    print({"selected_alpha": alpha, "selection_lock": str(LOCK)}, flush=True)


def test() -> None:
    selection = json.loads(LOCK.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (selection["scope"] != "mgtab_eight_projection_additive_spline_exploratory_lock"
            or selection["screen_sha256"] != sha256_file(SCREEN)
            or selection["parent_selection_lock_sha256"] != sha256_file(PARENT)):
        raise ValueError("Additive spline validation lock differs")
    alpha = float(selection["selected_alpha"])
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABReducedProjectionRuntime(ROOT, seed)
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if source["split_sha256"] != selection["split_sha256_by_seed"][str(seed)]:
            raise ValueError("Official MGTAB split differs")
        contract_path = ROOT / ("output/v3_mgtab_ab_group_additive_tree/"
                                f"seed{seed}/ab_direction_contract.npz")
        with np.load(contract_path, allow_pickle=False) as contract:
            b = contract["b_indices"].astype(int)
        x_b = content_columns(runtime.evidence_matrix(b))
        teacher_b = runtime.predict_rows(b)
        transform = SplineTransformer(n_knots=3, degree=2,
                                      include_bias=False, knots="quantile").fit(x_b)
        zero = transform.transform(np.zeros((1, x_b.shape[1])))
        phi_b = transform.transform(x_b) - zero
        model = Ridge(alpha=alpha, fit_intercept=False).fit(
            phi_b, teacher_b["projection_raw_logit"])
        if model.coef_.shape != (48,) or model.intercept_ != 0.0:
            raise ValueError("Additive spline capacity differs")
        val = split["val"]
        named_val = runtime.predict_rows(val)["named_logit"]
        phi_val = transform.transform(content_columns(runtime.evidence_matrix(val))) - zero
        residual_val = model.predict(phi_val)
        probability_val = expit(named_val + np.clip(residual_val, -CAP, CAP))
        threshold = best_validation_threshold(probability_val, labels[val])
        val_metric = evaluate(labels[val], probability_val, threshold)
        expected = next(row for row in screen["runs"][seed]["candidates"]
                        if row["family"] == FAMILY and row["alpha"] == alpha)
        if (abs(threshold - selection["threshold_by_seed"][str(seed)]) > 1e-12
                or any(abs(val_metric[key] - expected["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Additive spline validation failed exact replay")
        test_idx = split["test"]
        named_test = runtime.predict_rows(test_idx)["named_logit"]
        phi_test = transform.transform(
            content_columns(runtime.evidence_matrix(test_idx))) - zero
        raw_residual = model.predict(phi_test)
        logit = named_test + np.clip(raw_residual, -CAP, CAP)
        probability = expit(logit)
        test_metric = evaluate(labels[test_idx], probability, threshold)
        run_dir = OUT / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        residual_path = run_dir / "additive_spline_residual.joblib"
        joblib.dump({"transform": transform, "zero_basis": zero,
                     "coefficients": model.coef_.astype(float),
                     "alpha": alpha, "cap": CAP,
                     "threshold": threshold}, residual_path)
        save_json(run_dir / "test_predictions.json", {
            "manifest": {"dataset": "mgtab", "seed": seed,
                         "selection_lock_sha256": sha256_file(LOCK),
                         "parent_model_sha256": sha256_file(ROOT /
                             f"output/v3_mgtab_reduced_projection_ab_group/seed{seed}/model.joblib"),
                         "residual_sha256": sha256_file(residual_path),
                         "split_sha256": source["split_sha256"],
                         "threshold": threshold},
            "predictions": [{"account_id": ids[int(i)],
                             "label": int(labels[i]),
                             "probability": float(probability[j]),
                             "logit": float(logit[j]),
                             "named_logit": float(named_test[j]),
                             "spline_raw_residual": float(raw_residual[j])}
                            for j, i in enumerate(test_idx)],
        })
        hyp_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        if (hyp["manifest"]["split_sha256"] != source["split_sha256"]
                or [row["account_id"] for row in hyp["predictions"]] !=
                [ids[int(i)] for i in test_idx]
                or [row["label"] for row in hyp["predictions"]] !=
                labels[test_idx].astype(int).tolist()):
            raise ValueError("HypBot paired account list differs")
        hyp_pred = np.asarray([row["probability"] >= hyp["manifest"]["threshold"]
                               for row in hyp["predictions"]], dtype=bool)
        hyp_acc = float(np.mean(hyp_pred == labels[test_idx]))
        result = {
            "seed": seed, "test": test_metric, "dev": val_metric,
            "hypbot_acc": hyp_acc,
            "residual_trainable_coefficients": 48,
            "residual_sha256": sha256_file(residual_path),
            "predictions_sha256": sha256_file(run_dir / "test_predictions.json"),
            "accounts_clipped_on_test": int(np.count_nonzero(np.abs(raw_residual) > CAP)),
            "above_hypbot": bool(test_metric["acc"] > hyp_acc),
            "above_user_91_2_percent": bool(test_metric["acc"] > .912),
        }
        runs.append(result)
        print(result, flush=True)
    save_json(OUT / "fixed_test_report.json", {
        "scope": "mgtab_eight_projection_additive_spline_exploratory_fixed_test",
        "selection_lock_sha256": sha256_file(LOCK),
        "all_seeds_above_hypbot_and_91_2_percent": all(
            row["above_hypbot"] and row["above_user_91_2_percent"] for row in runs),
        "decision_to_examine_spline_after_linear_fixed_test": True,
        "prior_results_on_same_fixed_test_population_already_observed": True,
        "runs": runs,
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
