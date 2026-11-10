"""Dev-only supervised 16-coefficient residual on a frozen MGTAB named-evidence branch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_eight_projection_low_capacity_residual_dev import content_columns
from verifiable_bot.mgtab_reduced_projection_runtime import MGTABReducedProjectionRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_supervised_bounded_linear_residual/dev_only.json"
PARENT = ROOT / "output/v3_mgtab_reduced_projection_ab_group/selection_lock.json"
CAP = 3.0
L2_GRID = (0.0, .0001, .001, .01, .1)


def bounded(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return CAP * np.tanh(x @ weights / CAP)


def fit(x: np.ndarray, named_logit: np.ndarray,
        labels: np.ndarray, l2: float) -> tuple[np.ndarray, dict]:
    y = np.asarray(labels, dtype=float)
    def loss(weights: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ weights / CAP
        residual = CAP * np.tanh(linear)
        z = named_logit + residual
        objective = float(np.mean(np.logaddexp(0.0, z) - y*z) +
                          .5*l2*np.dot(weights, weights))
        derivative = (expit(z) - y) * (1.0 - np.tanh(linear)**2)
        gradient = (x.T @ derivative) / len(y) + l2*weights
        return objective, gradient
    result = minimize(loss, np.zeros(x.shape[1]), method="L-BFGS-B",
                      jac=True, options={"maxiter": 300, "ftol": 1e-11})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"bounded residual optimizer failed: {result.message}")
    return result.x, {"iterations": int(result.nit),
                      "objective": float(result.fun),
                      "coefficient_l2_norm": float(np.linalg.norm(result.x)),
                      "coefficient_max_abs": float(np.max(np.abs(result.x)))}


def main() -> None:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if parent["selected_projection_dimensions"] != 8:
        raise ValueError("Eight-coordinate parent differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABReducedProjectionRuntime(ROOT, seed)
        _, labels, _, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if source["split_sha256"] != parent["split_sha256_by_seed"][str(seed)]:
            raise ValueError("Official split differs")
        contract_path = ROOT / ("output/v3_mgtab_ab_group_additive_tree/"
                                f"seed{seed}/ab_direction_contract.npz")
        with np.load(contract_path, allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        if len(np.intersect1d(a, b)) or not np.array_equal(
                np.sort(np.r_[a, b]), np.sort(split["train"])):
            raise ValueError("A/B direction and fit partition differs")
        train_x = content_columns(runtime.evidence_matrix(b))
        train_named = runtime.predict_rows(b)["named_logit"]
        val = split["val"]
        val_x = content_columns(runtime.evidence_matrix(val))
        val_named = runtime.predict_rows(val)["named_logit"]
        teacher = runtime.predict_rows(val)
        teacher_metric = evaluate(labels[val], teacher["probability"],
                                  runtime.threshold)
        choices = []
        for l2 in L2_GRID:
            weights, optimization = fit(train_x, train_named, labels[b], l2)
            val_logit = val_named + bounded(val_x, weights)
            probability = expit(val_logit)
            threshold = best_validation_threshold(probability, labels[val])
            choices.append({
                "l2": l2,
                "trainable_coefficients": len(weights),
                "residual_absolute_logit_bound": CAP,
                "validation": evaluate(labels[val], probability, threshold),
                "threshold": threshold,
                "optimization": optimization,
                "residual_max_abs_on_val": float(np.max(np.abs(
                    bounded(val_x, weights)))),
                "residual_changes_named_only_class_on_val": int(np.count_nonzero(
                    (probability >= threshold) !=
                    (expit(val_named) >= threshold))),
            })
        runs.append({"seed": seed,
                     "split_sha256": source["split_sha256"],
                     "reference_version": runtime.reference.version,
                     "teacher_validation": teacher_metric,
                     "candidates": choices})
        print({"seed": seed,
               "dev_f1_by_l2": {row["l2"]: round(
                   row["validation"]["macro_f1"], 5) for row in choices}},
              flush=True)
    parent_mean = float(np.mean([row["teacher_validation"]["macro_f1"]
                                 for row in runs]))
    passing = []
    for j, l2 in enumerate(L2_GRID):
        chosen = [run["candidates"][j] for run in runs]
        mean_f1 = float(np.mean([row["validation"]["macro_f1"]
                                 for row in chosen]))
        if (mean_f1 >= parent_mean - .005 and all(
                row["validation"]["acc"] >=
                run["teacher_validation"]["acc"] - .01
                for row, run in zip(chosen, runs, strict=True))):
            passing.append({"l2": l2, "mean_dev_macro_f1": mean_f1})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_supervised_bounded_sixteen_coefficient_residual_dev_only",
        "parent_selection_lock_sha256": sha256_file(PARENT),
        "test_predictions_or_metrics_computed": False,
        "test_labels_used_for_selection": False,
        "train_labels_used_only_in_B_reasoner_fit": True,
        "regularization_grid": L2_GRID,
        "residual_form": "three_times_tanh(linear_eight_upper_and_lower_tail_score_over_three)",
        "gate": "mean dev Macro-F1 within 0.5 point and each seed ACC within 1 point of eight-coordinate grouped tree",
        "gate_passing_candidates": passing,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print({"gate_passing_candidates": passing}, flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
