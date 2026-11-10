"""Dev-only test of linear or additive low-capacity replacements for MGTAB content trees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from verifiable_bot.mgtab_reduced_projection_runtime import MGTABReducedProjectionRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_eight_projection_low_capacity_residual/dev_only.json"
PARENT = ROOT / "output/v3_mgtab_reduced_projection_ab_group/selection_lock.json"
ALPHAS = (0.1, 1.0, 10.0)
CAP = 3.0


def content_columns(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x[:, 86:94], x[:, 180:188]))


def candidate(x_b: np.ndarray, target_b: np.ndarray,
              x_val: np.ndarray, named_val: np.ndarray,
              target_val: np.ndarray, labels_val: np.ndarray, *, family: str,
              alpha: float) -> dict:
    if family == "linear":
        train_design, val_design = x_b, x_val
        parameter_count = x_b.shape[1]
    elif family == "additive_quadratic_splines_three_knots":
        transform = SplineTransformer(n_knots=3, degree=2,
                                      include_bias=False, knots="quantile")
        transform.fit(x_b)
        zero = transform.transform(np.zeros((1, x_b.shape[1])))
        train_design = transform.transform(x_b) - zero
        val_design = transform.transform(x_val) - zero
        parameter_count = train_design.shape[1]
    else:
        raise ValueError("unsupported low-capacity residual family")
    model = Ridge(alpha=alpha, fit_intercept=False).fit(train_design, target_b)
    residual = model.predict(val_design)
    probability = expit(named_val + np.clip(residual, -CAP, CAP))
    threshold = best_validation_threshold(probability, labels_val)
    return {
        "family": family,
        "alpha": alpha,
        "input_coordinates": x_b.shape[1],
        "trainable_coefficients": parameter_count,
        "content_residual_absolute_logit_cap": CAP,
        "validation": evaluate(labels_val, probability, threshold),
        "validation_threshold": threshold,
        "validation_residual_max_abs_before_cap": float(np.max(np.abs(residual))),
        "validation_residual_mse_to_teacher": float(np.mean((
            residual - target_val)**2)),
    }


def main() -> None:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if parent["selected_projection_dimensions"] != 8:
        raise ValueError("Selected eight-coordinate parent differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABReducedProjectionRuntime(ROOT, seed)
        _, labels, _, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if source["split_sha256"] != parent["split_sha256_by_seed"][str(seed)]:
            raise ValueError("Official split differs")
        contract_path = ROOT / ("output/v3_mgtab_ab_group_additive_tree/"
                                f"seed{seed}/ab_direction_contract.npz")
        with np.load(contract_path, allow_pickle=False) as contract:
            b = contract["b_indices"].astype(int)
        x_b = runtime.evidence_matrix(b)
        teacher_b = runtime.predict_rows(b)
        val = split["val"]
        x_val = runtime.evidence_matrix(val)
        teacher_val = runtime.predict_rows(val)
        content_b, content_val = content_columns(x_b), content_columns(x_val)
        rows = [candidate(
            content_b, teacher_b["projection_raw_logit"],
            content_val, teacher_val["named_logit"],
            teacher_val["projection_raw_logit"], labels[val],
            family=family, alpha=alpha)
            for family in ("linear", "additive_quadratic_splines_three_knots")
            for alpha in ALPHAS]
        teacher_metric = evaluate(labels[val], teacher_val["probability"],
                                  runtime.threshold)
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "teacher_validation": teacher_metric,
                     "candidates": rows})
        print({"seed": seed,
               "teacher_dev_f1": round(teacher_metric["macro_f1"], 5),
               "candidate_dev_f1": [(row["family"], row["alpha"],
                                     round(row["validation"]["macro_f1"], 5))
                                    for row in rows]}, flush=True)
    passing = []
    teacher_mean = float(np.mean([run["teacher_validation"]["macro_f1"]
                                  for run in runs]))
    for j in range(len(runs[0]["candidates"])):
        rows = [run["candidates"][j] for run in runs]
        mean = float(np.mean([row["validation"]["macro_f1"] for row in rows]))
        if (mean >= teacher_mean - .005 and all(
                row["validation"]["acc"] >=
                run["teacher_validation"]["acc"] - .01
                for row, run in zip(rows, runs, strict=True))):
            passing.append({"family": rows[0]["family"],
                            "alpha": rows[0]["alpha"],
                            "trainable_coefficients": rows[0]["trainable_coefficients"],
                            "mean_dev_macro_f1": mean})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_eight_projection_low_capacity_residual_dev_only",
        "parent_selection_lock_sha256": sha256_file(PARENT),
        "test_predictions_or_metrics_computed": False,
        "test_labels_used_for_selection": False,
        "families": ["linear", "additive_quadratic_splines_three_knots"],
        "ridge_alphas": ALPHAS,
        "gate": "mean dev Macro-F1 within 0.5 point and each seed ACC within 1 point of eight-coordinate tree teacher",
        "gate_passing_candidates": passing,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print({"gate_passing_candidates": passing})
    print(OUT)


if __name__ == "__main__":
    main()
