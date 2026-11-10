"""Evaluate the frozen raw-fact GAM specification on aligned test accounts.

The model family, feature sets, regularization and threshold rule come from the
earlier train/dev-only capacity screen. This script does not select a candidate
or a threshold from test results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_split_aligned_predictions import compare
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_group_tail_bottleneck import selected_facts
from verify_compact_inference_runtime import CONFIG


def model_specification():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        SplineTransformer(n_knots=5, degree=1, knots="quantile",
                          include_bias=False),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                           random_state=0),
    )


def source_variants(root: Path, dataset: str, config: dict, seed: int,
                    full_report: dict):
    data_dir = Path(config["data"])
    account = root / config["account"] if config.get("account") else None
    native = root / config["native"] if config.get("native") else None
    x0, y0, ids, _, split0, source = load(
        dataset, data_dir, account, seed,
        native if dataset == "botsim" else None)
    if dataset == "mgtab":
        args = SimpleNamespace(
            dataset=dataset, data_dir=data_dir, account_file=None,
            native_graph_facts=None,
            structural_facts=root / config["structural"],
            neighbor_facts=root / config["neighbor"],
            selected_model_dir=root / config["full"])
        x, y, names, split = selected_facts(args, seed, full_report)
        selected_dir = root / config["selected"]
        compact = json.loads((selected_dir / f"seed{seed}" / "report.json")
                             .read_text(encoding="utf-8"))
        selected = [int(j) for j in compact["manifest"]["selected_fact_indices"]]
        variants = {"same_20_source_facts": x[:, selected],
                    "all_150_source_facts": x}
    elif dataset == "fox8":
        with np.load(native, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("Fox8 native fact account order differs")
            x = np.column_stack((x0, cache["values"].astype(float)))
        y, split = y0, split0
        variants = {"all_source_facts": x}
    else:
        x, y, split = x0, y0, split0
        variants = {"all_source_facts": x}
    if (not np.array_equal(y, y0)
            or any(not np.array_equal(split[key], split0[key])
                   for key in ("train", "val", "test"))
            or source["split_sha256"] != full_report["manifest"]["split_sha256"]):
        raise ValueError("source population or split differs")
    return variants, y, ids, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    rows = []
    for dataset, config in CONFIG.items():
        dev_path = root / "output/v3_raw_fact_gam_capacity" / f"{dataset}_dev_only.json"
        prior = json.loads(dev_path.read_text(encoding="utf-8"))
        selected_dir = root / config["selected"]
        full_dir = root / config["full"]
        for seed in (0, 1, 2):
            full_report = json.loads((full_dir / f"seed{seed}" / "report.json")
                                     .read_text(encoding="utf-8"))
            fixed = json.loads((selected_dir / f"seed{seed}" / "test_predictions.json")
                               .read_text(encoding="utf-8"))
            variants, y, ids, split = source_variants(
                root, dataset, config, seed, full_report)
            if (fixed["manifest"]["split_sha256"] !=
                    prior["runs"][seed]["split_sha256"]):
                raise ValueError("compact/test split differs from dev-only control")
            for variant_name, x in variants.items():
                model = model_specification()
                model.fit(x[split["train"]], y[split["train"]])
                dev_probability = model.predict_proba(x[split["val"]])[:, 1]
                threshold = best_validation_threshold(
                    dev_probability, y[split["val"]])
                dev_metrics = evaluate(y[split["val"]], dev_probability, threshold)
                expected = prior["runs"][seed]["raw_fact_controls"][variant_name]
                if (abs(threshold - expected["threshold"]) > 1e-12
                        or any(abs(dev_metrics[k] - expected["validation"][k]) > 1e-12
                               for k in ("acc", "macro_f1", "auroc", "errors"))):
                    raise ValueError("frozen dev-only GAM control failed to replay")
                test_probability = model.predict_proba(x[split["test"]])[:, 1]
                out_dir = root / "output/v3_raw_fact_gam_fixed_test" / dataset / variant_name / f"seed{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                pred_path = out_dir / "test_predictions.json"
                pred_path.write_text(json.dumps({
                    "manifest": {
                        "dataset": dataset, "seed": seed,
                        "split_sha256": fixed["manifest"]["split_sha256"],
                        "threshold": threshold,
                        "threshold_selected_on": "validation_macro_f1",
                        "control_specification": "median_imputed_5_quantile_knot_linear_splines_l2_logistic_C1",
                        "source_dev_control_sha256": sha256_file(dev_path),
                        "raw_fact_count": int(x.shape[1]),
                    },
                    "predictions": [
                        {"account_id": ids[int(i)], "label": int(y[int(i)]),
                         "probability": float(p), "split": "test"}
                        for i, p in zip(split["test"], test_probability)],
                }, indent=2) + "\n", encoding="utf-8")
                comparison = compare(pred_path, selected_dir / f"seed{seed}" /
                                     "test_predictions.json")
                row = {
                    "dataset": dataset, "seed": seed, "variant": variant_name,
                    "raw_fact_count": int(x.shape[1]),
                    "raw_fact_dev": dev_metrics,
                    "raw_fact_test": evaluate(y[split["test"]], test_probability,
                                              threshold),
                    "fixed_evidence_test_acc": comparison["candidate_acc"],
                    "evidence_minus_raw_fact_acc": comparison["paired_acc_difference"],
                    "evidence_only_correct": comparison["candidate_correct_baseline_wrong"],
                    "raw_fact_only_correct": comparison["baseline_correct_candidate_wrong"],
                    "paired_account_bootstrap_95pct_ci": comparison[
                        "paired_account_bootstrap_95pct_ci"],
                    "predictions_file": str(pred_path),
                    "predictions_sha256": sha256_file(pred_path),
                    "dev_control_sha256": sha256_file(dev_path),
                }
                rows.append(row)
                print(dataset, seed, variant_name,
                      "raw", round(row["raw_fact_test"]["acc"], 4),
                      "evidence", round(row["fixed_evidence_test_acc"], 4),
                      flush=True)
    output = {
        "scope": "preexisting_dev_only_raw_fact_GAM_specification_fixed_test_comparison",
        "model_and_input_sets_selected_on_test": False,
        "threshold_selected_on": "validation_macro_f1",
        "same_test_accounts_and_split_verified": True,
        "prior_test_results_on_same_populations_already_observed": True,
        "model_capacity_differs_because_raw_GAM_uses_all_train_labels": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
