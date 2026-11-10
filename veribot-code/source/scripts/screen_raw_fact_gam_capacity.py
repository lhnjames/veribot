"""Dev-only raw-fact additive capacity control for compact evidence models."""

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
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_group_tail_bottleneck import selected_facts


def additive_control(x: np.ndarray, y: np.ndarray, train: np.ndarray,
                     dev: np.ndarray) -> dict:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        SplineTransformer(n_knots=5, degree=1, knots="quantile",
                          include_bias=False),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                           random_state=0),
    )
    model.fit(x[train], y[train])
    probability = model.predict_proba(x[dev])[:, 1]
    threshold = best_validation_threshold(probability, y[dev])
    transformed = model[:-1].transform(x[dev])
    return {
        "validation": evaluate(y[dev], probability, threshold),
        "threshold": threshold,
        "raw_fact_count": x.shape[1],
        "additive_basis_dimension": transformed.shape[1],
        "nonzero_coefficients": int(np.count_nonzero(
            np.abs(model[-1].coef_) > 1e-8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("botsim", "fox8", "mgtab"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-facts", type=Path)
    parser.add_argument("--structural-facts", type=Path)
    parser.add_argument("--neighbor-facts", type=Path)
    parser.add_argument("--full-model-dir", type=Path, required=True)
    parser.add_argument("--compact-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for seed in (0, 1, 2):
        compact = json.loads((args.compact_model_dir / f"seed{seed}" / "report.json")
                             .read_text(encoding="utf-8"))
        if args.dataset == "mgtab":
            full = json.loads((args.full_model_dir / f"seed{seed}" / "report.json")
                              .read_text(encoding="utf-8"))
            config = SimpleNamespace(
                dataset="mgtab", data_dir=args.data_dir, account_file=None,
                native_graph_facts=None, structural_facts=args.structural_facts,
                neighbor_facts=args.neighbor_facts,
                selected_model_dir=args.full_model_dir)
            x, y, names, split = selected_facts(config, seed, full)
            selected = [int(i) for i in compact["manifest"]["selected_fact_indices"]]
            variants = {
                "same_20_source_facts": x[:, selected],
                "all_150_source_facts": x,
            }
        else:
            x, y, ids, names, split, source = load(
                args.dataset, args.data_dir, args.account_file, seed,
                args.native_facts if args.dataset == "botsim" else None)
            if args.dataset == "fox8":
                if args.native_facts is None:
                    raise ValueError("Fox8 native interaction facts required")
                with np.load(args.native_facts, allow_pickle=False) as cache:
                    if cache["ids"].astype(str).tolist() != ids:
                        raise ValueError("Fox8 native source account order differs")
                    x = np.column_stack((x, cache["values"].astype(float)))
                    names += cache["names"].astype(str).tolist()
            variants = {"all_source_facts": x}
        if (compact["manifest"]["dataset"] != args.dataset
                or compact["manifest"]["seed"] != seed):
            raise ValueError("compact population differs")
        if args.dataset != "mgtab" and (
                source["split_sha256"] != compact["manifest"]["split_sha256"]):
            raise ValueError("compact source split differs")
        if args.dataset == "mgtab" and (
                full["manifest"]["split_sha256"] != compact["manifest"]["split_sha256"]):
            raise ValueError("MGTAB source split differs")
        train, dev = split["train"], split["val"]
        results = {label: additive_control(matrix, y, train, dev)
                   for label, matrix in variants.items()}
        runs.append({
            "seed": seed,
            "split_sha256": compact["manifest"]["split_sha256"],
            "fixed_evidence_validation": compact["validation"],
            "raw_fact_controls": results,
        })
    result = {
        "dataset": args.dataset,
        "scope": "same_population_train_dev_only_raw_fact_additive_capacity",
        "test_predictions_accessed": False,
        "test_labels_used_for_fit_or_metrics": False,
        "model": "median_imputed_univariate_5_quantile_knot_linear_splines_plus_l2_logistic_C1",
        "data_files_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in (args.data_dir / ("features.pt" if args.dataset == "mgtab"
                                          else "features.csv"),
                         args.native_facts, args.structural_facts,
                         args.neighbor_facts) if path is not None},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "runs": [
        {"seed": row["seed"], "fixed_evidence": row["fixed_evidence_validation"],
         "controls": {key: value["validation"]
                      for key, value in row["raw_fact_controls"].items()}}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
