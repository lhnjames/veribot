"""Dev-only capacity check of MGTAB's 86 named published source facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    selected_dir = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    config = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        full = json.loads((full_dir / f"seed{seed}" / "report.json")
                          .read_text(encoding="utf-8"))
        selected = json.loads((selected_dir / f"seed{seed}" / "report.json")
                              .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(config, seed, full)
        if (len(names) != 150 or selected["manifest"]["split_sha256"] !=
                full["manifest"]["split_sha256"]):
            raise ValueError("MGTAB fixed source population differs")
        named = x[:, :86]
        train, dev = split["train"], split["val"]
        shuffled = x.copy()
        rng = np.random.default_rng(20261103 + seed)
        shuffled[train, 86:] = x[rng.permutation(train), 86:]
        shuffled[dev, 86:] = x[rng.permutation(dev), 86:]
        alternatives = {
            "five_quantile_linear_spline_l2_gam": make_pipeline(
                SimpleImputer(strategy="median"),
                SplineTransformer(n_knots=5, degree=1, knots="quantile",
                                  include_bias=False),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                   random_state=0)),
            "shallow_hist_gradient_boosting": HistGradientBoostingClassifier(
                max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
                l2_regularization=10.0, learning_rate=0.05,
                random_state=0),
            "shallow_hist_gradient_boosting_all_150": HistGradientBoostingClassifier(
                max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
                l2_regularization=10.0, learning_rate=0.05,
                random_state=0),
            "shallow_hist_gradient_boosting_shuffled_64_content": HistGradientBoostingClassifier(
                max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
                l2_regularization=10.0, learning_rate=0.05,
                random_state=0),
        }
        results = {}
        for name, model in alternatives.items():
            matrix = (x if name.endswith("all_150") else
                      shuffled if name.endswith("shuffled_64_content") else named)
            model.fit(matrix[train], y[train])
            probability = model.predict_proba(matrix[dev])[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            results[name] = {
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
            }
        row = {"seed": seed, "split_sha256": full["manifest"]["split_sha256"],
               "named_fact_count": named.shape[1],
               "fixed_20_evidence_validation": selected["validation"],
               "full_150_evidence_validation": full["validation"],
               "capacity_controls": results}
        runs.append(row)
        print(seed, {key: value["validation"]["acc"]
                     for key, value in results.items()}, flush=True)
    output = {
        "dataset": "mgtab", "scope": "same_train_dev_named_published_fact_capacity",
        "test_predictions_accessed": False,
        "controls_use_all_train_labels": True,
        "shuffled_content_control": "preserves 64 content-column marginals within train and dev while independently permuting account rows without labels",
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "structural_facts_sha256": sha256_file(config.structural_facts),
        "neighbor_facts_sha256": sha256_file(config.neighbor_facts),
        "fact_names": names[:86], "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
