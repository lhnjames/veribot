"""Dev-only capacity screen on 86 named MGTAB facts after label-blind evidence mapping."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIGS = (
    (15, 150, 30, 10.0),
    (15, 300, 30, 10.0),
    (31, 150, 30, 10.0),
    (31, 300, 50, 10.0),
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    prior_path = root / "output/v3_mgtab_named_raw_capacity/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor, selected_model_dir=full_dir)
    if (prior["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or prior["structural_facts_sha256"] != sha256_file(structural)
            or prior["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("published source or verified facts differ")
    runs = []
    for seed in (0, 1, 2):
        report = json.loads((full_dir / f"seed{seed}" / "report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        prior_run = next(row for row in prior["runs"] if row["seed"] == seed)
        if (names[:86] != prior["fact_names"]
                or prior_run["split_sha256"] != report["manifest"]["split_sha256"]
                or x.shape != (10199, 150)):
            raise ValueError("same named facts and official split are required")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            x[train, :86], names[:86])
        train_evidence = transform_with_support(
            reference, x[train, :86], split["supports"][train])
        dev_evidence = transform_with_support(
            reference, x[dev, :86], split["supports"][dev])
        def matrix(batch):
            return np.column_stack((
                np.log1p(batch.upper) * batch.reliability,
                np.log1p(batch.lower) * batch.reliability))
        x_train, x_dev = matrix(train_evidence), matrix(dev_evidence)
        if (x_train.shape[1] != 172 or not np.isfinite(x_train).all()
                or not np.isfinite(x_dev).all()):
            raise ValueError("named evidence matrix differs")
        alternatives = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_leaf_nodes=leaves, max_iter=iterations,
                min_samples_leaf=min_leaf, l2_regularization=l2,
                learning_rate=0.05, random_state=0)
            model.fit(x_train, y[train])
            probability = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            alternatives.append({
                "max_leaf_nodes": leaves, "max_iter": iterations,
                "min_samples_leaf": min_leaf, "l2_regularization": l2,
                "threshold": threshold,
                "validation": evaluate(y[dev], probability, threshold),
            })
        runs.append({
            "seed": seed, "split_sha256": prior_run["split_sha256"],
            "reference_version": reference.version,
            "named_fact_count": 86, "evidence_input_count": 172,
            "named_raw_tree_control": prior_run["capacity_controls"][
                "shallow_hist_gradient_boosting"]["validation"],
            "alternatives": alternatives,
        })
        print(seed, [(round(row["validation"]["acc"] * 100, 2),
                      round(row["validation"]["macro_f1"] * 100, 2))
                     for row in alternatives], flush=True)
    result = {
        "scope": "mgtab_train_dev_only_named_source_evidence_boosting",
        "test_rows_accessed": False,
        "published_content_projection_used": False,
        "source_features_sha256": prior["source_features_sha256"],
        "structural_facts_sha256": prior["structural_facts_sha256"],
        "neighbor_facts_sha256": prior["neighbor_facts_sha256"],
        "prior_named_raw_capacity_sha256": sha256_file(prior_path),
        "configs": CONFIGS, "runs": runs,
    }
    path = root / "output/v3_mgtab_named_evidence_boosting/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
