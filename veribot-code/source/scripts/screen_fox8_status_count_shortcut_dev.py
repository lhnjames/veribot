"""Train/dev diagnostic for single-field Fox8 status-count shortcuts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cache_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    audit_path = root / "research/certificates/fox8_neighbor_atomic_evidence_full_replay.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["all_neighbor_means_match_raw_source_at_float32_precision"] is not True
            or audit["native_neighbor_accounts"] != 2279):
        raise ValueError("Neighbor source is not independently audited")
    with np.load(cache_path, allow_pickle=False) as cache:
        neighbor_ids = cache["ids"].astype(str).tolist()
        neighbor_names = cache["names"].astype(str).tolist()
        neighbor_values = cache["values"].astype(float)
    neighbor_index = neighbor_names.index(
        "native_interaction_neighbor_mean_statuses_count")
    args = SimpleNamespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None, structural_facts=None,
        neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    import pandas as pd
    ids = pd.read_csv(args.data_dir / "features.csv")["user_id"].astype(str).tolist()
    if ids != neighbor_ids:
        raise ValueError("Neighbor source account order differs")
    runs = []
    for seed in (0, 1, 2):
        report = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        own_status = x[:, names.index("statuses_count")]
        neighbor_status = neighbor_values[:, neighbor_index]
        if np.any(own_status < 0) or np.any(neighbor_status[np.isfinite(neighbor_status)] < 0):
            raise ValueError("Status counts cannot be negative")
        values = {
            "own_statuses_count": own_status.reshape(-1, 1),
            "native_neighbor_mean_statuses_count":
                np.nan_to_num(neighbor_status, nan=-1).reshape(-1, 1),
            "both_status_counts": np.column_stack((
                own_status, np.nan_to_num(neighbor_status, nan=-1))),
            "has_native_neighbor_only": np.isfinite(neighbor_status).astype(float).reshape(-1, 1),
        }
        train, dev = split["train"], split["val"]
        outcomes = {}
        for variant, source in values.items():
            configs = []
            for depth in (1, 2, 3, 4):
                model = DecisionTreeClassifier(
                    max_depth=depth, min_samples_leaf=30, random_state=0)
                model.fit(source[train], y[train])
                probability = model.predict_proba(source[dev])[:, 1]
                configs.append({
                    "max_depth": depth,
                    "validation": evaluate(y[dev], probability, 0.5),
                    "actual_depth": model.get_depth(),
                    "split_features": [int(j) for j in model.tree_.feature
                                       if j >= 0],
                    "split_thresholds": [float(t) for f, t in
                                         zip(model.tree_.feature,
                                             model.tree_.threshold) if f >= 0],
                })
            outcomes[variant] = configs
        runs.append({"seed": seed,
                     "split_sha256": report["manifest"]["split_sha256"],
                     "dev_accounts": len(dev), "outcomes": outcomes})
        print(json.dumps({
            "seed": seed,
            "dev_errors": {key: [int(round((1 - item["validation"]["acc"]) * len(dev)))
                                for item in value]
                           for key, value in outcomes.items()}}), flush=True)
    output = {
        "scope": "fox8_official_train_dev_only_status_count_shortcut_diagnostic",
        "test_labels_or_predictions_used": False,
        "source_cache_sha256": sha256_file(cache_path),
        "independent_source_audit_sha256": sha256_file(audit_path),
        "runs": runs,
    }
    path = root / "output/v3_fox8_neighbor_atomic_evidence_capacity/status_shortcut_dev.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
