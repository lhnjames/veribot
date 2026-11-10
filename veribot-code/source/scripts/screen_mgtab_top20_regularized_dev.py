"""Dev-only regularization screen for the fixed 20 named-source evidence facts.

The fact set and label-blind reference are inherited from the existing frozen
source model. Official test labels and predictions are not read in this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_named_top20_lightgbm_dev import matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


# leaves, rounds, min_child, L2, column fraction, row fraction
CONFIGS = (
    (15, 600, 100, 30.0, 0.8, 0.8),
    (31, 600, 100, 30.0, 0.8, 0.8),
    (31, 800, 100, 50.0, 0.8, 0.8),
    (63, 600, 100, 50.0, 0.8, 0.8),
    (31, 400, 100, 20.0, 1.0, 1.0),
    (15, 800, 50, 30.0, 1.0, 0.8),
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_named_top20_lightgbm"
    lock_path = folder / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    if (lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(structural)
            or lock["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("Frozen source differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor,
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    parent = root / "output/v3_mgtab_named_evidence_lightgbm/selected"
    runs = []
    for seed in (0, 1, 2):
        report = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        if (report["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("Official split differs")
        selected = np.asarray(lock["selected_fact_indices_by_seed"][str(seed)],
                              dtype=int)
        if [names[int(j)] for j in selected] != lock["selected_fact_names_by_seed"][str(seed)]:
            raise ValueError("Frozen fact set differs")
        reference_path = parent / f"seed{seed}/reference.json.gz"
        reference = LabelBlindReference.load(reference_path)
        if list(reference.names) != names[:86]:
            raise ValueError("Named reference differs")
        train, dev = split["train"], split["val"]
        columns = np.r_[selected, selected + 86]
        fit_x = matrix(transform_with_support(
            reference, x[train, :86], split["supports"][train]))[:, columns]
        eval_x = matrix(transform_with_support(
            reference, x[dev, :86], split["supports"][dev]))[:, columns]
        alternatives = []
        for leaves, rounds, child, l2, columns_fraction, rows_fraction in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=rounds, num_leaves=leaves,
                min_child_samples=child, reg_lambda=l2,
                colsample_bytree=columns_fraction,
                subsample=rows_fraction, subsample_freq=1 if rows_fraction < 1 else 0,
                learning_rate=0.05, n_jobs=4, random_state=0,
                deterministic=True, force_col_wise=True, verbosity=-1)
            model.fit(fit_x, y[train])
            probability = model.predict_proba(eval_x)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            alternatives.append({
                "config": [leaves, rounds, child, l2,
                           columns_fraction, rows_fraction],
                "threshold": threshold,
                "validation": evaluate(y[dev], probability, threshold),
            })
        runs.append({
            "seed": seed,
            "split_sha256": lock["split_sha256_by_seed"][str(seed)],
            "reference_sha256": sha256_file(reference_path),
            "selected_fact_indices": selected.tolist(),
            "alternatives": alternatives,
        })
        print(json.dumps({"seed": seed, "macro_f1": [
            round(z["validation"]["macro_f1"] * 100, 2)
            for z in alternatives]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_fixed_20_named_evidence_regularization",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "prior_report_with_test_metrics_loaded_for_manifest": True,
        "prior_test_metrics_used_for_fit_or_selection": False,
        "new_test_predictions_or_metrics_computed": False,
        "selection_lock_sha256": sha256_file(lock_path),
        "source_level": "published_normalized_profile_tensor_and_typed_edges",
        "content_projection_used": False,
        "configs": CONFIGS,
        "runs": runs,
    }
    path = root / "output/v3_mgtab_top20_regularized/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
