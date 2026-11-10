"""Paired, development-only MGTAB profile versus native-relation capacity test.

This tests whether source-audited graph facts add predictive information beyond
the 20 published normalized profile coordinates. It does not use the test split.
"""

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


def fit_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray,
                dev: np.ndarray) -> tuple[np.ndarray, float, dict]:
    model = HistGradientBoostingClassifier(
        max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
        l2_regularization=10.0, learning_rate=0.05, random_state=0)
    model.fit(x[train], y[train])
    probabilities = model.predict_proba(x[dev])[:, 1]
    threshold = best_validation_threshold(probabilities, y[dev])
    return probabilities, threshold, evaluate(y[dev], probabilities, threshold)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    selected_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    prior_path = root / "output/v3_mgtab_relation_only_capacity/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    audit_path = root / "research/certificates/mgtab_full_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="mgtab", data_dir=source_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=selected_dir)
    if (not source_dir.exists() or len(prior["runs"]) != 3
            or prior["test_predictions_accessed"] is not False):
        raise ValueError("published MGTAB data or dev-only paired reference missing")
    if (audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or audit["source_sha256"]["features.pt"] != sha256_file(
                source_dir / "features.pt")
            or audit["structural_facts_sha256"] != sha256_file(
                args.structural_facts)
            or audit["neighbor_facts_sha256"] != sha256_file(
                args.neighbor_facts)):
        raise ValueError("source-audited published facts or typed graph changed")

    runs = []
    for seed in (0, 1, 2):
        report = json.loads((selected_dir / f"seed{seed}/report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        previous = prior["runs"][seed]
        if (x.shape != (10199, 150) or len(names) != 150
                or previous["seed"] != seed
                or previous["split_sha256"] != report["manifest"]["split_sha256"]):
            raise ValueError(f"population, vocabulary, or split differs: "
                             f"shape={x.shape}, names={len(names)}, "
                             f"prior_seed={previous['seed']}, "
                             f"prior_split={previous['split_sha256']}, "
                             f"report_split={report['manifest']['split_sha256']}")
        train, dev = split["train"], split["val"]
        profile_prob, profile_threshold, profile_metric = fit_predict(
            x[:, :20], y, train, dev)
        joint_prob, joint_threshold, joint_metric = fit_predict(
            x[:, :86], y, train, dev)
        expected = previous["alternatives"][
            "published_profile_20_plus_graph_66"]
        if (joint_metric["errors"] != expected["validation"]["errors"]
                or abs(joint_metric["macro_f1"] - expected[
                    "validation"]["macro_f1"]) > 1e-12
                or abs(joint_threshold - expected["threshold"]) > 1e-12):
            raise ValueError("same-split graph model fails previous readback")
        profile_wrong = (profile_prob >= profile_threshold) != y[dev]
        joint_wrong = (joint_prob >= joint_threshold) != y[dev]
        runs.append({
            "seed": seed,
            "split_sha256": previous["split_sha256"],
            "dev_accounts": int(len(dev)),
            "published_profile_20": {
                "threshold_selected_on_dev": float(profile_threshold),
                "validation": profile_metric},
            "published_profile_20_plus_typed_graph_66": {
                "threshold_selected_on_dev": float(joint_threshold),
                "validation": joint_metric},
            "graph_minus_profile_macro_f1_points": float(100 * (
                joint_metric["macro_f1"] - profile_metric["macro_f1"])),
            "graph_minus_profile_correct_accounts": int(
                profile_wrong.sum() - joint_wrong.sum()),
            "paired_profile_wrong_graph_right": int(np.count_nonzero(
                profile_wrong & ~joint_wrong)),
            "paired_profile_right_graph_wrong": int(np.count_nonzero(
                ~profile_wrong & joint_wrong)),
        })
        print(json.dumps({"seed": seed, "profile_errors": profile_metric["errors"],
                          "profile_plus_graph_errors": joint_metric["errors"]}),
              flush=True)

    out = {
        "scope": "mgtab_official_train_dev_only_paired_raw_published_facts",
        "test_predictions_or_metrics_computed": False,
        "previous_paired_dev_report_sha256": sha256_file(prior_path),
        "independent_source_audit_sha256": sha256_file(audit_path),
        "published_features_sha256": sha256_file(source_dir / "features.pt"),
        "native_relation_fact_cache_sha256": {
            "structural": sha256_file(args.structural_facts),
            "neighbor": sha256_file(args.neighbor_facts)},
        "model_family": "sklearn_HistGradientBoostingClassifier_15_leaves_150_iters_30_min_leaf_l2_10",
        "threshold_rule": "same_dev_selected_MacroF1_threshold_for_each_model",
        "profile_source_level": "published_normalized_profile_tensor; original raw field values unavailable",
        "graph_source_level": "released_typed_edges_and_source_audited_derived_graph_facts",
        "runs": runs,
    }
    target = root / "research/certificates/mgtab_paired_profile_graph_dev.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
