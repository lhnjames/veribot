"""Dev-only A/B-separated named-source evidence bottleneck with additive trees.

A selects 20 source facts and the direction of each evidence tail. B alone fits
the final monotone model. Every tree may use only the two tails of one fact.
The official test split is never transformed or scored here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIGS = ((4, 300, 30, 10.0), (4, 600, 30, 10.0),
           (8, 300, 30, 10.0), (8, 600, 30, 10.0))


def fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def evidence_matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def additive_tree_audit(model: LGBMClassifier, concepts: int) -> dict:
    trees = model.booster_.dump_model()["tree_info"]
    nonconstant, max_features = 0, 0
    for tree in trees:
        stack = [tree["tree_structure"]]
        used = set()
        while stack:
            node = stack.pop()
            if "split_feature" in node:
                used.add(int(node["split_feature"]))
                stack.extend((node["left_child"], node["right_child"]))
        if used:
            nonconstant += 1
            max_features = max(max_features, len(used))
        if len({index % concepts for index in used}) > 1:
            raise ValueError("A fitted tree combines different evidence facts")
    return {"tree_count": len(trees), "nonconstant_tree_count": nonconstant,
            "maximum_features_per_tree": max_features,
            "all_trees_single_fact": True}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    selected_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    named_dir = root / "output/v3_mgtab_named_evidence_lightgbm/selected"
    args = SimpleNamespace(dataset="mgtab", data_dir=data_dir,
                           account_file=None, native_graph_facts=None,
                           structural_facts=structural, neighbor_facts=neighbor,
                           selected_model_dir=selected_dir)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((selected_dir / f"seed{seed}/report.json").read_text())
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference_path = named_dir / f"seed{seed}/reference.json.gz"
        reference = LabelBlindReference.load(reference_path)
        if list(reference.names) != names[:86] or len(names) != 150:
            raise ValueError("Named evidence reference or source vocabulary differs")
        matrix_train = evidence_matrix(transform_with_support(
            reference, x[train, :86], split["supports"][train]))
        matrix_dev = evidence_matrix(transform_with_support(
            reference, x[dev, :86], split["supports"][dev]))
        if matrix_train.shape[1] != 172 or not np.isfinite(matrix_train).all():
            raise ValueError("Named evidence matrix differs")
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        if len(set(a) & set(b)) or len(a) + len(b) != len(train):
            raise ValueError("A/B training partition overlaps")
        selector = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1)
        selector.fit(matrix_train[a], y[train][a])
        gain = selector.booster_.feature_importance(importance_type="gain")
        fact_gain = gain[:86] + gain[86:]
        selected = np.sort(np.argsort(-fact_gain, kind="stable")[:20])
        if len(set(map(int, selected))) != 20 or np.any(fact_gain[selected] <= 0):
            raise ValueError("A-only selector yielded invalid fact set")
        columns = np.r_[selected, selected + 86]
        selected_a = matrix_train[a][:, columns]
        selected_b = matrix_train[b][:, columns]
        selected_dev = matrix_dev[:, columns]
        directions = []
        for j in range(40):
            positive_mean = selected_a[y[train][a] == 1, j].mean()
            negative_mean = selected_a[y[train][a] == 0, j].mean()
            delta = float(positive_mean - negative_mean)
            directions.append(1 if delta > 1e-12 else -1 if delta < -1e-12 else 0)
        interaction = [[j, j + 20] for j in range(20)]
        alternatives = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations,
                num_leaves=leaves, min_child_samples=min_leaf,
                reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                random_state=0, deterministic=True, force_col_wise=True,
                monotone_constraints=directions,
                interaction_constraints=interaction, verbosity=-1)
            model.fit(selected_b, y[train][b])
            tree_audit = additive_tree_audit(model, 20)
            probability = model.predict_proba(selected_dev)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            raw = np.asarray(model.booster_.predict(selected_dev[:64], raw_score=True))
            contrib = np.asarray(model.booster_.predict(selected_dev[:64], pred_contrib=True))
            max_error = float(np.max(np.abs(contrib.sum(axis=1) - raw)))
            if contrib.shape != (min(64, len(dev)), 41) or max_error > 1e-9:
                raise ValueError("Contributions do not reconstruct the evidence classifier")
            alternatives.append({
                "config": [leaves, iterations, min_leaf, l2],
                "threshold": threshold,
                "validation": evaluate(y[dev], probability, threshold),
                "additive_tree_audit": tree_audit,
                "max_probe_logit_additivity_error": max_error,
            })
        controls = {}
        for name, use_monotone, use_additive in (
                ("B_only_unconstrained", False, False),
                ("B_only_additive", False, True),
                ("B_only_monotone", True, False)):
            control = LGBMClassifier(
                objective="binary", n_estimators=600,
                num_leaves=8, min_child_samples=30,
                reg_lambda=10.0, learning_rate=0.05, n_jobs=4,
                random_state=0, deterministic=True, force_col_wise=True,
                monotone_constraints=directions if use_monotone else None,
                interaction_constraints=interaction if use_additive else None,
                verbosity=-1)
            control.fit(selected_b, y[train][b])
            if use_additive:
                additive_tree_audit(control, 20)
            probability = control.predict_proba(selected_dev)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            controls[name] = {
                "threshold": threshold,
                "validation": evaluate(y[dev], probability, threshold),
            }
        run = {
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_sha256": sha256_file(reference_path),
            "a_indices_sha256": fingerprint(train[a]),
            "b_indices_sha256": fingerprint(train[b]),
            "a_size": len(a), "b_size": len(b),
            "selector_fit_labels": "A_only", "reasoner_fit_labels": "B_only",
            "selected_fact_indices": selected.tolist(),
            "selected_fact_names": [names[int(j)] for j in selected],
            "selected_profile_count": int(np.count_nonzero(selected < 20)),
            "selected_native_relation_count": int(np.count_nonzero(selected >= 20)),
            "tail_directions": directions,
            "alternatives": alternatives,
            "B_only_capacity_controls": controls,
        }
        runs.append(run)
        print(json.dumps({"seed": seed,
                          "selected_profile_count": run["selected_profile_count"],
                          "dev_macro_f1": [round(z["validation"]["macro_f1"] * 100, 2)
                                           for z in alternatives]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_A_selects_B_fits_named_monotone_additive_20_fact_evidence",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "prior_report_with_test_metrics_loaded_for_manifest": True,
        "prior_test_metrics_used_for_fit_or_selection": False,
        "new_test_predictions_or_metrics_computed": False,
        "published_content_projection_used": False,
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "structural_facts_sha256": sha256_file(structural),
        "neighbor_facts_sha256": sha256_file(neighbor),
        "selector_config": [31, 400, 50, 10.0],
        "fit_configs": CONFIGS,
        "fact_count": 20,
        "evidence_columns_per_fact": 2,
        "interaction_rule": "Each tree's split features lie within one fact's upper/lower tail pair",
        "direction_rule": "Sign of A-only difference in bot versus human mean evidence tail",
        "runs": runs,
    }
    path = root / "output/v3_mgtab_ab_named_monotone_gam/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
