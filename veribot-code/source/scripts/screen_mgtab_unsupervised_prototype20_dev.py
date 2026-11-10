"""Dev-only 20-fact MGTAB model with label-blind content prototypes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_twenty_source_facts_dev import rank_facts
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_neutral_anchor_runtime import MGTABNeutralAnchorRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_unsupervised_prototype20/dev_only.json"
PROFILE = 4
RELATION = 8
PROTOTYPES = 8
CONTENT_CAP = 3.0


def content_prototypes(runtime, train: np.ndarray) -> tuple[np.ndarray, dict]:
    # Published average-tweet tensor -> train-only PCA already frozen in runtime.
    # Clustering and empirical reference distribution use no account labels.
    content = np.asarray(runtime.facts[:, 86:150], dtype=np.float64)
    if content.shape != (10199, 64) or not np.isfinite(content).all():
        raise ValueError("Published content projection population differs")
    scaler = StandardScaler().fit(content[train])
    scaled = np.clip(scaler.transform(content), -8.0, 8.0)
    km = MiniBatchKMeans(
        n_clusters=PROTOTYPES, batch_size=1024, n_init=5,
        max_iter=100, random_state=0)
    km.fit(scaled[train])
    distance = np.sum((scaled[:, None, :] -
                       km.cluster_centers_[None, :, :]) ** 2, axis=2)
    distance_scale = float(np.median(np.min(distance[train], axis=1)))
    if not np.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("Prototype distance scale is undefined")
    similarity = 1.0 / (1.0 + distance / distance_scale)
    upper = np.empty_like(similarity)
    lower = np.empty_like(similarity)
    for j in range(PROTOTYPES):
        sorted_train = np.sort(similarity[train, j])
        fraction = np.searchsorted(sorted_train, similarity[:, j],
                                   side="right") / (len(train) + 1)
        upper[:, j] = np.maximum(2 * fraction - 1, 0)
        lower[:, j] = np.maximum(1 - 2 * fraction, 0)
    encoded = np.column_stack((np.log1p(upper), np.log1p(lower)))
    return encoded, {
        "prototype_count": PROTOTYPES,
        "prototype_fit_rows": len(train),
        "prototype_fit_label_access": False,
        "projection_source": "released_average_tweet_tensor_train_only_PCA64",
        "distance_scale": distance_scale,
        "kmeans_inertia": float(km.inertia_),
        "content_encoding": "eight_train_only_unsupervised_centroid_similarities_with_empirical_upper_lower_tails",
        "prototype_centers": km.cluster_centers_.tolist(),
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
    }


def main() -> None:
    if PROFILE + RELATION + PROTOTYPES != 20:
        raise ValueError("Twenty-fact budget differs")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABNeutralAnchorRuntime(ROOT, seed)
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or
                source["split_sha256"] != runtime.lock[
                    "split_sha256_by_seed"][str(seed)]):
            raise ValueError("Official MGTAB population differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        train, dev = split["train"], split["val"]
        if (len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(train))):
            raise ValueError("A/B training halves differ")
        prototype, metadata = content_prototypes(runtime, train)
        x_a, x_b, x_dev = (evidence_design(runtime, rows)
                           for rows in (a, b, dev))
        scores = rank_facts(x_a, labels[a])
        profile = sorted(range(20), key=lambda j: (-scores[j], j))[:PROFILE]
        relation = sorted(range(20, 86),
                          key=lambda j: (-scores[j], j))[:RELATION]
        selected = np.asarray(sorted(profile + relation), dtype=int)
        d = len(selected) + PROTOTYPES
        def design(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
            return np.column_stack((
                x[:, selected], prototype[rows, :PROTOTYPES],
                x[:, selected + 150], prototype[rows, PROTOTYPES:]))
        a_design = design(x_a, a)
        b_design = design(x_b, b)
        dev_design = design(x_dev, dev)
        signs = directions(a_design, labels[a]).astype(int)
        groups = []
        names = []
        for name, members in runtime.group_fact_indices.items():
            local = [j for j, original in enumerate(selected)
                     if original in members]
            if local:
                groups.append(local + [j + d for j in local])
                names.append(name)
        content_columns = list(range(len(selected), d)) + list(
            range(len(selected) + d, 2 * d))
        groups.append(content_columns)
        names.append("published_tweet_unsupervised_prototype_similarity")
        if sorted(j for members in groups for j in members) != list(range(2*d)):
            raise ValueError("Prototype source groups do not partition inputs")
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups)
        model.fit(b_design, labels[b])
        one_group, tree_counts = all_trees_one_group(model, groups)
        if not one_group:
            raise ValueError("Prototype model crossed source groups")
        full_logit = np.asarray(model.booster_.predict(
            dev_design, raw_score=True), dtype=float)
        named = dev_design.copy()
        named[:, content_columns] = 0.0
        named_logit = np.asarray(model.booster_.predict(
            named, raw_score=True), dtype=float)
        bounded = expit(named_logit + np.clip(
            full_logit - named_logit, -CONTENT_CAP, CONTENT_CAP))
        threshold = best_validation_threshold(bounded, labels[dev])
        metrics = evaluate(labels[dev], bounded, threshold)
        frozen = evaluate(labels[dev], runtime.model.predict_proba(
            x_dev)[:, 1], runtime.threshold)
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "selected_named_indices": selected.tolist(),
            "selected_named_facts": [runtime.names[j] for j in selected],
            "content_prototype_metadata": metadata,
            "source_groups": names,
            "tree_count_by_group": tree_counts,
            "all_trees_one_source_group": one_group,
            "monotone_signs": signs.tolist(),
            "selected_threshold": threshold,
            "content_logit_cap": CONTENT_CAP,
            "validation": metrics,
            "frozen_150_fact_validation": frozen,
            "macro_f1_loss_points": 100 * (
                frozen["macro_f1"] - metrics["macro_f1"]),
            "acc_loss_points": 100 * (
                frozen["acc"] - metrics["acc"]),
        })
        print({"seed": seed, "compact_dev_acc": metrics["acc"],
               "compact_dev_macro_f1": metrics["macro_f1"],
               "full_dev_macro_f1": frozen["macro_f1"]}, flush=True)
    mean_f1_loss = float(np.mean([r["macro_f1_loss_points"] for r in runs]))
    passed = bool(mean_f1_loss <= .5 and all(
        r["acc_loss_points"] <= 1.0 for r in runs))
    OUT.write_text(json.dumps({
        "scope": "MGTAB_twenty_fact_unsupervised_content_prototypes_dev_only",
        "test_predictions_or_metrics_accessed": False,
        "development_selection_rule": "mean_Macro_F1_loss_at_most_0.5_points_and_each_ACC_loss_at_most_1_point",
        "prototype_fit_labels_used": False,
        "named_fact_selection_and_directions": "A_half_labels_only",
        "tree_fit_labels": "B_half_only",
        "fact_budget": {"named_profile": PROFILE, "named_relation": RELATION,
                        "content_prototypes": PROTOTYPES},
        "bounded_content_logit_cap": CONTENT_CAP,
        "mean_macro_f1_loss_points": mean_f1_loss,
        "passes_dev_gate": passed,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print({"passes_dev_gate": passed,
           "mean_macro_f1_loss_points": mean_f1_loss,
           "output": str(OUT)})


if __name__ == "__main__":
    main()
