"""Dev-only 20-concept screen reserving one slot for native-neighbor volume."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_fox8_neighbor_atomic_evidence_capacity_dev import (
    CONFIGS, evidence_matrix, load_cache)
from screen_fox8_native_neighbor_profile_concepts_dev import (
    fit_and_score, weighted_atomic)
from screen_fox8_neighbor_20_concept_tree_dev import screen
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from gate0_shallow_tree import best_validation_threshold, evaluate
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


NEIGHBOR_FACT = "native_interaction_neighbor_mean_statuses_count"
REPLACED_GROUPS = ("presentation_flags", "presentation_text")


def screen_ab_group_tree(train_matrix: np.ndarray, dev_matrix: np.ndarray,
                         train_labels: np.ndarray, dev_labels: np.ndarray,
                         seed: int, group_names: list[str]) -> list[dict]:
    if train_matrix.shape[1] != 40 or len(group_names) != 10:
        raise ValueError("Fox8 A/B tree needs twenty two-tail concepts")
    a, b = train_test_split(np.arange(len(train_labels)), test_size=0.5,
                            stratify=train_labels, random_state=seed)
    signs = directions(train_matrix[a], train_labels[a])
    groups = [[2 * k, 2 * k + 1, 20 + 2 * k, 21 + 2 * k]
              for k in range(10)]
    results = []
    for leaves, rounds, min_leaf, l2 in CONFIGS:
        model = LGBMClassifier(
            objective="binary", n_estimators=rounds,
            num_leaves=leaves, min_child_samples=min_leaf,
            reg_lambda=l2, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups)
        model.fit(train_matrix[b], train_labels[b])
        single_group, tree_counts = all_trees_one_group(model, groups)
        if not single_group:
            raise ValueError("Fox8 tree crossed concept source groups")
        probability = model.predict_proba(dev_matrix)[:, 1]
        threshold = best_validation_threshold(probability, dev_labels)
        neighbor_index = group_names.index("native_neighbor_volume")
        masked = dev_matrix.copy()
        masked[:, groups[neighbor_index]] = 0.0
        masked_probability = model.predict_proba(masked)[:, 1]
        results.append({
            "config": [leaves, rounds, min_leaf, l2],
            "direction_half_count": len(a),
            "fit_half_count": len(b),
            "selected_threshold": threshold,
            "validation_selected": evaluate(dev_labels, probability, threshold),
            "all_trees_one_source_group": single_group,
            "tree_count_by_group": tree_counts,
            "native_neighbor_volume_zero_changed_dev_classes": int(
                np.count_nonzero((probability >= threshold) !=
                                 (masked_probability >= threshold))),
        })
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    audit_path = (root /
        "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["fact_artifact_sha256"] != sha256_file(raw_path)
            or audit["status"] != "sampled_raw_source_neighbor_means_verified"):
        raise ValueError("native neighbor fact source differs")
    ids, new_names, new_values, supports = load_cache(raw_path)
    if NEIGHBOR_FACT not in new_names:
        raise ValueError("native neighbor volume fact absent")
    args = SimpleNamespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None,
        structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    old_groups = json.loads((root / "research/fox8_20_concept_groups.json")
                            .read_text(encoding="utf-8"))
    previous_path = root / "output/v3_fox8_neighbor_20_concept_tree/dev_only.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    import pandas as pd
    if pd.read_csv(args.data_dir / "features.csv")[
            "user_id"].astype(str).tolist() != ids:
        raise ValueError("neighbor cache account order differs")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if (len(names) != 43 or len(new_names) != 31 or
                previous["runs"][seed]["split_sha256"] !=
                old["manifest"]["split_sha256"]):
            raise ValueError("official source or split differs")
        all_facts = np.column_stack((facts, new_values))
        all_names = names + new_names
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            all_facts[train], all_names)
        atomic_train = weighted_atomic(reference, all_facts[train], supports[train])
        atomic_dev = weighted_atomic(reference, all_facts[dev], supports[dev])
        candidates = []
        for removed_group in REPLACED_GROUPS:
            removed_facts = set(old_groups[removed_group])
            kept = [j for j, name in enumerate(all_names)
                    if (j < 43 and name not in removed_facts)
                    or name == NEIGHBOR_FACT]
            groups = {group: members for group, members in old_groups.items()
                      if group != removed_group}
            groups["native_neighbor_volume"] = [NEIGHBOR_FACT]
            train_atomic = columns(atomic_train, kept)
            dev_atomic = columns(atomic_dev, kept)
            concept_train, _ = collapse_group_tails(train_atomic, groups)
            concept_dev, _ = collapse_group_tails(dev_atomic, groups)
            if (len(concept_train.names) != 20 or
                    tuple(concept_train.names) != tuple(concept_dev.names)):
                raise ValueError("targeted bottleneck is not twenty concepts")
            spline, _ = fit_and_score(
                train_atomic, dev_atomic, labels[train], labels[dev], seed, groups)
            trees = screen(evidence_matrix(concept_train),
                           evidence_matrix(concept_dev), labels[train],
                           labels[dev], list(groups))
            ab_trees = screen_ab_group_tree(
                evidence_matrix(concept_train),
                evidence_matrix(concept_dev), labels[train],
                labels[dev], seed, list(groups))
            candidates.append({
                "removed_group": removed_group,
                "added_group": "native_neighbor_volume",
                "added_fact": NEIGHBOR_FACT,
                "retained_atomic_fact_count": len(kept),
                "concept_count": len(concept_train.names),
                "ab_spline": spline,
                "full_train_trees": trees,
                "ab_group_trees": ab_trees,
            })
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "previous_control_tree_errors": [
                row["validation_selected"]["errors"]
                for row in previous["runs"][seed]["control_tree"]],
            "atomic_74_dev_errors": previous["runs"][seed][
                "atomic_74_best_dev"]["errors"],
            "candidates": candidates,
        })
        print(json.dumps({
            "seed": seed,
            "candidate_errors": [
                {"removed": row["removed_group"],
                 "spline": row["ab_spline"]["validation"]["errors"],
                 "tree": [item["validation_selected"]["errors"]
                          for item in row["full_train_trees"]],
                 "ab_group_tree": [item["validation_selected"]["errors"]
                                   for item in row["ab_group_trees"]]}
                for row in candidates],
        }), flush=True)
    output = {
        "scope": "fox8_official_train_dev_only_targeted_native_neighbor_twenty_concepts",
        "test_predictions_or_metrics_computed": False,
        "neighbor_fact_source_sha256": sha256_file(raw_path),
        "neighbor_source_audit_sha256": sha256_file(audit_path),
        "previous_twenty_concept_dev_sha256": sha256_file(previous_path),
        "runs": runs,
    }
    path = root / "output/v3_fox8_targeted_neighbor_20_concepts/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
