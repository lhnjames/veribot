"""Dev-only A/B monotone trees restricted to original additive source groups."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_ab_monotone_evidence_tree_dev import CONFIGS, directions, evidence_matrix
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def source_groups(names: list[str], model_groups: dict[str, list[str]]) -> list[list[int]]:
    index = {name: i for i, name in enumerate(names)}
    groups = []
    for members in model_groups.values():
        cols = [index[name] for name in members]
        groups.append(cols + [j + len(names) for j in cols])
    flattened = [j for group in groups for j in group]
    if sorted(flattened) != list(range(2 * len(names))):
        raise ValueError("source groups do not partition all evidence tails")
    return groups


def all_trees_one_group(model: LGBMClassifier,
                        groups: list[list[int]]) -> tuple[bool, dict[int, int]]:
    by_feature = {j: group for group, members in enumerate(groups)
                  for j in members}
    counts: dict[int, int] = {}
    for tree in model.booster_.dump_model()["tree_info"]:
        used = set()

        def walk(node: dict) -> None:
            if "split_feature" not in node:
                return
            used.add(by_feature[int(node["split_feature"])])
            walk(node["left_child"])
            walk(node["right_child"])

        walk(tree["tree_structure"])
        if len(used) > 1:
            return False, counts
        key = next(iter(used), -1)
        counts[key] = counts.get(key, 0) + 1
    return True, counts


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    prior_path = root / "output/v3_mgtab_ab_monotone_evidence_tree/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior["scope"] != "mgtab_official_train_dev_only_ab_disjoint_monotone_evidence_tree":
        raise ValueError("Prior unconstrained-group A/B tree screen differs")
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((full_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if (reference.version != old["manifest"]["reference_version"]
                or prior["runs"][seed]["split_sha256"] !=
                old["manifest"]["split_sha256"]):
            raise ValueError("MGTAB source or A/B split differs")
        evidence = transform_with_support(reference, x, split["supports"])
        design = evidence_matrix(evidence)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        constraints = directions(design[a], y[a])
        groups = source_groups(names, old["reasoner"]["groups"])
        candidates = []
        for config_index, (leaves, iterations, min_leaf, l2) in enumerate(CONFIGS):
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations,
                num_leaves=leaves, min_child_samples=min_leaf,
                reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                random_state=0, deterministic=True, force_col_wise=True,
                monotone_constraints=constraints.tolist(),
                monotone_constraints_method="advanced",
                interaction_constraints=groups, verbosity=-1)
            model.fit(design[b], y[b])
            single_group, tree_groups = all_trees_one_group(model, groups)
            if not single_group:
                raise ValueError("LightGBM tree crossed source groups")
            probability = model.predict_proba(design[dev])[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            raw = np.asarray(model.booster_.predict(design[dev][:32], raw_score=True))
            contribution = np.asarray(model.booster_.predict(
                design[dev][:32], pred_contrib=True))
            additivity = float(np.max(np.abs(contribution.sum(axis=1) - raw)))
            if contribution.shape != (32, 301) or additivity > 1e-9:
                raise ValueError("evidence group tree score does not replay")
            candidates.append({
                "config": [leaves, iterations, min_leaf, l2],
                "validation_at_half": evaluate(y[dev], probability, 0.5),
                "selected_threshold": threshold,
                "validation_selected": evaluate(y[dev], probability, threshold),
                "all_trees_use_one_source_group": single_group,
                "tree_count_by_group_index": tree_groups,
                "max_probe_logit_additivity_error": additivity,
                "prior_cross_group_tree_validation": prior["runs"][seed][
                    "candidates"][config_index]["validation_selected"],
            })
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "direction_half_count": len(a), "fit_half_count": len(b),
            "source_group_names": list(old["reasoner"]["groups"]),
            "source_group_count": len(groups),
            "original_ab_spline_validation": old["validation"],
            "candidates": candidates,
        })
        print(json.dumps({
            "seed": seed,
            "original_acc": old["validation"]["acc"],
            "additive_group_tree_acc": [round(c["validation_selected"]["acc"], 5)
                                        for c in candidates],
            "cross_group_tree_acc": [round(c["prior_cross_group_tree_validation"][
                "acc"], 5) for c in candidates]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_ab_monotone_additive_source_group_trees",
        "test_predictions_or_metrics_computed": False,
        "reference_uses_train_facts_without_labels": True,
        "monotone_direction_labels": "A only",
        "tree_fit_labels": "B only",
        "group_additivity_enforced_by_single_source_group_per_tree": True,
        "prior_cross_group_screen_sha256": sha256_file(prior_path),
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "lightgbm_version": lightgbm.__version__,
        "configs": CONFIGS,
        "runs": runs,
    }
    path = root / "output/v3_mgtab_ab_group_additive_tree/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
