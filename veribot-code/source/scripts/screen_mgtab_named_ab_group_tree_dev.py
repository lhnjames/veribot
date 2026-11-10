"""Official-dev screen of A/B monotone group trees with named MGTAB facts only."""

from __future__ import annotations

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
from screen_mgtab_ab_monotone_evidence_tree_dev import (
    CONFIGS, directions, evidence_matrix)
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    named_dev_path = root / "output/v3_mgtab_named_evidence_lightgbm/dev_only.json"
    named_dev = json.loads(named_dev_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=source_dir)
    if (named_dev["source_features_sha256"] !=
            sha256_file(args.data_dir / "features.pt") or
            named_dev["structural_facts_sha256"] !=
            sha256_file(args.structural_facts) or
            named_dev["neighbor_facts_sha256"] !=
            sha256_file(args.neighbor_facts)):
        raise ValueError("named source population changed")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((source_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if (facts.shape != (10199, 150) or
                old["manifest"]["split_sha256"] !=
                named_dev["runs"][seed]["split_sha256"]):
            raise ValueError("official source or split differs")
        facts, names = facts[:, :86], names[:86]
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != named_dev["runs"][seed]["reference_version"]:
            raise ValueError("named train-only reference differs")
        dev_matrix = evidence_matrix(transform_with_support(
            reference, facts[dev], split["supports"][dev]))
        a, b = train_test_split(train, test_size=0.5,
                                stratify=labels[train], random_state=seed)
        a_matrix = evidence_matrix(transform_with_support(
            reference, facts[a], split["supports"][a]))
        b_matrix = evidence_matrix(transform_with_support(
            reference, facts[b], split["supports"][b]))
        constraint = directions(a_matrix, labels[a])
        if (len(constraint) != 172 or len(np.intersect1d(a, b)) or
                not np.array_equal(np.sort(np.r_[a, b]), np.sort(train))):
            raise ValueError("A/B named direction split differs")
        group_members = {
            group: members for group, members in old["reasoner"]["groups"].items()
            if group != "published_tweet_projections"}
        groups = source_groups(names, group_members)
        candidates = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations,
                num_leaves=leaves, min_child_samples=min_leaf,
                reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                random_state=0, deterministic=True, force_col_wise=True,
                monotone_constraints=constraint.tolist(),
                monotone_constraints_method="advanced",
                interaction_constraints=groups, verbosity=-1)
            model.fit(b_matrix, labels[b])
            single_group, group_counts = all_trees_one_group(model, groups)
            if not single_group:
                raise ValueError("named source tree crosses groups")
            probability = model.predict_proba(dev_matrix)[:, 1]
            threshold = best_validation_threshold(probability, labels[dev])
            candidates.append({
                "config": [leaves, iterations, min_leaf, l2],
                "selected_threshold": threshold,
                "validation": evaluate(labels[dev], probability, threshold),
                "tree_count_by_group": group_counts,
                "all_trees_one_group": single_group,
            })
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "direction_half_count": len(a), "fit_half_count": len(b),
            "named_source_fact_count": len(names),
            "anonymous_content_projection_used": False,
            "full_train_named_baseline_dev": named_dev["runs"][seed][
                "alternatives"][3]["validation"],
            "candidates": candidates,
        })
        print(json.dumps({
            "seed": seed,
            "baseline_dev_macro_f1": runs[-1]["full_train_named_baseline_dev"][
                "macro_f1"],
            "candidate_dev_macro_f1": [
                row["validation"]["macro_f1"] for row in candidates],
        }), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_named_ab_group_monotone_tree",
        "test_predictions_or_metrics_computed": False,
        "source_features_sha256": named_dev["source_features_sha256"],
        "named_dev_comparator_sha256": sha256_file(named_dev_path),
        "configs": CONFIGS, "runs": runs,
    }
    path = root / "output/v3_mgtab_named_ab_group_tree/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
