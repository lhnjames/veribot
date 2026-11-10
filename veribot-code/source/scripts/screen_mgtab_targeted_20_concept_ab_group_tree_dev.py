"""Official-dev screen of targeted 20-concept MGTAB A/B group trees."""

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
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_content
from screen_mgtab_ab_monotone_evidence_tree_dev import CONFIGS, directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


PROFILE_CHOICES = {
    "six_profile_two_relation_two_content": [3, 6, 7, 9, 13, 14],
    "four_profile_two_relation_four_content": [3, 6, 13, 14],
    "four_profile_two_relation_four_selected_content": [3, 6, 13, 14],
    "seven_profile_two_relation_one_content": [3, 6, 7, 9, 10, 13, 14],
}


def grouped(names: list[str], scheme: str, content_selected: list[int]) -> tuple[list[int], dict]:
    profile = PROFILE_CHOICES[scheme]
    relation_direct = [j for j in range(20, 86) if "_neighbor_" not in names[j]]
    relation_neighbor = [j for j in range(20, 86) if "_neighbor_" in names[j]]
    groups = {f"profile_{j:02d}": [names[j]] for j in profile}
    groups["native_relation_event_structure"] = [names[j] for j in relation_direct]
    groups["native_relation_neighbor_profile"] = [names[j] for j in relation_neighbor]
    if scheme == "four_profile_two_relation_four_selected_content":
        for j in content_selected:
            groups[f"published_content_coordinate_{j-86:03d}"] = [names[j]]
    else:
        blocks = 10 - len(profile) - 2
        for block, indices in enumerate(np.array_split(np.arange(86, 150), blocks)):
            groups[f"published_content_block_{block:02d}"] = [names[int(j)] for j in indices]
    chosen = [j for j in range(len(names))
              if names[j] in {name for members in groups.values() for name in members}]
    if len(groups) != 10 or len(chosen) != sum(len(v) for v in groups.values()):
        raise ValueError("Expected exactly ten nonoverlapping source groups")
    return chosen, groups


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    prior_path = root / "output/v3_mgtab_ab_group_additive_tree/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior["scope"] != "mgtab_official_train_dev_only_ab_monotone_additive_source_group_trees":
        raise ValueError("Full 150-fact A/B control differs")
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
        facts, labels, names, split = selected_facts(args, seed, old)
        if (len(names) != 150 or old["manifest"]["split_sha256"] !=
                prior["runs"][seed]["split_sha256"]):
            raise ValueError("MGTAB official split or source vocabulary differs")
        train, validation = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(facts[train], names)
        if reference.version != prior["runs"][seed]["reference_version"]:
            raise ValueError("Train-only label-blind reference differs")
        atomic_train = transform_with_support(reference, facts[train],
                                              split["supports"][train])
        atomic_dev = transform_with_support(reference, facts[validation],
                                            split["supports"][validation])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        selected_content = rank_content(atomic_train, labels[train], a, count=4)
        candidates = []
        for scheme in PROFILE_CHOICES:
            kept, groups = grouped(names, scheme, selected_content)
            train_concepts, _ = collapse_group_tails(columns(atomic_train, kept), groups)
            dev_concepts, _ = collapse_group_tails(columns(atomic_dev, kept), groups)
            if len(train_concepts.names) != 20 or tuple(train_concepts.names) != tuple(dev_concepts.names):
                raise ValueError("Twenty-concept bottleneck differs")
            x_train, x_dev = evidence_matrix(train_concepts), evidence_matrix(dev_concepts)
            signs = directions(x_train[a], labels[train][a])
            source_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
            variants = []
            for leaves, rounds, min_leaf, l2 in CONFIGS:
                model = LGBMClassifier(
                    objective="binary", n_estimators=rounds,
                    num_leaves=leaves, min_child_samples=min_leaf,
                    reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                    random_state=0, deterministic=True, force_col_wise=True,
                    monotone_constraints=signs.tolist(),
                    monotone_constraints_method="advanced",
                    interaction_constraints=source_cols, verbosity=-1)
                model.fit(x_train[b], labels[train][b])
                one_group, tree_counts = all_trees_one_group(model, source_cols)
                if not one_group:
                    raise ValueError("Tree crossed evidence source groups")
                p = model.predict_proba(x_dev)[:, 1]
                threshold = best_validation_threshold(p, labels[validation])
                variants.append({
                    "config": [leaves, rounds, min_leaf, l2],
                    "threshold": threshold,
                    "validation": evaluate(labels[validation], p, threshold),
                    "tree_count_by_group": tree_counts,
                })
            candidates.append({"scheme": scheme, "group_names": list(groups),
                               "retained_fact_names": [names[j] for j in kept],
                               "retained_fact_count": len(kept),
                               "concept_count": len(train_concepts.names),
                               "concept_version": train_concepts.reference_version,
                               "selected_content_on_A": ([names[j] for j in selected_content]
                                                        if "selected" in scheme else []),
                               "variants": variants})
        runs.append({"seed": seed,
                     "split_sha256": old["manifest"]["split_sha256"],
                     "reference_version": reference.version,
                     "direction_half_count": len(a), "fit_half_count": len(b),
                     "full_150_fact_group_tree_dev": prior["runs"][seed]["candidates"],
                     "candidates": candidates})
        print(json.dumps({"seed": seed,
                          "macro_f1": {r["scheme"]: [round(v["validation"]["macro_f1"], 4)
                                      for v in r["variants"]] for r in candidates}}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_targeted_twenty_concept_ab_group_tree",
        "test_predictions_or_metrics_computed": False,
        "reference_uses_train_facts_without_labels": True,
        "content_projection_source_level": "published_average_tweet_tensor_only",
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "full_150_fact_dev_sha256": sha256_file(prior_path),
        "configs": CONFIGS, "runs": runs,
    }
    path = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
