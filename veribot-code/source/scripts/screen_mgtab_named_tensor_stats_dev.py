"""Train/dev-only A/B screen of twenty source-verifiable MGTAB concepts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_ab_monotone_evidence_tree_dev import CONFIGS, directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.mgtab_mean_tweet_stats import STAT_NAMES, summaries
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    source_audit = json.loads((root / "research/certificates/mgtab_full_source_audit.json")
                              .read_text(encoding="utf-8"))
    source_hash = sha256_file(data_dir / "features.pt")
    if (source_audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or source_audit["source_sha256"]["features.pt"] != source_hash):
        raise ValueError("MGTAB published tensor differs from independent source audit")
    published = torch.load(data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy()
    stats = summaries(published)
    if stats.shape != (10199, 16):
        raise ValueError("published mean-tweet summary population differs")
    groups = {f"profile_{j:02d}": [f"published_profile_coordinate_{j:02d}"]
              for j in (3, 6, 13, 14)}
    runs = []
    for seed in (0, 1, 2):
        facts, labels, ids, names, split, source = load("mgtab", data_dir, None,
                                                        seed, None)
        facts, names, source = attach_mgtab_structural(
            facts, names, ids, source, data_dir, structural)
        facts, names, supports, source = attach_mgtab_neighbor(
            facts, names, ids, source, data_dir, neighbor)
        if (facts.shape != (10199, 86) or len(names) != 86
                or not np.array_equal(published[:, :20], facts[:, :20])):
            raise ValueError("named MGTAB fact population differs")
        matrix = np.column_stack((facts, stats))
        vocabulary = [*names, *STAT_NAMES]
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        groups["native_relation_event_structure"] = [name for name in names[20:]
                                                     if "_neighbor_" not in name]
        groups["native_relation_neighbor_profile"] = [name for name in names[20:]
                                                      if "_neighbor_" in name]
        for block in range(4):
            groups[f"published_tensor_stat_family_{block}"] = list(
                STAT_NAMES[4 * block:4 * block + 4])
        kept = [i for i, name in enumerate(vocabulary)
                if any(name in members for members in groups.values())]
        if len(groups) != 10 or len(kept) != 86:
            raise ValueError("ten source groups or selected facts differ")
        train_atomic = transform_with_support(reference, matrix[train], supports[train])
        dev_atomic = transform_with_support(reference, matrix[dev], supports[dev])
        train_concepts, _ = collapse_group_tails(columns(train_atomic, kept), groups)
        dev_concepts, _ = collapse_group_tails(columns(dev_atomic, kept), groups)
        x_train, x_dev = evidence_matrix(train_concepts), evidence_matrix(dev_concepts)
        if x_train.shape[1] != 40 or train_concepts.names != dev_concepts.names:
            raise ValueError("twenty-concept evidence vocabulary differs")
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        signs = directions(x_train[a], labels[train][a])
        source_columns = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
        variants = []
        for leaves, rounds, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=rounds, num_leaves=leaves,
                min_child_samples=min_leaf, reg_lambda=l2, learning_rate=0.05,
                n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
                monotone_constraints=signs.tolist(),
                monotone_constraints_method="advanced",
                interaction_constraints=source_columns, verbosity=-1)
            model.fit(x_train[b], labels[train][b])
            additive, tree_counts = all_trees_one_group(model, source_columns)
            if not additive:
                raise ValueError("source-group tree constraint differs")
            probability = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(probability, labels[dev])
            variants.append({"config": [leaves, rounds, min_leaf, l2],
                             "threshold": threshold,
                             "validation": evaluate(labels[dev], probability, threshold),
                             "tree_count_by_group": tree_counts})
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version,
                     "train_a": len(a), "train_b": len(b),
                     "source_fact_count": len(kept), "concept_count": 20,
                     "variants": variants})
        print(json.dumps({"seed": seed, "macro_f1": [
            v["validation"]["macro_f1"] for v in variants]}), flush=True)
    output = {"scope": "mgtab_official_train_dev_only_named_tensor_statistics",
              "test_predictions_or_metrics_computed": False,
              "source_level": "published_mean_tweet_tensor_statistics_not_post_semantics",
              "source_features_sha256": source_hash,
              "stat_names": STAT_NAMES,
              "groups": groups, "configs": CONFIGS, "runs": runs}
    path = root / "output/v3_mgtab_named_tensor_stats/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
