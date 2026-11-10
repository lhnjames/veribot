"""Dev-only 20 directly sourced MGTAB facts: 4 structural + 16 content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_atomic, rank_content
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--structural-facts", required=True, type=Path)
    parser.add_argument("--neighbor-facts", required=True, type=Path)
    parser.add_argument("--selected-model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    runs = []
    for seed in (0, 1, 2):
        directory = args.selected_model_dir / f"seed{seed}"
        original = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, original)
        reference = LabelBlindReference.load(directory / "reference.json.gz")
        if reference.version != original["manifest"]["reference_version"]:
            raise ValueError("reference version differs")
        train, dev = split["train"], split["val"]
        atomic_train = transform_with_support(reference, x[train],
                                              split["supports"][train])
        atomic_dev = transform_with_support(reference, x[dev],
                                            split["supports"][dev])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        structural = rank_atomic(atomic_train, y[train], a, list(range(86)), 4)
        content = rank_content(atomic_train, y[train], a, 16)
        selected = structural + content
        compact_train = columns(atomic_train, selected)
        compact_dev = columns(atomic_dev, selected)
        original_groups = original["reasoner"]["groups"]
        groups = {group: [name for name in members if name in compact_train.names]
                  for group, members in original_groups.items()}
        groups = {group: members for group, members in groups.items() if members}
        alternatives = {}
        for l2 in (1e-5, 1e-4, 1e-3):
            model = MonotoneSplineReasoner(groups, l2=l2)
            model.fit(rows(compact_train, a), y[train][a],
                      rows(compact_train, b), y[train][b])
            probability = model.predict_proba(compact_dev)
            threshold = best_validation_threshold(probability, y[dev])
            alternatives[str(l2)] = {
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
                "nonzero_slopes": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({
            "seed": seed, "split_sha256": original["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "structural_fact_names": [names[j] for j in structural],
            "content_fact_names": [names[j] for j in content],
            "concept_count": len(compact_train.names),
            "selected_using": "training_subset_A_labels_only",
            "full_validation": original["validation"],
            "alternatives": alternatives,
        })
    result = {"dataset": "mgtab", "scope": "validation_only_direct_20_numeric_fact_capacity",
              "test_predictions_accessed": False,
              "prior_test_results_already_observed": True,
              "source_limit": "published_profile_edge_and_mean_tweet_tensors",
              "semantic_content_concepts_verified": False,
              "selection_rule": "top 4 structural and top 16 content fact scores in A; B-only fit",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{
        "seed": row["seed"], "structural": row["structural_fact_names"],
        "alternatives": {key: val["validation"] for key, val in row["alternatives"].items()}
    } for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
