"""Dev-only MGTAB published-fact capacity without content projections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_atomic
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fact-set", choices=("relation_20", "relation_all",
                                                "published_profile_and_relation_all"),
                        default="relation_20")
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    runs = []
    for seed in (0, 1, 2):
        directory = args.selected_model_dir / f"seed{seed}"
        original = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        x, labels, names, split = selected_facts(args, seed, original)
        reference = LabelBlindReference.load(directory / "reference.json.gz")
        if (reference.version != original["manifest"]["reference_version"]
                or len(names) != 150 or len(reference.names) != 150):
            raise ValueError("fixed source reference differs")
        train, dev = split["train"], split["val"]
        atomic_train = transform_with_support(reference, x[train], split["supports"][train])
        atomic_dev = transform_with_support(reference, x[dev], split["supports"][dev])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        named_relation_indices = [index for index, name in enumerate(names)
                                  if 20 <= index < 86 and name.startswith("published_")
                                  and "coordinate" not in name]
        if len(named_relation_indices) != 66:
            raise ValueError("named relation fact vocabulary differs")
        if args.fact_set == "relation_20":
            selected = rank_atomic(atomic_train, labels[train], a,
                                   named_relation_indices, 20)
        elif args.fact_set == "relation_all":
            selected = named_relation_indices
        else:
            selected = list(range(86))
        if any(index not in named_relation_indices for index in selected) and \
                args.fact_set != "published_profile_and_relation_all":
            raise ValueError("profile coordinate entered relation-only branch")
        evidence_train = columns(atomic_train, selected)
        evidence_dev = columns(atomic_dev, selected)
        groups = {name: [name] for name in evidence_train.names}
        alternatives = {}
        for l2 in (1e-5, 1e-4, 1e-3):
            model = MonotoneSplineReasoner(groups, l2=l2)
            model.fit(rows(evidence_train, a), labels[train][a],
                      rows(evidence_train, b), labels[train][b])
            probability = model.predict_proba(evidence_dev)
            threshold = best_validation_threshold(probability, labels[dev])
            alternatives[str(l2)] = {
                "validation": evaluate(labels[dev], probability, threshold),
                "threshold": threshold,
                "nonzero_slopes": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({
            "seed": seed, "split_sha256": original["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "selected_fact_indices": selected,
            "selected_fact_names": [names[index] for index in selected],
            "selected_from": args.fact_set,
            "selection_labels": ("train_A_only" if args.fact_set == "relation_20"
                                 else "none_all_prespecified_facts"),
            "concept_count": len(selected),
            "alternatives": alternatives,
        })
    result = {
        "dataset": "mgtab", "scope": "official_train_validation_only_published_fact_capacity",
        "test_predictions_accessed": False,
        "content_projections_used": False,
        "published_normalized_profile_coordinates_used":
            args.fact_set == "published_profile_and_relation_all",
        "source_level": ("published_normalized_profile_tensor_typed_edges_"
                         "and_neighbor_aggregates" if args.fact_set ==
                         "published_profile_and_relation_all" else
                         "published_typed_edges_and_neighbor_profile_aggregates"),
        "fact_set": args.fact_set,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{"seed": row["seed"],
                                "selected_fact_names": row["selected_fact_names"],
                                "alternatives": row["alternatives"]}
                               for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
