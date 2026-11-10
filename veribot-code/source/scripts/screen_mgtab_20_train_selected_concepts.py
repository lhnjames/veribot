"""Dev-only upper-bound screen: A-selected 16 numeric content coordinates."""

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
from screen_mgtab_20_hybrid_concepts import concepts
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def rank_atomic(atomic, labels: np.ndarray, a: np.ndarray,
                candidates: list[int], count: int) -> list[int]:
    """Rank source facts by class-standardized tail separation using A alone."""
    if not 1 <= count <= len(candidates):
        raise ValueError("selected concept count must fit candidate set")
    scores = []
    for j in candidates:
        tails = []
        for matrix in (atomic.upper, atomic.lower):
            value = np.log1p(matrix[a, j]) * atomic.reliability[a, j]
            left, right = value[labels[a] == 0], value[labels[a] == 1]
            pooled = float(np.sqrt(np.var(left) + np.var(right) + 1e-12))
            tails.append(abs(float(right.mean() - left.mean())) / pooled)
        scores.append((max(tails), j))
    selected = [j for _, j in sorted(scores, key=lambda item: (-item[0], item[1]))[:count]]
    return sorted(selected)


def rank_content(atomic, labels: np.ndarray, a: np.ndarray,
                 count: int = 16) -> list[int]:
    """Select the requested count of content coordinates using labels from A."""
    return rank_atomic(atomic, labels, a, list(range(86, 150)), count)


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
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        selected = rank_content(atomic_train, y[train], a)
        compact_train = concepts(atomic_train, selected)
        compact_dev = concepts(transform_with_support(
            reference, x[dev], split["supports"][dev]), selected)
        groups = {"published_profile": list(compact_train.names[:2]),
                  "published_typed_relation": list(compact_train.names[2:4])}
        for name in compact_train.names[4:]:
            groups[name] = [name]
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
            "concept_version": compact_train.reference_version,
            "concept_count": len(compact_train.names),
            "selected_content_fact_names": [names[j] for j in selected],
            "selected_using": "training_subset_A_labels_only",
            "full_validation": original["validation"],
            "alternatives": alternatives,
        })
    result = {"dataset": "mgtab", "scope": "validation_only_supervised_numeric_capacity_upper_bound",
              "test_predictions_accessed": False,
              "prior_test_results_already_observed": True,
              "source_limit": "published_averaged_tweet_tensor",
              "semantic_concepts_verified": False,
              "selection_rule": "top 16 class-standardized tail-score differences in training A",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": "mgtab", "runs": [
        {"seed": row["seed"], "selected": row["selected_content_fact_names"],
         "alternatives": {key: value["validation"]
                          for key, value in row["alternatives"].items()}}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
