"""Dev-only source-concept capacity curve; no test predictions are read."""

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
from screen_mgtab_20_train_selected_concepts import rank_content
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
    budgets = (8, 16, 24, 32, 48, 64)
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
        budget_rows = []
        for budget in budgets:
            selected = rank_content(atomic_train, y[train], a, count=budget)
            compact_train = concepts(atomic_train, selected)
            compact_dev = concepts(atomic_dev, selected)
            groups = {"published_profile": list(compact_train.names[:2]),
                      "published_typed_relation": list(compact_train.names[2:4])}
            for name in compact_train.names[4:]:
                groups[name] = [name]
            model = MonotoneSplineReasoner(groups, l2=1e-5)
            model.fit(rows(compact_train, a), y[train][a],
                      rows(compact_train, b), y[train][b])
            probability = model.predict_proba(compact_dev)
            threshold = best_validation_threshold(probability, y[dev])
            budget_rows.append({
                "content_coordinates": budget,
                "total_concepts": budget + 4,
                "selected_content_facts": [names[j] for j in selected],
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
            })
        runs.append({"seed": seed,
                     "split_sha256": original["manifest"]["split_sha256"],
                     "reference_version": reference.version,
                     "full_150_fact_validation": original["validation"],
                     "budgets": budget_rows})
    result = {"dataset": "mgtab", "scope": "validation_only_numeric_concept_budget_curve",
              "test_predictions_accessed": False,
              "prior_test_results_already_observed": True,
              "source_limit": "published_averaged_tweet_tensor",
              "semantic_concepts_verified": False,
              "selection_rule": "A-only class-standardized tail difference, then B-only fit",
              "l2": 1e-5, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [
        {"seed": run["seed"], "full_acc": run["full_150_fact_validation"]["acc"],
         "budgets": [{"concepts": row["total_concepts"],
                      "acc": row["validation"]["acc"],
                      "macro_f1": row["validation"]["macro_f1"]}
                     for row in run["budgets"]]}
        for run in runs]}, indent=2))


if __name__ == "__main__":
    main()
