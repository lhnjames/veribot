"""Dev-only attribution of MGTAB accuracy loss to structural or content compression."""

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
from screen_mgtab_20_hybrid_concepts import columns, concepts
from screen_mgtab_20_train_selected_concepts import rank_content
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def fit_eval(train_evidence, dev_evidence, groups, y_train, y_dev, a, b) -> dict:
    model = MonotoneSplineReasoner(groups, l2=1e-5)
    model.fit(rows(train_evidence, a), y_train[a],
              rows(train_evidence, b), y_train[b])
    probability = model.predict_proba(dev_evidence)
    threshold = best_validation_threshold(probability, y_dev)
    return {"validation": evaluate(y_dev, probability, threshold),
            "threshold": threshold,
            "feature_count": len(train_evidence.names)}


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
        selected16 = rank_content(atomic_train, y[train], a, count=16)
        original_groups = original["reasoner"]["groups"]
        compact_structure = concepts(atomic_train, list(range(86, 150)))
        compact_structure_dev = concepts(atomic_dev, list(range(86, 150)))
        compact_groups = {"published_profile": list(compact_structure.names[:2]),
                          "published_typed_relation": list(compact_structure.names[2:4])}
        for name in compact_structure.names[4:]:
            compact_groups[name] = [name]
        compact_result = fit_eval(compact_structure, compact_structure_dev,
                                  compact_groups, y[train], y[dev], a, b)
        retained = list(range(86)) + selected16
        raw_structure = columns(atomic_train, retained)
        raw_structure_dev = columns(atomic_dev, retained)
        raw_groups = {group: members for group, members in original_groups.items()
                      if group != "published_tweet_projections"}
        for j in selected16:
            raw_groups[names[j]] = [names[j]]
        raw_result = fit_eval(raw_structure, raw_structure_dev, raw_groups,
                              y[train], y[dev], a, b)
        full_replay = fit_eval(atomic_train, atomic_dev, original_groups,
                               y[train], y[dev], a, b)
        if any(abs(full_replay["validation"][key] - original["validation"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("full 150-fact validation result did not replay: " +
                             json.dumps({"seed": seed,
                                         "replayed": full_replay["validation"],
                                         "original": original["validation"]}))
        runs.append({
            "seed": seed, "split_sha256": original["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "selected16_content_fact_names": [names[j] for j in selected16],
            "compressed_structure_all64_content": compact_result,
            "all86_structure_selected16_content": raw_result,
            "full150_replay": full_replay,
        })
    result = {"dataset": "mgtab",
              "scope": "validation_only_structural_vs_content_compression_attribution",
              "test_predictions_accessed": False,
              "prior_test_results_already_observed": True,
              "l2": 1e-5, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{
        "seed": row["seed"],
        "compressed_structure_all64_content_acc": row[
            "compressed_structure_all64_content"]["validation"]["acc"],
        "all86_structure_selected16_content_acc": row[
            "all86_structure_selected16_content"]["validation"]["acc"],
        "full150_acc": row["full150_replay"]["validation"]["acc"],
    } for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
