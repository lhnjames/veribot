"""Dev-only MGTAB capacity screen: 16 direct content coordinates + 4 source tails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def columns(evidence: EvidenceBatch, indices: list[int]) -> EvidenceBatch:
    return EvidenceBatch(evidence.upper[:, indices], evidence.lower[:, indices],
                         evidence.reliability[:, indices], evidence.available[:, indices],
                         evidence.reference_version,
                         tuple(evidence.names[j] for j in indices))


def concepts(evidence: EvidenceBatch,
             content_indices: list[int] | None = None) -> EvidenceBatch:
    names = list(evidence.names)
    selection_rule = ("first_16_train_only_PCA_coordinates" if content_indices is None
                      else f"explicit_{len(content_indices)}_train_selected_PCA_coordinates")
    if content_indices is None:
        content_indices = list(range(86, 102))
    if (len(names) != 150 or names[86] != "published_tweet_projection_000"
            or names[101] != "published_tweet_projection_015"
            or not 1 <= len(content_indices) <= 64
            or len(set(content_indices)) != len(content_indices)
            or any(not 86 <= j < 150 for j in content_indices)):
        raise ValueError("MGTAB 150-fact source vocabulary differs")
    structural = columns(evidence, list(range(86)))
    aggregate, _ = collapse_group_tails(structural, {
        "published_profile": names[:20],
        "published_typed_relation": names[20:86],
    })
    direct = columns(evidence, content_indices)
    payload = {"atomic_reference_version": evidence.reference_version,
               "rule": "two_source_tail_max_per_group_plus_" + selection_rule,
               "names": list(aggregate.names) + list(direct.names)}
    version = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    return EvidenceBatch(
        np.column_stack((aggregate.upper, direct.upper)),
        np.column_stack((aggregate.lower, direct.lower)),
        np.column_stack((aggregate.reliability, direct.reliability)),
        np.column_stack((aggregate.available, direct.available)),
        version, aggregate.names + direct.names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
            raise ValueError("selected reference differs")
        train, dev = split["train"], split["val"]
        concept_train = concepts(transform_with_support(
            reference, x[train], split["supports"][train]))
        concept_dev = concepts(transform_with_support(
            reference, x[dev], split["supports"][dev]))
        if concept_train.reference_version != concept_dev.reference_version:
            raise ValueError("train/dev concept version differs")
        groups = {"published_profile": list(concept_train.names[:2]),
                  "published_typed_relation": list(concept_train.names[2:4])}
        for name in concept_train.names[4:]:
            groups[name] = [name]
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        alternatives = {}
        for l2 in (1e-5, 1e-4, 1e-3):
            model = MonotoneSplineReasoner(groups, l2=l2)
            model.fit(rows(concept_train, a), y[train][a],
                      rows(concept_train, b), y[train][b])
            probability = model.predict_proba(concept_dev)
            threshold = best_validation_threshold(probability, y[dev])
            alternatives[str(l2)] = {
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
                "nonzero_slopes": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, "split_sha256": original["manifest"]["split_sha256"],
                     "reference_version": reference.version,
                     "concept_version": concept_train.reference_version,
                     "concept_count": len(concept_train.names),
                     "concept_names": list(concept_train.names),
                     "full_validation": original["validation"],
                     "alternatives": alternatives})
    result = {"dataset": "mgtab", "scope": "validation_only_20_hybrid_concept_capacity",
              "test_predictions_accessed": False,
              "prior_test_results_already_observed": True,
              "source_limit": "published_profile_edge_and_mean_tweet_tensors",
              "formula": "4 source-group tail maxima plus first 16 train-only PCA coordinates",
              "selected_model_dir": str(args.selected_model_dir), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": "mgtab", "runs": [
        {"seed": run["seed"], "concept_count": run["concept_count"],
         "alternatives": {key: val["validation"]
                          for key, val in run["alternatives"].items()}}
        for run in runs]}, indent=2))


if __name__ == "__main__":
    main()
