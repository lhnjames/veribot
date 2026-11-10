"""Validation-only screen of source-derived MGTAB structure concepts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import groups_for, rows
from verifiable_bot.mgtab_source_certificate import RELATIONS
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def screen(x: np.ndarray, y: np.ndarray, names: list[str], groups: dict,
           split: dict, seed: int) -> dict:
    reference = LabelBlindReference(min_context_n=30).fit(x[split["train"]], names)
    evidence = reference.transform(x)
    a, b = train_test_split(split["train"], test_size=0.5,
                            stratify=y[split["train"]], random_state=seed)
    reasoner = SignConstrainedReasoner(groups, l2=0.01)
    reasoner.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
    probability = reasoner.predict_proba(rows(evidence, split["val"]))
    threshold = best_validation_threshold(probability, y[split["val"]])
    return {"metrics": evaluate(y[split["val"]], probability, threshold),
            "threshold": threshold, "reference_version": reference.version,
            "nonzero_coefficients": int(np.count_nonzero(reasoner.coef > 0)),
            "feature_count": len(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    feature_manifest = json.loads(args.structural_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if feature_manifest["facts_sha256"] != sha256_file(args.structural_facts):
        raise ValueError("structural fact artifact differs from source manifest")
    for name, expected in feature_manifest["input_sha256"].items():
        if sha256_file(args.data_dir / name) != expected:
            raise ValueError(f"MGTAB source changed: {name}")
    with np.load(args.structural_facts, allow_pickle=False) as cache:
        structural_ids = cache["ids"].astype(int)
        structural_names = cache["names"].astype(str).tolist()
        structural_values = cache["values"].astype(np.float32)
    if structural_names != feature_manifest["feature_names"]:
        raise ValueError("structural fact vocabulary differs from manifest")
    if not np.array_equal(structural_ids, np.arange(len(structural_ids))):
        raise ValueError("structural facts are not in published account order")
    if not np.isfinite(structural_values).all():
        raise ValueError("nonfinite structural fact")

    runs = []
    for seed in args.seeds:
        x, y, ids, names, split, manifest = load("mgtab", args.data_dir,
                                                   None, seed)
        if len(ids) != len(structural_ids):
            raise ValueError("structural fact population differs")
        baseline_groups = groups_for("mgtab", names)
        augmented_groups = {key: list(value) for key, value in baseline_groups.items()}
        for index, relation in enumerate(RELATIONS):
            expected_prefix = f"published_{relation}_"
            extras = structural_names[4*index:4*(index+1)]
            if len(extras) != 4 or any(not name.startswith(expected_prefix)
                                       for name in extras):
                raise ValueError("structural fact order differs from relation vocabulary")
            augmented_groups[f"published_relation_{index}"].extend(extras)
        augmented_x = np.column_stack((x, structural_values))
        graph_names = names[20:] + structural_names
        graph_x = np.column_stack((x[:, 20:], structural_values))
        graph_groups = {key: value for key, value in augmented_groups.items()
                        if key != "published_profile_coordinates"}
        runs.append({"seed": seed, "split_sha256": manifest["split_sha256"],
                     "baseline": screen(x, y, names, baseline_groups, split, seed),
                     "augmented": screen(augmented_x, y, names + structural_names,
                                         augmented_groups, split, seed),
                     "graph_only": screen(graph_x, y, graph_names,
                                          graph_groups, split, seed)})
    result = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "source_provenance": "published_typed_edge_tensor",
        "structural_facts_sha256": feature_manifest["facts_sha256"],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"],
                      "runs": [{"seed": row["seed"],
                                "baseline": row["baseline"]["metrics"],
                                "augmented": row["augmented"]["metrics"],
                                "graph_only": row["graph_only"]["metrics"]}
                               for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
