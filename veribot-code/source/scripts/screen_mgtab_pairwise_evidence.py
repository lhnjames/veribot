"""Validation-only MGTAB screen of same-direction pairwise evidence."""

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
from verifiable_bot.mgtab_pair_specs import fixed_pairs
from verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    cache_manifest = json.loads(args.structural_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if sha256_file(args.structural_facts) != cache_manifest["facts_sha256"]:
        raise ValueError("structural facts differ from source manifest")
    for name, expected in cache_manifest["input_sha256"].items():
        if sha256_file(args.data_dir / name) != expected:
            raise ValueError(f"MGTAB source changed: {name}")
    with np.load(args.structural_facts, allow_pickle=False) as cache:
        cache_ids = cache["ids"].astype(int)
        structural_names = cache["names"].astype(str).tolist()
        structural = cache["values"].astype(np.float32)
    runs = []
    for seed in args.seeds:
        x, y, ids, base_names, split, source = load("mgtab", args.data_dir,
                                                     None, seed)
        if not np.array_equal(cache_ids, np.arange(len(ids))) or \
                structural_names != cache_manifest["feature_names"]:
            raise ValueError("structural fact population or vocabulary differs")
        names = base_names + structural_names
        groups = groups_for("mgtab", names)
        pairs = fixed_pairs(names)
        reference = LabelBlindReference(min_context_n=30).fit(
            np.column_stack((x, structural))[split["train"]], names)
        evidence = reference.transform(np.column_stack((x, structural)))
        a, b = train_test_split(split["train"], test_size=0.5,
                                stratify=y[split["train"]], random_state=seed)
        results = {}
        for name, reasoner in (
            ("base_62", SignConstrainedReasoner(groups, l2=0.01)),
            ("pairwise_62", PairwiseEvidenceReasoner(groups, pairs, l2=0.01)),
        ):
            reasoner.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = reasoner.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(probability, y[split["val"]])
            result = {"validation": evaluate(y[split["val"]], probability,
                                              threshold),
                      "threshold": threshold,
                      "nonzero_single_coefficients": int(np.count_nonzero(
                          reasoner.coef[:len(names)] > 0))}
            if isinstance(reasoner, PairwiseEvidenceReasoner):
                active = [int(reasoner.sign[i] == reasoner.sign[j] != 0)
                          for i, j, _ in reasoner.pair_indices]
                result["active_same_direction_pairs"] = sum(active)
                result["nonzero_pair_coefficients"] = int(np.count_nonzero(
                    reasoner.coef[len(names):] > 0))
            results[name] = result
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version, **results})
    report = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "structural_facts_sha256": cache_manifest["facts_sha256"],
        "pair_count": len(pairs), "pairs": [list(pair) for pair in pairs],
        "pair_basis": "same A-split sign times minimum of two signed evidence magnitudes; zero if signs differ",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "pair_count": len(pairs),
                      "runs": [{"seed": row["seed"],
                                "base_62": row["base_62"],
                                "pairwise_62": row["pairwise_62"]}
                               for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
