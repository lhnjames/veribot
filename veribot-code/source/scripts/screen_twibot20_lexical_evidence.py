"""Official-dev-only screen of raw-tweet lexical evidence on TwiBot-20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import groups as base_groups, metric, select, sha256, threshold_on_dev
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--lexical-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    lexical_manifest = json.loads(args.lexical_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if sha256(args.lexical_facts) != lexical_manifest["facts_sha256"] or \
            sha256(args.base_facts) != lexical_manifest["base_facts_sha256"]:
        raise ValueError("TwiBot-20 lexical source cache differs")
    with np.load(args.base_facts, allow_pickle=False) as base:
        x = base["values"].astype(float)
        y = base["labels"].astype(int)
        ids = base["ids"].astype(str)
        split = base["split"].astype(str)
        contexts = base["contexts"].astype(str)
        positions = base["positions"].astype(int)
        names = base["names"].astype(str).tolist()
    with np.load(args.lexical_facts, allow_pickle=False) as cache:
        if not np.array_equal(ids, cache["ids"].astype(str)) or \
                not np.array_equal(split, cache["split"].astype(str)) or \
                not np.array_equal(positions, cache["positions"].astype(int)):
            raise ValueError("lexical facts differ from official account order")
        added_names = cache["names"].astype(str).tolist()
        added = cache["values"].astype(float)
    if len(names) != 22 or len(added_names) != 8 or added.shape != (len(x), 8) or \
            set(names) & set(added_names):
        raise ValueError("fixed TwiBot-20 evidence vocabulary differs")
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    if (len(train), len(dev)) != (8278, 2365):
        raise ValueError("official train/dev population differs")
    expanded = np.column_stack((x, added))
    candidate_names = names + added_names
    groups = {**base_groups(names), "tweet_text_structure": added_names}
    runs = []
    for seed in args.seeds:
        a, b = train_test_split(train, test_size=0.5,
                                stratify=y[train], random_state=seed)
        outputs = {}
        for label, matrix, vocabulary, model in (
            ("spline_22", x, names,
             MonotoneSplineReasoner(base_groups(names), l2=0.01)),
            ("dual_tail_22", x, names,
             DualTailEvidenceReasoner(base_groups(names), l2=0.01)),
            ("spline_30", expanded, candidate_names,
             MonotoneSplineReasoner(groups, l2=0.01)),
            ("dual_tail_30", expanded, candidate_names,
             DualTailEvidenceReasoner(groups, l2=0.01)),
        ):
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[train], vocabulary, contexts=contexts[train])
            evidence = reference.transform(matrix, contexts=contexts)
            model.fit(select(evidence, a), y[a], select(evidence, b), y[b])
            probability = model.predict_proba(select(evidence, dev))
            threshold = threshold_on_dev(probability, y[dev])
            outputs[label] = {
                "validation": metric(y[dev], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, **outputs})
    report = {
        "dataset": "twibot20", "scope": "official_dev_only",
        "test_predictions_accessed": False,
        "base_facts_sha256": sha256(args.base_facts),
        "lexical_facts_sha256": lexical_manifest["facts_sha256"],
        "source_archive_sha256": lexical_manifest["source_archive_sha256"],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "runs": [
        {"seed": row["seed"],
         "spline_22": row["spline_22"]["validation"],
         "dual_tail_22": row["dual_tail_22"]["validation"],
         "spline_30": row["spline_30"]["validation"],
         "dual_tail_30": row["dual_tail_30"]["validation"]}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
