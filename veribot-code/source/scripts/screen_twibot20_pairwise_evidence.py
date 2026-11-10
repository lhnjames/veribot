"""Official-dev-only screen of TwiBot-20 pairwise evidence reasoner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import groups, metric, select, sha256, threshold_on_dev
from verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from verifiable_bot.twibot20_pair_specs import fixed_pairs
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    with np.load(args.facts, allow_pickle=False) as cached:
        x = cached["values"].astype(float)
        y = cached["labels"].astype(int)
        split = cached["split"].astype(str)
        contexts = cached["contexts"].astype(str)
        names = cached["names"].astype(str).tolist()
    source = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    if source["dataset"] != "twibot20" or x.shape != (11826, len(names)):
        raise ValueError("official TwiBot-20 source cache differs")
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    if (len(train), len(dev)) != (8278, 2365):
        raise ValueError("official train/dev population differs")
    pairs = fixed_pairs(names)
    group_spec = groups(names)
    reference = LabelBlindReference(min_context_n=30).fit(
        x[train], names, contexts=contexts[train])
    evidence = reference.transform(x, contexts=contexts)
    runs = []
    for seed in args.seeds:
        a, b = train_test_split(train, test_size=0.5,
                                stratify=y[train], random_state=seed)
        results = {}
        for key, reasoner in (
            ("single", SignConstrainedReasoner(group_spec, l2=0.01)),
            ("pairwise", PairwiseEvidenceReasoner(group_spec, pairs, l2=0.01)),
        ):
            reasoner.fit(select(evidence, a), y[a], select(evidence, b), y[b])
            probability = reasoner.predict_proba(select(evidence, dev))
            threshold = threshold_on_dev(probability, y[dev])
            item = {"dev": metric(y[dev], probability, threshold),
                    "threshold": threshold}
            if isinstance(reasoner, PairwiseEvidenceReasoner):
                item["same_direction_pairs"] = int(sum(
                    reasoner.sign[i] == reasoner.sign[j] != 0
                    for i, j, _ in reasoner.pair_indices))
                item["nonzero_pair_coefficients"] = int(np.count_nonzero(
                    reasoner.coef[len(names):] > 0))
            results[key] = item
        runs.append({"seed": seed, **results})
    result = {
        "dataset": "twibot20", "scope": "official_dev_only",
        "test_predictions_accessed": False,
        "source_facts_sha256": sha256(args.facts),
        "source_archive_sha256": source["archive_sha256"],
        "reference_version": reference.version,
        "pair_count": len(pairs), "pairs": [list(pair) for pair in pairs],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"], "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
