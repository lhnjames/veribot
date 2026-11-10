"""Official-dev-only screen of raw-source follow-neighbor profile evidence."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    neighbor_manifest = json.loads(args.neighbor_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (neighbor_manifest["facts_sha256"] != sha256(args.neighbor_facts)
            or neighbor_manifest["base_facts_sha256"] != sha256(args.base_facts)
            or audit["fact_cache_sha256"] != sha256(args.neighbor_facts)
            or audit["status"] !=
            "verified_all_follow_neighbor_profile_fact_cells_against_official_zip"):
        raise ValueError("follow-neighbor source audit is missing or differs")
    with np.load(args.base_facts, allow_pickle=False) as base:
        x = base["values"].astype(float)
        y = base["labels"].astype(int)
        ids = base["ids"].astype(str)
        split = base["split"].astype(str)
        context = base["contexts"].astype(str)
        names = base["names"].astype(str).tolist()
        positions = base["positions"].astype(int)
    with np.load(args.neighbor_facts, allow_pickle=False) as cache:
        if (not np.array_equal(ids, cache["ids"].astype(str))
                or not np.array_equal(split, cache["split"].astype(str))
                or not np.array_equal(positions, cache["positions"].astype(int))):
            raise ValueError("official account order differs")
        added = cache["values"].astype(float)
        added_names = cache["names"].astype(str).tolist()
    if len(names) != 22 or len(added_names) != 10 or set(names) & set(added_names):
        raise ValueError("candidate vocabulary differs")
    expanded = np.column_stack((x, added))
    expanded_names = names + added_names
    expanded_groups = {**base_groups(names),
                       "follow_neighbor_profile": added_names}
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    if (len(train), len(dev)) != (8278, 2365):
        raise ValueError("official train/dev population differs")
    runs = []
    for seed in (0, 1, 2):
        a, b = train_test_split(train, test_size=0.5,
                                stratify=y[train], random_state=seed)
        results = {}
        for label, matrix, vocabulary, groups in (
                ("dual_tail_22", x, names, base_groups(names)),
                ("dual_tail_32_neighbor", expanded, expanded_names,
                 expanded_groups)):
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[train], vocabulary, contexts=context[train])
            evidence = reference.transform(matrix, contexts=context)
            model = DualTailEvidenceReasoner(groups, l2=0.01)
            model.fit(select(evidence, a), y[a], select(evidence, b), y[b])
            probability = model.predict_proba(select(evidence, dev))
            threshold = threshold_on_dev(probability, y[dev])
            results[label] = {
                "validation": metric(y[dev], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, **results})
    result = {
        "dataset": "twibot20", "scope": "official_dev_only",
        "test_predictions_accessed": False,
        "base_facts_sha256": sha256(args.base_facts),
        "neighbor_facts_sha256": sha256(args.neighbor_facts),
        "source_audit_sha256": sha256(args.source_audit),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
