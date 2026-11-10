"""Official-dev-only screen of source-addressable token evidence."""

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
    parser.add_argument("--token-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    token_manifest = json.loads(args.token_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (token_manifest["facts_sha256"] != sha256(args.token_facts)
            or token_manifest["base_facts_sha256"] != sha256(args.base_facts)
            or token_manifest["label_fields_read"] is not False):
        raise ValueError("token cache or label-blind vocabulary manifest differs")
    with np.load(args.base_facts, allow_pickle=False) as base:
        x = base["values"].astype(float)
        y = base["labels"].astype(int)
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        positions = base["positions"].astype(int)
        contexts = base["contexts"].astype(str)
        names = base["names"].astype(str).tolist()
    with np.load(args.token_facts, allow_pickle=False) as cache:
        if (not np.array_equal(ids, cache["ids"].astype(str))
                or not np.array_equal(parts, cache["split"].astype(str))
                or not np.array_equal(positions, cache["positions"].astype(int))):
            raise ValueError("token facts differ from official account population")
        token_names = cache["names"].astype(str).tolist()
        token_values = cache["values"].astype(float)
    if (len(names) != 22 or len(token_names) != 128
            or token_names != token_manifest["fact_names"]):
        raise ValueError("unexpected evidence vocabulary")
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if (len(train), len(dev)) != (8278, 2365):
        raise ValueError("official train/dev split differs")
    candidates = (("dual_tail_22", 0), ("dual_tail_token64", 64),
                  ("dual_tail_token128", 128))
    runs = []
    for seed in args.seeds:
        a, b = train_test_split(train, test_size=0.5,
                                stratify=y[train], random_state=seed)
        outputs = {}
        for label, n_tokens in candidates:
            vocabulary = names + token_names[:n_tokens]
            matrix = np.column_stack((x, token_values[:, :n_tokens]))
            groups = base_groups(names)
            if n_tokens:
                groups["tweet_token_incidence"] = token_names[:n_tokens]
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[train], vocabulary, contexts=contexts[train])
            evidence = reference.transform(matrix, contexts=contexts)
            reasoner = DualTailEvidenceReasoner(groups, l2=0.01)
            reasoner.fit(select(evidence, a), y[a], select(evidence, b), y[b])
            probability = reasoner.predict_proba(select(evidence, dev))
            threshold = threshold_on_dev(probability, y[dev])
            outputs[label] = {
                "validation": metric(y[dev], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_coefficients": int(np.count_nonzero(reasoner.coef > 0)),
            }
        runs.append({"seed": seed, **outputs})
    report = {
        "dataset": "twibot20", "scope": "official_dev_only",
        "test_predictions_accessed": False,
        "base_facts_sha256": sha256(args.base_facts),
        "token_facts_sha256": token_manifest["facts_sha256"],
        "source_archive_sha256": token_manifest["source_archive_sha256"],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "runs": [
        {"seed": row["seed"], **{name: row[name]["validation"]
                                 for name, _ in candidates}} for row in runs]},
        indent=2))


if __name__ == "__main__":
    main()
