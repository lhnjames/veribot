"""Pair two official-test evidence models on identical accounts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = json.loads(args.baseline.read_text(encoding="utf-8"))
    b = json.loads(args.candidate.read_text(encoding="utf-8"))
    if (a["manifest"]["dataset"] != b["manifest"]["dataset"]
            or a["manifest"]["official_split"] is not True
            or b["manifest"]["official_split"] is not True
            or a["manifest"]["source_archive_sha256"] !=
            b["manifest"]["source_archive_sha256"]
            or a["manifest"]["base_facts_sha256"] !=
            b["manifest"]["base_facts_sha256"]
            or a["manifest"]["reference_version"] !=
            b["manifest"]["reference_version"]):
        raise ValueError("paired official source or reference differs")
    old = a["predictions"]
    new = b["predictions"]
    if len(old) != 1183 or len(new) != len(old) or any(
            x["account_id"] != y["account_id"] or x["label"] != y["label"]
            for x, y in zip(old, new)):
        raise ValueError("official test accounts or labels differ")
    labels = np.asarray([row["label"] for row in old])
    prior = (np.asarray([row["probability"] for row in old]) >=
             a["manifest"]["threshold"]) == labels
    after = (np.asarray([row["probability"] for row in new]) >=
             b["manifest"]["threshold"]) == labels
    delta = after.astype(int) - prior.astype(int)
    rng = np.random.default_rng(20261102)
    draws = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean()
                        for _ in range(10_000)])
    result = {
        "dataset": "twibot20", "test_accounts": len(old),
        "comparison_scope": "identical official test accounts, base facts and label-blind reference; L1 C=1 reasoner; candidate adds active unigram evidence",
        "baseline_acc": float(prior.mean()),
        "candidate_acc": float(after.mean()),
        "paired_acc_difference": float(delta.mean()),
        "candidate_correct_baseline_wrong": int(np.sum(after & ~prior)),
        "baseline_correct_candidate_wrong": int(np.sum(prior & ~after)),
        "paired_account_bootstrap_95pct_ci": [float(x) for x in
                                               np.quantile(draws, [0.025, 0.975])],
        "baseline": str(args.baseline), "candidate": str(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
