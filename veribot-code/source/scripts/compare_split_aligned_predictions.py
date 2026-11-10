"""Compare frozen classifiers on identical test accounts and split hashes.

The two models may use different input features; this is an outcome comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    a = baseline["manifest"]
    b = candidate["manifest"]
    if (a["dataset"] != b["dataset"] or a["seed"] != b["seed"]
            or a["split_sha256"] != b["split_sha256"]):
        raise ValueError("dataset, seed or test split differs")
    old = {str(row["account_id"]): row for row in baseline["predictions"]}
    new = {str(row["account_id"]): row for row in candidate["predictions"]}
    if len(old) != len(baseline["predictions"]) or \
            len(new) != len(candidate["predictions"]) or set(old) != set(new):
        raise ValueError("test account IDs differ or duplicate")
    ids = sorted(old)
    y = np.asarray([int(old[uid]["label"]) for uid in ids])
    if not np.array_equal(y, [int(new[uid]["label"]) for uid in ids]):
        raise ValueError("test labels differ")
    prior = (np.asarray([old[uid]["probability"] for uid in ids]) >=
             float(a["threshold"])) == y
    after = (np.asarray([new[uid]["probability"] for uid in ids]) >=
             float(b["threshold"])) == y
    delta = after.astype(int) - prior.astype(int)
    rng = np.random.default_rng(20261102)
    draws = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean()
                        for _ in range(10_000)])
    return {
        "dataset": a["dataset"], "seed": a["seed"],
        "split_sha256": a["split_sha256"], "test_accounts": len(ids),
        "comparison_scope": "same test accounts, labels and split; feature and graph inputs may differ",
        "baseline": str(baseline_path), "candidate": str(candidate_path),
        "baseline_acc": float(prior.mean()),
        "candidate_acc": float(after.mean()),
        "paired_acc_difference": float(delta.mean()),
        "candidate_correct_baseline_wrong": int(np.sum(after & ~prior)),
        "baseline_correct_candidate_wrong": int(np.sum(prior & ~after)),
        "paired_account_bootstrap_95pct_ci": [float(v) for v in
                                               np.quantile(draws, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
