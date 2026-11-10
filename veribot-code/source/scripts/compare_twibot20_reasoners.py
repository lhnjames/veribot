"""Paired official-test comparison of linear and validation-selected spline reasoners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def compare(seed: int, linear_dir: Path, spline_dir: Path) -> tuple[dict, np.ndarray]:
    old = linear_dir / f"seed{seed}"
    new = spline_dir / f"seed{seed}"
    base_report = json.loads((old / "evidence_report.json").read_text(encoding="utf-8"))
    new_report = json.loads((new / "report.json").read_text(encoding="utf-8"))
    if base_report["manifest"]["facts_sha256"] != new_report["manifest"]["facts_sha256"]:
        raise ValueError("reasoners used different fact caches")
    base = json.loads((old / "test_predictions.json").read_text(encoding="utf-8"))["predictions"]
    candidate = json.loads((new / "test_predictions.json").read_text(encoding="utf-8"))["predictions"]
    if len(base) != 1183 or len(candidate) != len(base) or any(
        a["account_id"] != b["account_id"] or a["label"] != b["label"]
        for a, b in zip(base, candidate)
    ):
        raise ValueError("official test accounts or labels do not align")
    labels = np.array([row["label"] for row in base])
    old_ok = (np.array([row["probability"] for row in base]) >=
              base_report["manifest"]["threshold"]) == labels
    new_ok = (np.array([row["probability"] for row in candidate]) >=
              new_report["manifest"]["threshold"]) == labels
    gained = int(np.sum(new_ok & ~old_ok))
    lost = int(np.sum(old_ok & ~new_ok))
    p = float(binomtest(min(gained, lost), n=gained + lost, p=0.5).pvalue) if gained + lost else 1.0
    return {"seed": seed, "n": len(base),
            "linear_acc": float(old_ok.mean()), "spline_acc": float(new_ok.mean()),
            "acc_difference": float(new_ok.mean() - old_ok.mean()),
            "spline_correct_linear_wrong": gained,
            "linear_correct_spline_wrong": lost,
            "exact_paired_two_sided_p": p}, new_ok.astype(int) - old_ok.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-dir", type=Path, required=True)
    parser.add_argument("--spline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    pairs = [compare(seed, args.linear_dir, args.spline_dir) for seed in args.seeds]
    rng = np.random.default_rng(20261101)
    draws = np.empty(5000)
    for j in range(len(draws)):
        chosen = rng.integers(0, len(pairs), size=len(pairs))
        draws[j] = np.mean([pairs[i][1][rng.integers(0, len(pairs[i][1]),
                                                   size=len(pairs[i][1]))].mean()
                            for i in chosen])
    report = {"dataset": "twibot20", "comparison": "same official test accounts and fact cache",
              "mean_acc_difference": float(np.mean([pair[0]["acc_difference"] for pair in pairs])),
              "two_stage_paired_bootstrap_95pct_ci": [float(q) for q in
                                                       np.quantile(draws, [0.025, 0.975])],
              "runs": [pair[0] for pair in pairs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
