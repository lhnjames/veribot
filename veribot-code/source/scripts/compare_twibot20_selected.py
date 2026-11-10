"""Paired official-test comparison of two frozen TwiBot-20 evidence models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(baseline: Path, candidate: Path, seed: int,
            baseline_report: str, candidate_report: str) -> tuple[dict, np.ndarray]:
    old_dir = baseline / f"seed{seed}"
    new_dir = candidate / f"seed{seed}"
    old_report = json.loads((old_dir / baseline_report).read_text(encoding="utf-8"))
    new_report = json.loads((new_dir / candidate_report).read_text(encoding="utf-8"))
    old_manifest = old_report["manifest"]
    new_manifest = new_report["manifest"]
    if old_manifest["dataset"] != new_manifest["dataset"] or \
            old_manifest["seed"] != seed or new_manifest["seed"] != seed or \
            old_manifest["facts_sha256"] != new_manifest.get(
                "facts_sha256", new_manifest.get("base_facts_sha256")) or \
            old_manifest.get("archive_sha256", old_manifest.get(
                "source_archive_sha256")) != new_manifest.get(
                "source_archive_sha256") or \
            old_manifest["official_split"] is not True or \
            new_manifest["official_split"] is not True:
        raise ValueError("paired models differ in official source population")
    old = json.loads((old_dir / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    new = json.loads((new_dir / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    if len(old) != 1183 or len(new) != len(old) or any(
        a["account_id"] != b["account_id"] or a["label"] != b["label"]
        for a, b in zip(old, new)
    ):
        raise ValueError("official test accounts or labels differ")
    labels = np.asarray([row["label"] for row in old], dtype=np.int8)
    old_ok = (np.asarray([row["probability"] for row in old]) >=
              old_manifest["threshold"]) == labels
    new_ok = (np.asarray([row["probability"] for row in new]) >=
              new_manifest["threshold"]) == labels
    return {
        "seed": seed, "test_accounts": len(old),
        "baseline_acc": float(old_ok.mean()),
        "candidate_acc": float(new_ok.mean()),
        "paired_acc_difference": float(new_ok.mean() - old_ok.mean()),
        "candidate_correct_baseline_wrong": int(np.sum(new_ok & ~old_ok)),
        "baseline_correct_candidate_wrong": int(np.sum(old_ok & ~new_ok)),
    }, new_ok.astype(np.int8) - old_ok.astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-report", default="report.json")
    parser.add_argument("--candidate-report", default="evidence_report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [compare(args.baseline, args.candidate, seed,
                     args.baseline_report, args.candidate_report)
             for seed in (0, 1, 2)]
    rng = np.random.default_rng(20261102)
    draws = np.empty(10_000)
    for k in range(len(draws)):
        selected = rng.integers(0, 3, size=3)
        draws[k] = np.mean([
            np.mean(pairs[i][1][rng.integers(0, len(pairs[i][1]),
                                              size=len(pairs[i][1]))])
            for i in selected])
    result = {
        "dataset": "twibot20",
        "comparison": "same official test accounts, base fact cache, and source archive; candidate may add audited facts",
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "runs": [pair[0] for pair in pairs],
        "mean_paired_acc_difference": float(np.mean([
            pair[0]["paired_acc_difference"] for pair in pairs])),
        "two_stage_paired_bootstrap_95pct_ci": [float(value) for value in
                                                  np.quantile(draws, [0.025, 0.975])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
