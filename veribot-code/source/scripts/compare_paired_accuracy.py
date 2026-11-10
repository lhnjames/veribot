"""Strict paired ACC gate for VeriBot v2 against HypBot.

Each input is a JSON file with a ``manifest`` and test ``predictions``. The
manifest records the dataset, seed, source/split/feature/graph hashes and the
validation-selected threshold. Rows contain account_id, label, and probability.
Run this only after both models have frozen their checkpoints and thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAPER_ACC = {"mgtab": 0.9118, "fox8": 0.9927, "botsim": 0.9947}
REQUIRED_MANIFEST = (
    "dataset", "seed", "raw_sha256", "split_sha256",
    "feature_sha256", "graph_sha256", "threshold", "model_sha256",
)
PAIRED_FIELDS = ("dataset", "seed", "raw_sha256", "split_sha256", "feature_sha256", "graph_sha256")


def load_run(path: Path) -> tuple[dict, dict[str, tuple[int, float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    manifest = obj["manifest"]
    missing = [key for key in REQUIRED_MANIFEST if key not in manifest]
    if missing:
        raise ValueError(f"{path}: missing manifest fields {missing}")
    threshold = float(manifest["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{path}: threshold must be validation-selected probability in (0,1)")
    rows: dict[str, tuple[int, float]] = {}
    for row in obj["predictions"]:
        if row.get("split") != "test":
            raise ValueError(f"{path}: every prediction must be on the test split")
        uid = str(row["account_id"])
        if uid in rows:
            raise ValueError(f"{path}: repeated test account {uid}")
        label, probability = int(row["label"]), float(row["probability"])
        if label not in (0, 1) or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{path}: invalid label/probability for {uid}")
        rows[uid] = (label, probability)
    if not rows:
        raise ValueError(f"{path}: empty test predictions")
    return manifest, rows


def compare(baseline_path: Path, candidate_path: Path) -> dict:
    b_manifest, b_rows = load_run(baseline_path)
    c_manifest, c_rows = load_run(candidate_path)
    for key in PAIRED_FIELDS:
        if b_manifest[key] != c_manifest[key]:
            raise ValueError(f"protocol mismatch: {key}: {b_manifest[key]!r} != {c_manifest[key]!r}")
    if b_rows.keys() != c_rows.keys():
        raise ValueError("test account sets differ; paired comparison is invalid")
    ids = sorted(b_rows)
    y = np.array([b_rows[uid][0] for uid in ids], dtype=np.int8)
    if np.any(y != np.array([c_rows[uid][0] for uid in ids], dtype=np.int8)):
        raise ValueError("test labels differ for matched accounts")
    b_prob = np.array([b_rows[uid][1] for uid in ids], dtype=float)
    c_prob = np.array([c_rows[uid][1] for uid in ids], dtype=float)
    b_ok = (b_prob >= float(b_manifest["threshold"])) == y
    c_ok = (c_prob >= float(c_manifest["threshold"])) == y
    win = int(np.sum(c_ok & ~b_ok))
    lose = int(np.sum(b_ok & ~c_ok))
    return {
        "dataset": b_manifest["dataset"], "seed": b_manifest["seed"], "n_test": len(ids),
        "baseline_acc": float(b_ok.mean()), "candidate_acc": float(c_ok.mean()),
        "acc_difference": float(c_ok.mean() - b_ok.mean()),
        "candidate_correct_baseline_wrong": win,
        "baseline_correct_candidate_wrong": lose,
        "both_correct": int(np.sum(b_ok & c_ok)), "both_wrong": int(np.sum(~b_ok & ~c_ok)),
        "baseline_path": str(baseline_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "paired_correctness": (c_ok.astype(np.int8) - b_ok.astype(np.int8)).tolist(),
    }


def summarize(rows: list[dict], paper_acc: float | None = None, bootstraps: int = 10_000) -> dict:
    if not rows:
        raise ValueError("at least one paired run is required")
    datasets = {row["dataset"] for row in rows}
    seeds = [row["seed"] for row in rows]
    if len(datasets) != 1 or len(set(seeds)) != len(seeds):
        raise ValueError("one dataset and unique seeds are required")
    differences = np.array([row["acc_difference"] for row in rows], dtype=float)
    base = np.array([row["baseline_acc"] for row in rows], dtype=float)
    cand = np.array([row["candidate_acc"] for row in rows], dtype=float)
    # Two-stage bootstrap keeps paired account outcomes and seed variation together.
    rng = np.random.default_rng(20261101)
    per_seed = [np.asarray(row["paired_correctness"], dtype=np.int8) for row in rows]
    draws = np.empty(bootstraps, dtype=float)
    for b in range(bootstraps):
        chosen = rng.integers(0, len(rows), size=len(rows))
        draws[b] = np.mean([
            per_seed[i][rng.integers(0, len(per_seed[i]), size=len(per_seed[i]))].mean()
            for i in chosen
        ])
    target = PAPER_ACC.get(next(iter(datasets))) if paper_acc is None else paper_acc
    result = {
        "dataset": next(iter(datasets)), "seeds": seeds,
        "baseline_mean_acc": float(base.mean()), "candidate_mean_acc": float(cand.mean()),
        "mean_paired_acc_difference": float(differences.mean()),
        "paired_bootstrap_95pct_ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "paper_reference_acc": target,
        "beats_paired_baseline": bool(cand.mean() > base.mean()),
        "beats_paper_reference": bool(target is not None and cand.mean() > target),
        "three_seed_gate": bool(len(rows) >= 3 and cand.mean() > base.mean()
                                and target is not None and cand.mean() > target),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstraps", type=int, default=10_000)
    args = parser.parse_args()
    if len(args.baseline) != len(args.candidate):
        parser.error("provide one candidate file for each baseline file")
    rows = [compare(b, c) for b, c in zip(args.baseline, args.candidate)]
    summary = summarize(rows, bootstraps=args.bootstraps)
    report = {"summary": summary, "runs": [
        {k: v for k, v in row.items() if k != "paired_correctness"} for row in rows
    ]}
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
