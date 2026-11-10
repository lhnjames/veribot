"""Compare frozen v3 candidates on identical test accounts and source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_run(baseline: Path, candidate: Path, seed: int) -> tuple[dict, np.ndarray]:
    b = json.loads((baseline / f"seed{seed}" / "test_predictions.json").read_text(
        encoding="utf-8"))
    c = json.loads((candidate / f"seed{seed}" / "test_predictions.json").read_text(
        encoding="utf-8"))
    bm, cm = b["manifest"], c["manifest"]
    for field in ("dataset", "seed", "split_sha256", "population", "split_sizes"):
        if bm[field] != cm[field]:
            raise ValueError(f"paired protocol differs in {field}")
    for path, digest in bm["files_sha256"].items():
        if cm["files_sha256"].get(path) != digest:
            raise ValueError(f"common source file differs: {path}")
    br = {str(row["account_id"]): row for row in b["predictions"]}
    cr = {str(row["account_id"]): row for row in c["predictions"]}
    if len(br) != len(b["predictions"]) or len(cr) != len(c["predictions"]) or \
            br.keys() != cr.keys() or len(br) != bm["split_sizes"]["test"]:
        raise ValueError("paired test account population differs")
    ids = sorted(br)
    y = np.asarray([int(br[uid]["label"]) for uid in ids], dtype=np.int8)
    if any(int(cr[uid]["label"]) != label for uid, label in zip(ids, y)):
        raise ValueError("paired labels differ")
    bp = np.asarray([float(br[uid]["probability"]) for uid in ids])
    cp = np.asarray([float(cr[uid]["probability"]) for uid in ids])
    if not (np.isfinite(bp).all() and np.isfinite(cp).all()):
        raise ValueError("nonfinite saved probability")
    bok = (bp >= float(bm["threshold"])) == y
    cok = (cp >= float(cm["threshold"])) == y
    return {
        "dataset": bm["dataset"], "seed": seed, "n_test": len(ids),
        "baseline_acc": float(np.mean(bok)),
        "candidate_acc": float(np.mean(cok)),
        "paired_acc_difference": float(np.mean(cok.astype(int) - bok.astype(int))),
        "candidate_correct_baseline_wrong": int(np.sum(cok & ~bok)),
        "baseline_correct_candidate_wrong": int(np.sum(bok & ~cok)),
    }, cok.astype(np.int8) - bok.astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--bootstraps", type=int, default=10_000)
    args = parser.parse_args()
    rows, differences = zip(*(paired_run(args.baseline, args.candidate, seed)
                              for seed in args.seeds))
    rng = np.random.default_rng(20261101)
    draws = np.empty(args.bootstraps)
    for k in range(args.bootstraps):
        selected = rng.integers(0, len(differences), size=len(differences))
        draws[k] = np.mean([
            np.mean(differences[i][rng.integers(0, len(differences[i]),
                                               size=len(differences[i]))])
            for i in selected
        ])
    result = {
        "dataset": rows[0]["dataset"],
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "seeds": list(args.seeds), "runs": list(rows),
        "baseline_mean_acc": float(np.mean([row["baseline_acc"] for row in rows])),
        "candidate_mean_acc": float(np.mean([row["candidate_acc"] for row in rows])),
        "mean_paired_acc_difference": float(np.mean([
            row["paired_acc_difference"] for row in rows])),
        "paired_two_stage_bootstrap_95pct_ci": [float(x) for x in
                                                   np.quantile(draws, [0.025, 0.975])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
