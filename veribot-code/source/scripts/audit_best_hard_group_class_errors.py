"""Class-specific hard-group errors for frozen best evidence and HypBot models."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_compact_activity_relation_strata import (
    CONFIG, ordered_predictions, raw_axes)
from gate0_shallow_tree import load


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/certificates/three_dataset_best_hard_group_class_errors.json"
CANDIDATES = {
    "botsim": "output/v3_computable/botsim_group_tail_12",
    "fox8": "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": "output/v3_mgtab_ab_group_additive_tree",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_result(mask: np.ndarray, labels: np.ndarray,
                 candidate: np.ndarray, baseline: np.ndarray,
                 seed: int, group_index: int) -> dict:
    y, c, b = labels[mask], candidate[mask], baseline[mask]
    n = len(y)
    if n == 0 or len(np.unique(y)) != 1:
        raise ValueError("class-conditioned hard group is empty or mixed")
    c_error = c != y
    b_error = b != y
    delta = c_error.astype(float) - b_error.astype(float)
    rng = np.random.default_rng(20261103 + 1000*seed + group_index)
    sampled = rng.integers(n, size=(5000, n))
    bootstrap = delta[sampled].mean(axis=1)
    return {
        "n": n, "label": int(y[0]),
        "evidence_errors": int(c_error.sum()),
        "hypbot_errors": int(b_error.sum()),
        "evidence_error_rate": float(c_error.mean()),
        "hypbot_error_rate": float(b_error.mean()),
        "paired_error_rate_difference": float(delta.mean()),
        "paired_bootstrap_95pct_ci": [float(v) for v in
                                       np.quantile(bootstrap, [0.025, 0.975])],
        "evidence_only_wrong": int(np.sum(c_error & ~b_error)),
        "hypbot_only_wrong": int(np.sum(b_error & ~c_error)),
    }


def audit_dataset(dataset: str, config: dict) -> dict:
    runs = []
    for seed in (0, 1, 2):
        x, labels, ids, names, split, source = load(
            dataset, Path(config["data"]),
            ROOT / config["account_file"] if config.get("account_file") else None,
            seed,
            ROOT / config["native_graph"] if dataset == "botsim" else None)
        axes = raw_axes(ROOT, dataset, x, ids, names, config)
        if len(axes) != 2:
            raise ValueError("expected two prespecified source axes")
        candidate_path = (ROOT / CANDIDATES[dataset] /
                          f"seed{seed}/test_predictions.json")
        baseline_path = (ROOT / config["baseline"] /
                         f"seed{seed}/test_predictions.json")
        _, candidate = ordered_predictions(
            candidate_path, ids, labels, split["test"],
            source["split_sha256"])
        _, baseline = ordered_predictions(
            baseline_path, ids, labels, split["test"],
            source["split_sha256"])
        test_labels = labels[split["test"]]
        axis_rows = []
        for axis_index, (name, values) in enumerate(axes.items()):
            finite = np.isfinite(values[split["train"]])
            if not finite.all():
                raise ValueError("hard group source axis contains missing train values")
            lower = float(np.quantile(values[split["train"]], 0.25))
            upper = float(np.quantile(values[split["train"]], 0.75))
            test_values = values[split["test"]]
            if not np.isfinite(test_values).all():
                raise ValueError("hard group source axis contains missing test values")
            groups = {
                "high_value_human_false_positive":
                    (test_values >= upper) & (test_labels == 0),
                "low_value_bot_false_negative":
                    (test_values <= lower) & (test_labels == 1),
            }
            axis_rows.append({
                "axis": name,
                "train_quantile_25": lower,
                "train_quantile_75": upper,
                "groups": {group: group_result(mask, test_labels,
                                                candidate, baseline, seed,
                                                axis_index*2 + j)
                           for j, (group, mask) in enumerate(groups.items())},
            })
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "fixed_test_accounts": len(split["test"]),
            "evidence_predictions_sha256": sha256(candidate_path),
            "hypbot_predictions_sha256": sha256(baseline_path),
            "axes": axis_rows,
        })
        print({"dataset": dataset, "seed": seed,
               "groups": [(axis["axis"], {key: (value["n"],
                                              value["evidence_errors"],
                                              value["hypbot_errors"])
                    for key, value in axis["groups"].items()})
                    for axis in axis_rows]}, flush=True)
    return {"dataset": dataset, "model": CANDIDATES[dataset], "runs": runs}


def main() -> None:
    result = {
        "scope": "frozen_best_evidence_vs_official_hypbot_class_specific_train_quantile_groups",
        "selection": "train source quartiles, fixed predictions and thresholds; no model or subgroup chosen by test outcome",
        "interpretation": "descriptive within-release subgroup audit; not OOD or a human audit of dataset labels",
        "datasets": [audit_dataset(name, config)
                     for name, config in CONFIG.items()],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
