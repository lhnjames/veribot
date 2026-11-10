"""Cluster paired ACC comparisons by account across repeated test appearances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file


DATASETS = {
    "mgtab": {
        "folder": "output/v3_mgtab_full_atomic_evidence_lightgbm",
        "baseline": "output/hypbot_mgtab_paired",
        "baseline_kind": "official_native_graph",
        "new_count_key": "only_new_correct",
        "baseline_count_key": "only_hypbot_correct",
    },
    "mgtab_ab_group": {
        "folder": "output/v3_mgtab_ab_group_additive_tree",
        "baseline": "output/hypbot_mgtab_paired",
        "baseline_kind": "official_native_graph",
        "new_count_key": "only_new_correct",
        "baseline_count_key": "only_hypbot_correct",
    },
    "fox8": {
        "folder": "output/v3_fox8_neighbor_atomic_evidence_capacity",
        "baseline": "output/hypbot_fox8_native_graph_paired",
        "baseline_kind": "source_matched_native_graph_variant_not_official_kNN",
        "new_count_key": "paired_veribot_only_correct",
        "baseline_count_key": "paired_native_hypbot_only_correct",
    },
}


def audit_one(root: Path, name: str, settings: dict) -> dict:
    folder = root / settings["folder"]
    report = json.loads((folder / "fixed_test_report.json").read_text(encoding="utf-8"))
    clusters: dict[str, list[int]] = {}
    runs = []
    for seed in (0, 1, 2):
        candidate_path = folder / f"seed{seed}/test_predictions.json"
        baseline_path = root / settings["baseline"] / f"seed{seed}/test_predictions.json"
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (report["runs"][seed]["predictions_sha256"] != sha256_file(candidate_path)
                or candidate["manifest"]["split_sha256"] !=
                baseline["manifest"]["split_sha256"]):
            raise ValueError("paired prediction artifacts or splits differ")
        paired = 0
        for mine, other in zip(candidate["predictions"], baseline["predictions"]):
            if (mine["account_id"] != other["account_id"]
                    or mine["label"] != other["label"]):
                raise ValueError("paired account IDs or labels differ")
            target = int(mine["label"])
            mine_ok = int((mine["probability"] >=
                           candidate["manifest"]["threshold"]) == target)
            other_ok = int((other["probability"] >=
                            baseline["manifest"]["threshold"]) == target)
            diff = mine_ok - other_ok
            account = mine["account_id"]
            if account not in clusters:
                clusters[account] = [0, 0]
            clusters[account][0] += diff
            clusters[account][1] += 1
            paired += diff
        if paired != (report["runs"][seed][settings["new_count_key"]] -
                      report["runs"][seed][settings["baseline_count_key"]]):
            raise ValueError("paired discordance differs from evaluation report")
        runs.append({"seed": seed, "test_accounts": len(candidate["predictions"]),
                     "paired_acc_delta": paired / len(candidate["predictions"])})
    counts = np.asarray(list(clusters.values()), dtype=np.int64)
    observed = float(counts[:, 0].sum() / counts[:, 1].sum())
    rng = np.random.default_rng(20261103)
    draws = np.empty(10_000, dtype=float)
    for start in range(0, len(draws), 500):
        sample = rng.integers(0, len(counts), size=(500, len(counts)))
        selected = counts[sample]
        draws[start:start + 500] = (selected[:, :, 0].sum(axis=1) /
                                      selected[:, :, 1].sum(axis=1))
    interval = [float(item) for item in np.quantile(draws, [0.025, 0.975])]
    return {
        "dataset": name, "baseline_kind": settings["baseline_kind"],
        "unique_account_ids": len(clusters),
        "test_appearances": int(counts[:, 1].sum()),
        "paired_acc_difference": observed,
        "account_cluster_bootstrap_95pct_ci": interval,
        "bootstrap_draws": len(draws),
        "prior_results_on_same_test_population_already_observed": True,
        "runs": runs,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = {
        "scope": "new_atomic_evidence_models_paired_accuracy_account_cluster_bootstrap",
        "resampling_unit": "account_id_across_all_seed_test_appearances",
        "datasets": [audit_one(root, name, settings)
                     for name, settings in DATASETS.items()],
    }
    path = root / "research/certificates/new_atomic_evidence_paired_cluster_intervals.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["datasets"]:
        print(row["dataset"], row["paired_acc_difference"],
              row["account_cluster_bootstrap_95pct_ci"], flush=True)


if __name__ == "__main__":
    main()
