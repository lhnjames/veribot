"""Paired account-cluster uncertainty for compact MGTAB versus HypBot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    report_path = folder / "fixed_test_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_account: dict[str, list[int]] = {}
    runs = []
    for seed in (0, 1, 2):
        new_path = folder / f"seed{seed}/test_predictions.json"
        base_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        new = json.loads(new_path.read_text(encoding="utf-8"))
        base = json.loads(base_path.read_text(encoding="utf-8"))
        a, b = new["predictions"], base["predictions"]
        if (len(a) != len(b) or len(a) != report["runs"][seed]["test_accounts"]
                or new["manifest"]["split_sha256"] !=
                base["manifest"]["split_sha256"]):
            raise ValueError("Paired MGTAB population differs")
        only_new = only_base = 0
        for left, right in zip(a, b):
            if (left["account_id"] != right["account_id"]
                    or left["label"] != right["label"]):
                raise ValueError("Account-level pair differs")
            new_correct = int((left["probability"] >= new["manifest"]["threshold"])
                              == left["label"])
            base_correct = int((right["probability"] >= base["manifest"]["threshold"])
                               == right["label"])
            only_new += int(new_correct == 1 and base_correct == 0)
            only_base += int(new_correct == 0 and base_correct == 1)
            by_account.setdefault(left["account_id"], []).append(new_correct - base_correct)
        if (only_new != report["runs"][seed]["paired_only_veribot_correct"]
                or only_base != report["runs"][seed]["paired_only_hypbot_correct"]):
            raise ValueError("Saved paired count differs")
        runs.append({"seed": seed, "new_only_correct": only_new,
                     "hypbot_only_correct": only_base,
                     "net_correct": only_new - only_base,
                     "exact_discordance_p_two_sided": float(binomtest(
                         only_new, only_new + only_base, 0.5).pvalue),
                     "new_prediction_sha256": sha256_file(new_path),
                     "hypbot_prediction_sha256": sha256_file(base_path)})
    clusters = list(by_account.values())
    sums = np.asarray([sum(row) for row in clusters], dtype=float)
    counts = np.asarray([len(row) for row in clusters], dtype=float)
    rng = np.random.default_rng(20261103)
    draws = rng.integers(0, len(clusters), size=(10000, len(clusters)), dtype=np.int32)
    differences = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    observed = float(sums.sum() / counts.sum())
    output = {
        "scope": "MGTAB_compact_20_concept_vs_same_account_HypBot",
        "fixed_test_report_sha256": sha256_file(report_path),
        "account_clusters": len(clusters),
        "seed_test_rows": int(counts.sum()),
        "accuracy_difference": observed,
        "cluster_bootstrap_95_percentile_interval": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975))],
        "bootstrap_draws": len(differences),
        "prior_same_test_population_exploration": True,
        "runs": runs,
    }
    path = root / "research/certificates/mgtab_targeted_20_concept_paired.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"difference": observed,
                      "interval": output["cluster_bootstrap_95_percentile_interval"],
                      "runs": runs}), flush=True)


if __name__ == "__main__":
    main()
