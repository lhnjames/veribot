"""Audit compact evidence count, paired ACC and independent replay on three datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from compare_split_aligned_predictions import compare


CONFIG = {
    "botsim": {
        "directory": "output/v3_computable/botsim_group_tail_12",
        "count": 12,
        "source_level": "raw_user_post_comment_and_native_reply_graph",
        "verification_status": "all_source_to_concept_to_decision_cards_replayed",
        "verifier_probability_key": "max_probability_difference",
    },
    "fox8": {
        "directory": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
        "count": 20,
        "source_level": "raw_profile_sampled_tweets_and_native_targets",
        "verification_status": "all_source_to_concept_to_decision_cards_replayed",
        "verifier_probability_key": "max_probability_difference",
    },
    "mgtab": {
        "directory": "output/v3_mgtab_ab_spline_content/direct_selected_20",
        "count": 20,
        "source_level": "published_profile_edge_and_mean_tweet_tensors",
        "verification_status": "all_published_tensor_to_evidence_to_decision_cards_replayed",
        "verifier_probability_key": "max_logit_or_probability_difference",
    },
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root: Path, name: str, config: dict) -> dict:
    directory = root / config["directory"]
    verification = read(directory / "independent_verification.json")
    verifier = {row["seed"]: row for row in verification["runs"]}
    baseline_replay = read(
        root / "research" / "certificates" /
        f"hypbot_{name}_paired_checkpoint_replay.json")
    baseline_verified = {row["seed"]: row for row in baseline_replay["runs"]}
    if len(verifier) != 3:
        raise ValueError(f"{name}: independent verifier does not cover three seeds")
    if (baseline_replay["status"] != "all_paired_hypbot_checkpoints_replayed"
            or baseline_replay["dataset"] != name or len(baseline_verified) != 3):
        raise ValueError(f"{name}: HypBot checkpoint replay is incomplete")
    runs = []
    cluster: dict[str, list[int]] = {}
    for seed in (0, 1, 2):
        report = read(directory / f"seed{seed}" / "report.json")
        paired = read(directory / f"paired_vs_hypbot_seed{seed}.json")
        baseline_path = root / paired["baseline"]
        candidate_path = root / paired["candidate"]
        recomputed = compare(baseline_path, candidate_path)
        baseline_manifest = read(baseline_path)["manifest"]
        baseline_checkpoint = baseline_path.parent / "model.pt"
        if sha256(baseline_checkpoint) != baseline_manifest["model_sha256"]:
            raise ValueError(f"{name} seed {seed}: HypBot checkpoint hash differs")
        baseline_check = baseline_verified[seed]
        if (baseline_check["checkpoint_sha256"] != sha256(baseline_checkpoint)
                or baseline_check["prediction_file_sha256"] != sha256(baseline_path)
                or baseline_check["max_probability_difference"] > 1e-5
                or baseline_check["decision_mismatches"] != 0
                or baseline_check["test_accounts"] != recomputed["test_accounts"]
                or baseline_check["recomputed_acc"] != recomputed["baseline_acc"]):
            raise ValueError(f"{name} seed {seed}: HypBot frozen replay differs")
        checked = verifier[seed]
        manifest = report["manifest"]
        if (manifest["dataset"] != name or manifest["seed"] != seed
                or manifest["concept_count"] != config["count"]
                or manifest["prior_test_results_on_same_population_already_observed"] is not True
                or paired["dataset"] != name or paired["seed"] != seed
                or paired["split_sha256"] != manifest["split_sha256"]
                or paired["test_accounts"] != report["test"]["n"]
                or abs(paired["candidate_acc"] - report["test"]["acc"]) > 1e-12
                or checked["status"] != config["verification_status"]
                or checked["test_accounts"] != report["test"]["n"]
                or checked[config["verifier_probability_key"]] > 1e-12
                or abs(report["all_evidence_mask_probability"] -
                       float(expit(report["reasoner"]["intercept"]))) > 1e-12):
            raise ValueError(f"{name} seed {seed}: fixed model audit differs")
        for key in ("baseline_acc", "candidate_acc", "paired_acc_difference",
                    "candidate_correct_baseline_wrong",
                    "baseline_correct_candidate_wrong", "test_accounts",
                    "split_sha256"):
            if recomputed[key] != paired[key]:
                raise ValueError(f"{name} seed {seed}: paired {key} does not replay")
        if not np.array_equal(recomputed["paired_account_bootstrap_95pct_ci"],
                              paired["paired_account_bootstrap_95pct_ci"]):
            raise ValueError(f"{name} seed {seed}: paired interval does not replay")
        if paired["candidate_acc"] <= paired["baseline_acc"]:
            raise ValueError(f"{name} seed {seed}: compact ACC does not exceed HypBot")
        baseline_predictions = read(baseline_path)
        candidate_predictions = read(candidate_path)
        baseline_by_id = {str(row["account_id"]): row
                          for row in baseline_predictions["predictions"]}
        for row in candidate_predictions["predictions"]:
            uid = str(row["account_id"])
            prior = baseline_by_id[uid]
            y = int(row["label"])
            old_correct = int((prior["probability"] >=
                               baseline_predictions["manifest"]["threshold"]) == y)
            new_correct = int((row["probability"] >=
                               candidate_predictions["manifest"]["threshold"]) == y)
            if uid not in cluster:
                cluster[uid] = [0, 0]
            cluster[uid][0] += new_correct - old_correct
            cluster[uid][1] += 1
        runs.append({
            "seed": seed, "concept_count": config["count"],
            "test_accounts": checked["test_accounts"],
            "candidate_acc": paired["candidate_acc"],
            "hypbot_acc": paired["baseline_acc"],
            "paired_acc_delta": paired["paired_acc_difference"],
            "paired_bootstrap_95pct_ci": paired["paired_account_bootstrap_95pct_ci"],
            "verified_terms": checked["fact_terms"] if name == "mgtab"
                              else checked["concept_terms"],
            "hypbot_predictions_sha256": sha256(baseline_path),
            "hypbot_checkpoint_sha256": sha256(baseline_checkpoint),
            "hypbot_checkpoint_replayed": True,
            "candidate_predictions_sha256": sha256(candidate_path),
            "independent_verification_status": checked["status"],
            "all_evidence_mask_returns_intercept": True,
        })
    counts = np.asarray(list(cluster.values()), dtype=np.int64)
    observed = float(counts[:, 0].sum() / counts[:, 1].sum())
    rng = np.random.default_rng(20261103)
    draws = np.empty(10_000, dtype=float)
    for start in range(0, len(draws), 500):
        sample = rng.integers(0, len(counts), size=(500, len(counts)))
        selected = counts[sample]
        draws[start:start + 500] = (selected[:, :, 0].sum(axis=1) /
                                      selected[:, :, 1].sum(axis=1))
    cluster_interval = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    if abs(observed - np.mean([row["paired_acc_delta"] for row in runs])) > 1e-12:
        raise ValueError(f"{name}: account-cluster point estimate differs")
    return {"dataset": name, "source_level": config["source_level"],
            "mean_candidate_acc": float(np.mean([row["candidate_acc"] for row in runs])),
            "mean_hypbot_acc": float(np.mean([row["hypbot_acc"] for row in runs])),
            "all_three_numeric_acc_above_hypbot": True,
            "account_clustered_paired_acc": {
                "unique_accounts": len(cluster),
                "test_appearances": int(counts[:, 1].sum()),
                "paired_acc_difference": observed,
                "bootstrap_95pct_ci": cluster_interval,
                "resample_unit": "account_id_across_all_three_seed_appearances",
                "bootstrap_draws": len(draws),
            },
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = [audit_dataset(args.project_root, name, config)
                for name, config in CONFIG.items()]
    result = {
        "scope": "three_dataset_compact_numeric_evidence_model_audit",
        "datasets": datasets,
        "all_datasets_all_seeds_numeric_acc_above_same_account_hypbot": True,
        "claim_limits": {
            "paired_ci_excludes_zero_all_seeds": all(
                run["paired_bootstrap_95pct_ci"][0] > 0
                for dataset in datasets for run in dataset["runs"]),
            "account_clustered_ci_excludes_zero_all_datasets": all(
                dataset["account_clustered_paired_acc"]["bootstrap_95pct_ci"][0] > 0
                for dataset in datasets),
            "same_test_population_prior_candidates_observed": True,
            "mgtab_content_semantic_ground_truth": False,
            "external_semantic_validation": "not_proven_by_this_audit",
            "natural_error_correction_benefit": "not_proven_by_this_audit",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": [{
        "dataset": row["dataset"],
        "mean_candidate_acc": row["mean_candidate_acc"],
        "mean_hypbot_acc": row["mean_hypbot_acc"],
        "concept_count": row["runs"][0]["concept_count"],
    } for row in datasets]}, indent=2))


if __name__ == "__main__":
    main()
