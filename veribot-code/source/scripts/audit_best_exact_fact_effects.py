"""All-test exact per-fact withholding for frozen best Fox8/MGTAB models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.high_performance_fact_effects import (
    account_fact_effects, withhold_all_facts)
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/certificates/high_performance_exact_fact_effects.json"
ARTIFACTS = ROOT / "output/v3_best_exact_fact_effects"
CONFIG = {
    "fox8": {
        "folder": "output/v3_fox8_neighbor_atomic_evidence_capacity",
        "groups": {"account_and_profile": slice(0, 23),
                   "sampled_tweet_activity": slice(23, 31),
                   "native_interaction": slice(31, 43),
                   "native_neighbor_means": slice(43, 74)},
    },
    "mgtab": {
        "folder": "output/v3_mgtab_ab_group_additive_tree",
        "groups": {"published_profile": slice(0, 20),
                   "published_typed_relations": slice(20, 86),
                   "published_tweet_projection": slice(86, 150)},
    },
}


def run(dataset: str, seed: int, config: dict) -> dict:
    runtime = (Fox8NeighborAtomicRuntime(ROOT, seed) if dataset == "fox8"
               else MGTABFullAtomicRuntime(ROOT, seed,
                                            variant="ab_group_additive"))
    _, labels, ids, _, split, source = load(
        dataset, runtime.data_dir, None, seed)
    saved_path = ROOT / config["folder"] / f"seed{seed}/test_predictions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    manifest = saved["manifest"]
    test = split["test"]
    test_ids = [ids[int(row)] for row in test]
    frozen = saved["predictions"]
    if (ids != runtime.ids or
            source["split_sha256"] != manifest["split_sha256"] or
            manifest["threshold"] != runtime.threshold or
            [row["account_id"] for row in frozen] != test_ids or
            [row["label"] for row in frozen] != labels[test].tolist()):
        raise ValueError("frozen test population differs")
    result = withhold_all_facts(runtime, test)
    probability = result["full_probability"]
    saved_probability = np.asarray([row["probability"] for row in frozen])
    max_probability_error = float(np.max(np.abs(
        probability - saved_probability)))
    if max_probability_error > 1e-12:
        raise ValueError("frozen evidence did not replay test probabilities")
    without_p = result["without_probability"]
    without_logit = result["without_logit"]
    delta_logit = result["full_logit"][:, None] - without_logit
    class_flip = (without_p >= runtime.threshold) != (
        probability >= runtime.threshold)[:, None]
    score_relevant = np.abs(delta_logit) > 1e-10
    if np.any(class_flip & ~score_relevant):
        raise ValueError("class flip without a logit effect")
    d = len(runtime.reference.names)
    if (without_p.shape != (len(test), d) or
            sorted(j for group in config["groups"].values()
                   for j in range(group.start, group.stop)) != list(range(d))):
        raise ValueError("fact withholding or source partition differs")
    sample_rows = sorted(set((0, len(test)//2, len(test)-1)))
    sample_max_error = 0.0
    for row in sample_rows:
        live = account_fact_effects(runtime, test_ids[row])
        sample_max_error = max(sample_max_error,
            max(abs(fact["probability_without_fact"] - without_p[row, j])
                for j, fact in enumerate(live["fact_effects"])))
        if (len(live["fact_effects"]) != d or
                live["prediction"] != int(probability[row] >=
                                          runtime.threshold)):
            raise ValueError("live fact effects differ from frozen batch")
    if sample_max_error > 1e-12:
        raise ValueError("live and batch effects differ")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS / f"{dataset}_seed{seed}.npz"
    np.savez_compressed(
        artifact, ids=np.asarray(test_ids),
        fact_names=np.asarray(runtime.reference.names),
        full_probability=probability,
        full_logit=result["full_logit"],
        without_probability=without_p,
        without_logit=without_logit,
        encoded_nonzero=result["encoded_nonzero"],
        threshold=np.asarray(runtime.threshold))
    groups = {}
    for name, members in config["groups"].items():
        groups[name] = {
            "fact_count": members.stop - members.start,
            "score_relevant_account_fact_pairs": int(
                np.count_nonzero(score_relevant[:, members])),
            "class_critical_account_fact_pairs": int(
                np.count_nonzero(class_flip[:, members])),
            "accounts_with_score_relevant_fact": int(
                np.count_nonzero(np.any(score_relevant[:, members], axis=1))),
            "accounts_with_class_critical_fact": int(
                np.count_nonzero(np.any(class_flip[:, members], axis=1))),
        }
    return {
        "dataset": dataset, "seed": seed,
        "test_accounts": len(test), "fact_count": d,
        "withholding_queries": int(without_p.size),
        "source_split_sha256": source["split_sha256"],
        "saved_predictions_sha256": sha256_file(saved_path),
        "model_sha256": manifest["model_sha256"],
        "reference_version": runtime.reference.version,
        "max_frozen_probability_replay_error": max_probability_error,
        "max_live_vs_batch_sample_effect_error": sample_max_error,
        "score_relevant_account_fact_pairs": int(np.count_nonzero(
            score_relevant)),
        "class_critical_account_fact_pairs": int(np.count_nonzero(
            class_flip)),
        "context_only_account_fact_pairs": int(
            without_p.size - np.count_nonzero(score_relevant)),
        "accounts_with_any_score_relevant_fact": int(np.count_nonzero(
            np.any(score_relevant, axis=1))),
        "accounts_with_any_class_critical_fact": int(np.count_nonzero(
            np.any(class_flip, axis=1))),
        "encoded_nonzero_account_fact_pairs": int(np.count_nonzero(
            result["encoded_nonzero"])),
        "source_groups": groups,
        "effect_matrix_path": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "effect_matrix_sha256": sha256_file(artifact),
    }


def main() -> None:
    runs = []
    for dataset, config in CONFIG.items():
        for seed in (0, 1, 2):
            row = run(dataset, seed, config)
            runs.append(row)
            print({key: row[key] for key in (
                "dataset", "seed", "withholding_queries",
                "score_relevant_account_fact_pairs",
                "class_critical_account_fact_pairs")}, flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "status": "frozen_best_full_test_exact_encoded_fact_withholding",
        "scope": "Fox8 74 and MGTAB 150 source-derived facts, each fixed test account and seed",
        "intervention": "set one fact's two encoded evidence tails to zero; do not modify original records or native graph",
        "interpretation": "Exact frozen-model input dependence, distinct from factual source verification and natural causal effect of account behavior.",
        "class_labels_used_to_select_effects": False,
        "test_labels_loaded_for_official_split_reconstruction": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
