"""Verify exact one-concept withholding for every BotSim fixed-test account."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verify_compact_raw_source_predictions import CONFIG, source_inputs
from verifiable_bot.compact_concept_effects import concept_effects
from verifiable_bot.compact_evidence_runtime import CompactEvidenceRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_BOT = CONFIG["botsim"]
OUTPUT = ROOT / "research/certificates/botsim_best_exact_concept_effects.json"
ARTIFACTS = ROOT / "output/v3_best_exact_fact_effects"


def run(seed: int) -> dict:
    selected = ROOT / CONFIG_BOT["selected"] / f"seed{seed}" / "test_predictions.json"
    saved = json.loads(selected.read_text(encoding="utf-8"))
    frozen = saved["predictions"]
    ids = [row["account_id"] for row in frozen]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fixed test account")
    deployment = ROOT / "output/v3_current_compact_deployment"
    runtime = CompactEvidenceRuntime(
        "botsim", ROOT / CONFIG_BOT["selected"], ROOT / CONFIG_BOT["full"],
        seed, deployment_metadata=deployment / f"botsim_runtime_seed{seed}.json")
    names, facts, source_error, source_level = source_inputs(
        ROOT, "botsim", CONFIG_BOT, ids, seed=seed,
        inference_metadata=deployment / f"botsim_source_seed{seed}.json")
    import pandas as pd
    all_ids = pd.read_csv(ROOT / CONFIG_BOT["account"])["user_id"].astype(str).tolist()
    index = {account: i for i, account in enumerate(all_ids)}
    if len(index) != len(all_ids) or not set(ids) <= set(index):
        raise ValueError("raw account population differs")
    values = facts[[index[account] for account in ids]]
    decisions = runtime.predict(values, names, ids)
    evidence, _ = runtime._evidence(values, names)
    d = len(evidence.names)
    if d != 12 or len(decisions) != len(frozen):
        raise ValueError("selected BotSim model differs")
    full_probability = np.asarray([row["probability"] for row in decisions])
    saved_probability = np.asarray([row["probability"] for row in frozen])
    replay_error = float(np.max(np.abs(full_probability - saved_probability)))
    if replay_error > 1e-12 or any(
            decision["account_id"] != row["account_id"] for decision, row
            in zip(decisions, frozen, strict=True)):
        raise ValueError("live BotSim decisions differ from frozen test")
    without_probability = np.column_stack([
        np.asarray([row["probability_without_concept"] for row in
                    concept_effects(decision)["concept_effects"]])
        for decision in decisions
    ]).T
    if without_probability.shape != (len(ids), d):
        raise ValueError("concept effect matrix differs")
    direct_error = 0.0
    for j in range(d):
        upper = evidence.upper.copy()
        lower = evidence.lower.copy()
        upper[:, j] = 0.0
        lower[:, j] = 0.0
        altered = EvidenceBatch(upper, lower, evidence.reliability,
                                evidence.available, evidence.reference_version,
                                evidence.names)
        direct = runtime.reasoner.predict_proba(altered)
        direct_error = max(direct_error, float(np.max(np.abs(
            direct - without_probability[:, j]))))
    if direct_error > 1e-12:
        raise ValueError("additive calculation differs from direct concept masking")
    critical = ((without_probability >= runtime.threshold) !=
                (full_probability >= runtime.threshold)[:, None])
    difference = np.asarray([
        [term["logit_contribution"] for term in decision["terms"]]
        for decision in decisions
    ])
    score_relevant = np.abs(difference) > 1e-10
    if np.any(critical & ~score_relevant):
        raise ValueError("class change without concept score effect")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS / f"botsim_seed{seed}.npz"
    np.savez_compressed(
        artifact, ids=np.asarray(ids), concept_names=np.asarray(evidence.names),
        full_probability=full_probability,
        without_probability=without_probability,
        logit_difference=difference,
        threshold=np.asarray(runtime.threshold))
    groups = {}
    for group, members in runtime.reasoner.groups.items():
        indexes = [evidence.names.index(name) for name in members]
        groups[group] = {
            "concept_count": len(indexes),
            "score_relevant_account_concept_pairs": int(np.count_nonzero(
                score_relevant[:, indexes])),
            "class_critical_account_concept_pairs": int(np.count_nonzero(
                critical[:, indexes])),
            "accounts_with_class_critical_concept": int(np.count_nonzero(
                np.any(critical[:, indexes], axis=1))),
        }
    return {
        "dataset": "botsim", "seed": seed, "test_accounts": len(ids),
        "concept_count": d, "withholding_queries": int(without_probability.size),
        "source_level": source_level,
        "max_raw_to_released_fact_difference": source_error,
        "saved_predictions_sha256": sha256_file(selected),
        "reference_version": runtime.reference.version,
        "max_frozen_probability_replay_error": replay_error,
        "max_additive_vs_direct_mask_error": direct_error,
        "score_relevant_account_concept_pairs": int(np.count_nonzero(score_relevant)),
        "class_critical_account_concept_pairs": int(np.count_nonzero(critical)),
        "accounts_with_any_score_relevant_concept": int(np.count_nonzero(
            np.any(score_relevant, axis=1))),
        "accounts_with_any_class_critical_concept": int(np.count_nonzero(
            np.any(critical, axis=1))),
        "source_groups": groups,
        "effect_matrix_path": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "effect_matrix_sha256": sha256_file(artifact),
    }


def main() -> None:
    rows = [run(seed) for seed in (0, 1, 2)]
    OUTPUT.write_text(json.dumps({
        "status": "frozen_best_full_test_exact_encoded_concept_withholding",
        "scope": "BotSim 12 selected source-linked concepts, each fixed test account and seed",
        "intervention": "zero one encoded concept in the frozen additive classifier; source records and graph stay fixed",
        "interpretation": "Exact frozen-model input dependence, not a natural behavioral intervention or independent semantic label.",
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print({key: row[key] for key in (
            "seed", "test_accounts", "withholding_queries",
            "score_relevant_account_concept_pairs",
            "class_critical_account_concept_pairs",
            "max_additive_vs_direct_mask_error")})
    print(OUTPUT)


if __name__ == "__main__":
    main()
