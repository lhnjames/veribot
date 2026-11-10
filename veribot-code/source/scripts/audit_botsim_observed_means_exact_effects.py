"""Recompute every exact concept mask for the selected BotSim observed-means model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/botsim_observed_means_exact_effects.json"
MATRICES = ROOT / "output/v3_best_exact_fact_effects"
SAVED = ROOT / "output/v3_botsim_observed_means_12"


def run(seed: int) -> dict:
    path = SAVED / f"seed{seed}/test_predictions.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    ids = [str(row["account_id"]) for row in recorded]
    if len(ids) != len(set(ids)):
        raise ValueError("BotSim test accounts repeat")
    runtime = BotSimObservedMeanRuntime(ROOT, seed)
    decisions = runtime.predict_many(ids)
    rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
    atomic = runtime.reference.transform(runtime.facts[rows])
    concepts, _ = collapse_group_tails(atomic, runtime.atomic_groups)
    if (len(concepts.names) != 12 or concepts.names != runtime.reasoner.names or
            len(decisions) != len(ids)):
        raise ValueError("observed-means concept population differs")
    logit = np.asarray([row["logit"] for row in decisions], dtype=float)
    full = np.asarray([row["probability"] for row in decisions], dtype=float)
    saved = np.asarray([row["probability"] for row in recorded], dtype=float)
    if np.max(np.abs(full-saved)) > 1e-12:
        raise ValueError("selected BotSim decision differs from frozen test")
    contribution = np.asarray([
        [term["logit_contribution"] for term in row["terms"]]
        for row in decisions], dtype=float)
    if (contribution.shape != (len(ids), 12) or
            np.max(np.abs(logit - runtime.reasoner.intercept -
                          contribution.sum(axis=1))) > 1e-9):
        raise ValueError("concept effects do not reconstruct observed model")
    without = expit(logit[:, None] - contribution)
    direct_error = 0.0
    for j in range(12):
        upper = concepts.upper.copy()
        lower = concepts.lower.copy()
        upper[:, j] = 0.0
        lower[:, j] = 0.0
        masked = EvidenceBatch(upper, lower, concepts.reliability,
                               concepts.available, concepts.reference_version,
                               concepts.names)
        direct = runtime.reasoner.predict_proba(masked)
        direct_error = max(direct_error,
                           float(np.max(np.abs(direct-without[:, j]))))
    if direct_error > 1e-12:
        raise ValueError("additive subtraction differs from direct concept masking")
    matrix_path = MATRICES / f"botsim_observed_seed{seed}.npz"
    np.savez_compressed(matrix_path,
                        ids=np.asarray(ids),
                        concept_names=np.asarray(concepts.names),
                        full_probability=full,
                        without_probability=without,
                        logit_difference=contribution,
                        threshold=np.asarray(runtime.threshold))
    critical = (without >= runtime.threshold) != (
        full >= runtime.threshold)[:, None]
    relevant = np.abs(contribution) > 1e-10
    return {
        "dataset": "botsim", "seed": seed,
        "test_accounts": len(ids), "concept_count": 12,
        "withholding_queries": int(without.size),
        "saved_predictions_sha256": sha256_file(path),
        "reference_version": runtime.reference.version,
        "model_source_binding": runtime.source_binding,
        "max_probability_replay_error": float(np.max(np.abs(full-saved))),
        "max_additive_vs_direct_mask_error": direct_error,
        "score_relevant_account_concept_pairs": int(np.count_nonzero(relevant)),
        "class_critical_account_concept_pairs": int(np.count_nonzero(critical)),
        "accounts_with_any_class_critical_concept": int(np.count_nonzero(
            np.any(critical, axis=1))),
        "effect_matrix_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "effect_matrix_sha256": sha256_file(matrix_path),
    }


def main() -> None:
    MATRICES.mkdir(parents=True, exist_ok=True)
    runs = [run(seed) for seed in (0, 1, 2)]
    OUT.write_text(json.dumps({
        "status": "selected_observed_means_bot_sim_full_exact_concept_masks",
        "scope": "all_1311_fixed_test_decisions_twelve_concepts_each",
        "source_to_decision_audit_sha256": sha256_file(ROOT /
            "research/certificates/botsim_observed_means_source_to_decision.json"),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    for row in runs:
        print({key: row[key] for key in (
            "seed", "test_accounts", "withholding_queries",
            "class_critical_account_concept_pairs",
            "max_additive_vs_direct_mask_error")}, flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
