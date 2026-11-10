"""Count which corrected BotSim source concepts actually change decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimLiteralZeroRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_botsim_literal_upvote_zero_12"
OUT = ROOT / "research/certificates/botsim_literal_zero_decision_dependency.json"


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimLiteralZeroRuntime(ROOT, seed)
        path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(path.read_text(encoding="utf-8"))["predictions"]
        ids = [row["account_id"] for row in saved]
        decisions = runtime.predict_many(ids)
        labels = np.asarray([row["label"] for row in saved], dtype=int)
        logit = np.asarray([row["logit"] for row in decisions], dtype=float)
        threshold = runtime.threshold
        baseline = np.asarray([row["prediction"] for row in decisions], dtype=int)
        if (baseline.tolist() != [row["prediction"] for row in saved] or
                max(abs(row["probability"] - expected["probability"])
                    for row, expected in zip(decisions, saved, strict=True)) > 1e-12):
            raise ValueError("source-bound decisions differ from fixed test")
        contribution = np.asarray([[term["logit_contribution"] for term in row["terms"]]
                                   for row in decisions], dtype=float)
        if np.max(np.abs(logit - (runtime.reasoner.intercept +
                                 contribution.sum(axis=1)))) > 1e-9:
            raise ValueError("all source concepts fail to reconstruct model")
        empty_class = np.asarray([int(expit(runtime.reasoner.intercept) >= threshold)]
                                 * len(ids), dtype=int)
        concept_rows = []
        for j, name in enumerate(runtime.reasoner.names):
            without = (expit(logit - contribution[:, j]) >= threshold).astype(int)
            flips = int(np.count_nonzero(without != baseline))
            concept_rows.append({
                "name": name,
                "nonzero_effect_accounts": int(np.count_nonzero(
                    np.abs(contribution[:, j]) > 1e-10)),
                "max_absolute_logit_effect": float(np.max(np.abs(contribution[:, j]))),
                "class_changes_when_removed": flips,
                "correct_to_wrong_when_removed": int(np.count_nonzero(
                    (baseline == labels) & (without != labels))),
                "wrong_to_correct_when_removed": int(np.count_nonzero(
                    (baseline != labels) & (without == labels))),
            })
        runs.append({
            "seed": seed, "test_accounts": len(ids),
            "clean_correct": int(np.count_nonzero(baseline == labels)),
            "no_evidence_correct": int(np.count_nonzero(empty_class == labels)),
            "class_changes_without_all_evidence": int(np.count_nonzero(
                baseline != empty_class)),
            "accounts_with_at_least_one_pivotal_concept": int(np.count_nonzero(
                np.any(np.column_stack([
                    (expit(logit - contribution[:, j]) >= threshold).astype(int)
                    != baseline for j in range(contribution.shape[1])]), axis=1))),
            "concepts": concept_rows,
            "saved_predictions_sha256": sha256_file(path),
        })
        print({"seed": seed,
               "clean_correct": runs[-1]["clean_correct"],
               "no_evidence_correct": runs[-1]["no_evidence_correct"],
               "all_removed_class_changes": runs[-1]["class_changes_without_all_evidence"],
               "some_single_concept_pivotal": runs[-1]["accounts_with_at_least_one_pivotal_concept"]},
              flush=True)
    OUT.write_text(json.dumps({
        "status": "literal_zero_source_concepts_exact_frozen_decision_dependency",
        "operation": "set frozen concept logit effect to zero without refitting",
        "source_to_decision_audit_sha256": sha256_file(
            ROOT / "research/certificates/botsim_literal_zero_source_to_decision.json"),
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
