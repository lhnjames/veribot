"""Evaluate dev-locked literal-zero BotSim concepts on fixed paired test IDs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_upvote_zero_dev import (
    ACTIVITY, DATA, NATIVE, ROOT, SELECTED, USERS, literal_ratios)
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


FOLDER = ROOT / "output/v3_botsim_literal_upvote_zero_12"
DEV = FOLDER / "dev_only.json"
OUT = FOLDER / "fixed_test_report.json"
HYPBOT = ROOT / "output/hypbot_botsim_paired"


def main() -> None:
    locked = json.loads(DEV.read_text(encoding="utf-8"))
    if (locked["test_predictions_or_metrics_accessed"] is not False or
            locked["fixed_l2_from_prior_validation_lock"] != .01 or
            len(locked["runs"]) != 3 or
            any(run["validation"]["errors"] != 0
                for run in locked["runs"])):
        raise ValueError("literal-zero candidate was not dev-locked")
    runs = []
    for seed, plan in enumerate(locked["runs"]):
        if plan["seed"] != seed:
            raise ValueError("literal-zero seed lock differs")
        x, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        literal, _ = literal_ratios(ids)
        corrected = x.copy()
        corrected[:, names.index("avg_upvote_ratio")] = literal
        artifact_dir = FOLDER / f"seed{seed}"
        reference_path = artifact_dir / "reference.json.gz"
        reasoner_path = artifact_dir / "reasoner.json"
        if (sha256_file(reference_path) != plan["reference_artifact_sha256"]
                or sha256_file(reasoner_path) !=
                plan["reasoner_artifact_sha256"]
                or source["split_sha256"] != plan["split_sha256"]
                or sha256_file(ACTIVITY) != plan["activity_sha256"]):
            raise ValueError("dev-locked artifacts or raw source differ")
        reference = LabelBlindReference.load(reference_path)
        reasoner = MonotoneSplineReasoner.from_manifest(json.loads(
            reasoner_path.read_text(encoding="utf-8")))
        full = json.loads((ROOT / "output/v3_computable/botsim_native_graph" /
                           f"seed{seed}/report.json").read_text(encoding="utf-8"))
        atomic_groups = full["reasoner"]["groups"]
        if (reference.version != plan["reference_version"] or
                reasoner.reference_version !=
                collapse_group_tails(reference.transform(
                    corrected[split["test"]]),
                    atomic_groups)[0].reference_version):
            raise ValueError("literal-zero reference or concept differs")
        test = split["test"]
        concepts, _ = collapse_group_tails(reference.transform(
            corrected[test]), atomic_groups)
        probability = reasoner.predict_proba(concepts)
        basis = reasoner._design(concepts)
        logit = reasoner.intercept + basis @ reasoner.coef
        if np.max(np.abs(expit(logit) - probability)) > 1e-12:
            raise ValueError("literal-zero concepts do not reconstruct decision")
        threshold = float(plan["threshold"])
        metrics = evaluate(labels[test], probability, threshold)
        old_path = SELECTED / f"seed{seed}/test_predictions.json"
        hyp_path = HYPBOT / f"seed{seed}/test_predictions.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))["predictions"]
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))["predictions"]
        test_ids = [ids[int(i)] for i in test]
        if ([row["account_id"] for row in old] != test_ids or
                [row["account_id"] for row in hyp] != test_ids or
                [row["label"] for row in old] != labels[test].tolist() or
                [row["label"] for row in hyp] != labels[test].tolist()):
            raise ValueError("paired fixed test accounts or labels differ")
        old_threshold = json.loads((SELECTED / f"seed{seed}/report.json")
                                   .read_text(encoding="utf-8"))["manifest"]["threshold"]
        old_class = np.asarray([row["probability"] >= old_threshold
                                for row in old])
        hyp_threshold = float(json.loads(hyp_path.read_text(
            encoding="utf-8"))["manifest"]["threshold"])
        hyp_class = np.asarray([row["probability"] >= hyp_threshold
                                for row in hyp])
        corrected_class = probability >= threshold
        if metrics["acc"] <= .994 or metrics["acc"] <= np.mean(
                hyp_class == labels[test]):
            # Record an honest failure; do not promote this candidate.
            passed = False
        else:
            passed = True
        predictions = [{"account_id": uid, "label": int(label),
                        "probability": float(p), "prediction": int(pred)}
                       for uid, label, p, pred in zip(
                           test_ids, labels[test], probability,
                           corrected_class, strict=True)]
        pred_path = artifact_dir / "test_predictions.json"
        pred_path.write_text(json.dumps({
            "seed": seed, "dev_lock_sha256": sha256_file(DEV),
            "predictions": predictions}, indent=2) + "\n", encoding="utf-8")
        runs.append({
            "seed": seed,
            "test_accounts": len(test),
            "split_sha256": source["split_sha256"],
            "threshold": threshold,
            "validation": plan["validation"],
            "test": metrics,
            "hypbot_correct": int(np.count_nonzero(
                hyp_class == labels[test])),
            "old_model_correct": int(np.count_nonzero(
                old_class == labels[test])),
            "literal_model_correct": int(np.count_nonzero(
                corrected_class == labels[test])),
            "old_vs_literal_changed_classes": int(np.count_nonzero(
                old_class != corrected_class)),
            "literal_only_correct_vs_old": int(np.count_nonzero(
                (corrected_class == labels[test]) &
                (old_class != labels[test]))),
            "old_only_correct_vs_literal": int(np.count_nonzero(
                (old_class == labels[test]) &
                (corrected_class != labels[test]))),
            "strictly_above_hypbot_and_99_4_percent": passed,
            "test_predictions_sha256": sha256_file(pred_path),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
            "old_predictions_sha256": sha256_file(old_path),
        })
        print({"seed": seed, "test_acc": metrics["acc"],
               "hypbot_correct": runs[-1]["hypbot_correct"],
               "old_correct": runs[-1]["old_model_correct"],
               "new_correct": runs[-1]["literal_model_correct"],
               "passed": passed}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_literal_zero_twelve_concepts_fixed_test_after_dev_lock",
        "dev_lock_sha256": sha256_file(DEV),
        "all_seeds_strict_numeric_gate_pass": all(
            row["strictly_above_hypbot_and_99_4_percent"] for row in runs),
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
