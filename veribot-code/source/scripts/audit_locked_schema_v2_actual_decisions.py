"""Audit exact source-group score effects of frozen Fox8/BotSim candidates.

This audit reads frozen models and saved predictions. It does not select or
refit any model, and it does not use test labels to choose explanations.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _source_groups
from lock_group_additive_schema_v2_candidates import _design, OUT as LOCK_DIR
from evaluate_locked_schema_v2_candidates import _test_context
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = LOCK_DIR / "selection_lock.json"
TEST = LOCK_DIR / "fixed_test_report.json"
OUT = LOCK_DIR / "actual_decision_dependence.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    fixed = json.loads(TEST.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA) or \
            fixed["selection_lock_sha256"] != sha256_file(LOCK):
        raise ValueError("schema or frozen model lock differs")
    summary = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [item["fact"] for item in spec["concepts"]]
        group_names, group_indices = _source_groups(names, spec)
        runs = []
        for seed in (0, 1, 2):
            frozen = lock["runs"][dataset][seed]
            tested = fixed["runs"][dataset][seed]
            raw, _, ids, fact_names, split, source = _source_facts(dataset, seed)
            if source["split_sha256"] != frozen["split_sha256"] or \
                    source["files_sha256"] != frozen["source_files_sha256"] or \
                    frozen["selected_fact_names"] != names:
                raise ValueError("test population or source facts differ")
            raw = _missing_means(dataset, raw, fact_names, ids)
            test = split["test"]
            facts = raw[:, [fact_names.index(name) for name in names]]
            reference_path = ROOT / frozen["reference_path"]
            model_paths = [ROOT / item for item in frozen["model_paths"]]
            direction_path = ROOT / frozen["direction_path"]
            if sha256_file(reference_path) != frozen["reference_sha256"] or \
                    [sha256_file(item) for item in model_paths] != frozen["model_sha256"] or \
                    sha256_file(direction_path) != frozen["direction_sha256"]:
                raise ValueError("frozen artifacts differ")
            reference = LabelBlindReference.load(reference_path)
            directions = json.loads(direction_path.read_text(encoding="utf-8"))
            evidence = reference.transform(
                facts[test], _test_context(facts[test], names, frozen))
            fold_logits = []
            fold_zeros = []
            fold_effects = []
            fold_without = []
            for path, direction in zip(model_paths, directions, strict=True):
                model = joblib.load(path)
                x = _design(evidence, direction, names)
                full = np.asarray(model.booster_.predict(x, raw_score=True))
                zero = np.asarray(model.booster_.predict(np.zeros_like(x), raw_score=True))
                effects = []
                withouts = []
                for group in group_indices:
                    masked = x.copy()
                    masked[:, group] = 0.0
                    without = np.asarray(model.booster_.predict(masked, raw_score=True))
                    effects.append(full - without)
                    withouts.append(without)
                fold_logits.append(full)
                fold_zeros.append(zero)
                fold_effects.append(np.column_stack(effects))
                fold_without.append(np.column_stack(withouts))
            logit = np.mean(fold_logits, axis=0)
            zero = np.mean(fold_zeros, axis=0)
            effects = np.mean(fold_effects, axis=0)
            without = np.mean(fold_without, axis=0)
            error = float(np.max(np.abs(logit - zero - np.sum(effects, axis=1))))
            if error > 1e-9:
                raise ValueError("group effects do not reconstruct frozen score")
            predictions = json.loads((ROOT / tested["prediction_path"]).read_text(
                encoding="utf-8"))["predictions"]
            probability = expit(logit)
            pred_error = float(np.max(np.abs(probability - np.asarray(
                [row["probability"] for row in predictions]))))
            threshold = float(frozen["classification_threshold_selected_on_dev"])
            pred = probability >= threshold
            if pred_error > 1e-12 or [int(x) for x in pred] != \
                    [row["class"] for row in predictions] or \
                    [ids[int(i)] for i in test] != [row["account_id"] for row in predictions]:
                raise ValueError("actual decision audit does not replay saved predictions")
            zero_pred = expit(zero) >= threshold
            masked_pred = expit(without) >= threshold
            pivotal = masked_pred != pred[:, None]
            toward_prediction = np.where(pred[:, None], effects > 1e-12,
                                         effects < -1e-12)
            against_prediction = np.where(pred[:, None], effects < -1e-12,
                                          effects > 1e-12)
            output_path = LOCK_DIR / dataset / f"seed{seed}" / "actual_decision_dependence.npz"
            np.savez_compressed(
                output_path, ids=np.asarray([ids[int(i)] for i in test]),
                group_names=np.asarray(list(group_names)), logit=logit,
                zero_logit=zero, group_effects=effects,
                without_group_logits=without, predicted_class=pred.astype(np.int8),
                threshold=np.asarray(threshold))
            runs.append({
                "seed": seed, "accounts": len(test),
                "max_score_additivity_error": error,
                "max_saved_probability_error": pred_error,
                "full_vs_zero_class_changes": int(np.count_nonzero(pred != zero_pred)),
                "default_class_account_count": int(np.count_nonzero(pred == zero_pred)),
                "any_single_group_pivotal": int(np.count_nonzero(np.any(pivotal, axis=1))),
                "group_effects": {name: {
                    "pivotal_accounts": int(np.count_nonzero(pivotal[:, j])),
                    "supports_predicted_class": int(np.count_nonzero(toward_prediction[:, j])),
                    "opposes_predicted_class": int(np.count_nonzero(against_prediction[:, j])),
                } for j, name in enumerate(group_names)},
                "account_matrix": str(output_path.relative_to(ROOT)),
                "account_matrix_sha256": sha256_file(output_path),
            })
            print({"dataset": dataset, "seed": seed, "additivity_error": error,
                   "default_class_count": runs[-1]["default_class_account_count"]},
                  flush=True)
        summary[dataset] = runs
    OUT.write_text(json.dumps({
        "scope": "exact_input_dependence_of_frozen_evidence_classifiers",
        "source_level_behavior_causality_or_semantic_truth_claimed": False,
        "selection_lock_sha256": sha256_file(LOCK),
        "fixed_test_report_sha256": sha256_file(TEST),
        "runs": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
