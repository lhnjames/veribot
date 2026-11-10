"""Evaluate the frozen dual-tail neutral candidate on existing fixed tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _source_groups
from screen_schema_v2_dual_tail_neutral_dev import _context, _design
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
PARENT_LOCK = ROOT / "output/evidence_schema_v2_group_additive_candidates/selection_lock.json"
PARENT_TEST = ROOT / "output/evidence_schema_v2_group_additive_candidates/fixed_test_report.json"
LOCK = ROOT / "output/schema_v2_dual_tail_neutral_candidate/selection_lock.json"
OUT = ROOT / "output/schema_v2_dual_tail_neutral_candidate/fixed_test_report.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    prior = json.loads(PARENT_TEST.read_text(encoding="utf-8"))
    if (lock["status"] != "dual_tail_zero_anchor_candidate_locked_before_this_test_score"
            or lock["test_predictions_or_metrics_accessed"] is not False
            or lock["schema_sha256"] != sha256_file(SCHEMA)
            or lock["parent_lock_sha256"] != sha256_file(PARENT_LOCK)
            or prior["selection_lock_sha256"] != sha256_file(PARENT_LOCK)
            or lock["selected_anchor_ratio"] != 0.02):
        raise ValueError("Frozen selection protocol differs")
    results = {}
    for dataset in ("fox8", "botsim"):
        names = [row["fact"] for row in schema["schema"][dataset]["concepts"]]
        group_names, atomic_groups = _source_groups(
            names, schema["schema"][dataset])
        groups = [[2*j+k for j in members for k in (0, 1)]
                  for members in atomic_groups]
        dataset_runs = []
        for seed, frozen in enumerate(lock["runs"][dataset]):
            raw, labels, ids, fact_names, split, source = _source_facts(
                dataset, seed)
            if (source["split_sha256"] != frozen["split_sha256"]
                    or source["files_sha256"] != frozen["source_files_sha256"]
                    or names != frozen["selected_fact_names"]
                    or frozen["source_groups"] !=
                        dict(zip(group_names, groups, strict=True))):
                raise ValueError("Source or schema differs from frozen model")
            raw = _missing_means(dataset, raw, fact_names, ids)
            facts = raw[:, [fact_names.index(name) for name in names]]
            reference_path = ROOT / frozen["reference_path"]
            signs_path = ROOT / frozen["signs_path"]
            model_paths = [ROOT / item for item in frozen["model_paths"]]
            if (sha256_file(reference_path) != frozen["reference_sha256"]
                    or sha256_file(signs_path) != frozen["signs_sha256"]
                    or [sha256_file(item) for item in model_paths] !=
                        frozen["model_sha256"]):
                raise ValueError("Frozen artifacts differ")
            reference = LabelBlindReference.load(reference_path)
            if reference.version != frozen["reference_version"]:
                raise ValueError("Reference version differs")
            signs = json.loads(signs_path.read_text(encoding="utf-8"))
            test = split["test"]
            context = _context(facts, names, frozen)
            evidence = reference.transform(
                facts[test], None if context is None else context[test])
            x = _design(evidence)
            scores = []
            zeros = []
            fold_effects = []
            fold_without = []
            for path, direction in zip(model_paths, signs, strict=True):
                model = joblib.load(path)
                if (len(direction) != x.shape[1]
                        or not all_trees_one_group(model, groups)[0]):
                    raise ValueError("Saved model direction/group constraint differs")
                full = np.asarray(model.booster_.predict(x, raw_score=True))
                zero = float(model.booster_.predict(
                    np.zeros((1, x.shape[1])), raw_score=True)[0])
                if abs(zero - frozen["zero_input_logit_by_fold"][len(zeros)]) > 1e-12:
                    raise ValueError("Frozen mathematical origin differs")
                effects = []
                withouts = []
                for members in groups:
                    masked = x.copy()
                    masked[:, members] = 0
                    without = np.asarray(model.booster_.predict(
                        masked, raw_score=True))
                    effects.append(full - without)
                    withouts.append(without - zero)
                scores.append(full-zero)
                zeros.append(zero)
                fold_effects.append(np.column_stack(effects))
                fold_without.append(np.column_stack(withouts))
            score = np.mean(scores, axis=0)
            group_effects = np.mean(fold_effects, axis=0)
            without = np.mean(fold_without, axis=0)
            additivity = float(np.max(np.abs(score-group_effects.sum(axis=1))))
            probability = expit(score)
            threshold = float(frozen["decision_probability_threshold"])
            if not np.array_equal(probability >= threshold, score > 0):
                raise ValueError("Neutral decision convention differs")
            prediction = (score > 0).astype(int)
            metric = evaluate(labels[test], probability, threshold)
            previous = prior["runs"][dataset][seed]
            if previous["split_sha256"] != source["split_sha256"]:
                raise ValueError("Paired parent test population differs")
            if additivity > 1e-9:
                raise ValueError("Source-group effects do not reconstruct score")
            predicted_bot = int(np.count_nonzero(score > 0))
            predicted_human = int(np.count_nonzero(score < 0))
            tie_count = int(np.count_nonzero(score == 0))
            if predicted_bot + predicted_human + tie_count != len(test):
                raise ValueError("Prediction categories do not partition test")
            pivotal = (without > 0) != (score[:, None] > 0)
            prediction_path = LOCK.parent / dataset / f"seed{seed}" / "test_predictions.json"
            prediction_path.write_text(json.dumps({
                "dataset": dataset,
                "seed": seed,
                "selection_lock_sha256": sha256_file(LOCK),
                "split_sha256": source["split_sha256"],
                "predictions": [
                    {"account_id": ids[int(i)], "label": int(labels[int(i)]),
                     "score_relative_to_mathematical_zero": float(s),
                     "probability": float(p), "class": int(c)}
                    for i, s, p, c in zip(test, score, probability,
                                           prediction, strict=True)],
            }, indent=2) + "\n", encoding="utf-8")
            matrix_path = prediction_path.with_name("group_effects.npz")
            np.savez_compressed(
                matrix_path, ids=np.asarray([ids[int(i)] for i in test]),
                group_names=np.asarray(list(group_names)), score=score,
                group_effects=group_effects, without_group_scores=without,
                predicted_class=prediction,
                zero_input_logit_by_fold=np.asarray(zeros))
            dataset_runs.append({
                "seed": seed,
                "split_sha256": source["split_sha256"],
                "test_accounts": len(test),
                "candidate_correct": len(test)-metric["errors"],
                "candidate_metric": metric,
                "parent_single_tail_correct": previous["candidate_correct"],
                "paired_hypbot_correct": previous["hypbot_correct"],
                "bot_predictions": predicted_bot,
                "human_predictions": predicted_human,
                "zero_score_ties": tie_count,
                "max_group_additivity_error": additivity,
                "any_group_pivotal_accounts": int(np.count_nonzero(
                    np.any(pivotal, axis=1))),
                "group_pivotal_accounts": {name: int(np.count_nonzero(pivotal[:, j]))
                                           for j, name in enumerate(group_names)},
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "prediction_sha256": sha256_file(prediction_path),
                "group_matrix_path": str(matrix_path.relative_to(ROOT)),
                "group_matrix_sha256": sha256_file(matrix_path),
            })
            print(dataset, seed, len(test)-metric["errors"],
                  "parent", previous["candidate_correct"],
                  "hypbot", previous["hypbot_correct"],
                  "ties", tie_count, flush=True)
        results[dataset] = dataset_runs
    OUT.write_text(json.dumps({
        "scope": "dev_selected_dual_tail_neutral_candidate_existing_fixed_test",
        "selection_lock_sha256": sha256_file(LOCK),
        "parent_test_report_sha256": sha256_file(PARENT_TEST),
        "test_population_previously_explored_by_other_candidates": True,
        "zero_vector_is_observed_account": False,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
