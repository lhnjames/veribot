"""Replay frozen schema-v2 decisions from independently recomputed raw facts.

This audit does not fit, select, or alter a model. It distinguishes exact
model-input dependence from behavioral semantics or real-world causality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _source_groups
from evaluate_locked_schema_v2_candidates import _test_context
from audit_schema_v2_raw_fact_binding import (
    BOTSIM_ACTIVITY, BOTSIM_PRIOR, BOTSIM_RAW_NAMES, BOTSIM_USERS,
    FOX8_EVENTS, FOX8_PRIOR, FOX8_RAW, FOX8_RAW_NAMES,
    independently_recomputed_botsim, fox8_raw_truth,
)
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = ROOT / "output/evidence_schema_v2_group_additive_candidates/selection_lock.json"
FIXED = ROOT / "output/evidence_schema_v2_group_additive_candidates/fixed_test_report.json"
DEPENDENCE = ROOT / "output/evidence_schema_v2_group_additive_candidates/actual_decision_dependence.json"
BINDING = ROOT / "research/certificates/schema_v2_raw_fact_binding.json"
OUT = ROOT / "research/certificates/schema_v2_raw_to_frozen_decision.json"


def independently_encode(evidence, direction: dict, names: list[str]) -> np.ndarray:
    settings = direction["directions"]
    columns = []
    for j, name in enumerate(names):
        tail = settings[name]["tail"]
        if tail not in {"upper", "lower"}:
            raise ValueError(f"Invalid tail for {name}")
        raw = evidence.upper[:, j] if tail == "upper" else evidence.lower[:, j]
        columns.append(np.log1p(raw) * evidence.reliability[:, j])
    return np.column_stack(columns)


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    fixed = json.loads(FIXED.read_text(encoding="utf-8"))
    dependence = json.loads(DEPENDENCE.read_text(encoding="utf-8"))
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    if (lock["schema_sha256"] != sha256_file(SCHEMA)
            or fixed["selection_lock_sha256"] != sha256_file(LOCK)
            or dependence["selection_lock_sha256"] != sha256_file(LOCK)
            or binding["schema_sha256"] != sha256_file(SCHEMA)
            or binding["lock_sha256"] != sha256_file(LOCK)
            or binding["fox8_raw_sha256"] != sha256_file(FOX8_RAW)
            or binding["botsim_users_sha256"] != sha256_file(BOTSIM_USERS)
            or binding["botsim_activity_sha256"] != sha256_file(BOTSIM_ACTIVITY)
            or binding["fox8_prior_audit_sha256"] != sha256_file(FOX8_PRIOR)
            or binding["botsim_prior_audit_sha256"] != sha256_file(BOTSIM_PRIOR)):
        raise ValueError("Raw-source, schema, or frozen-model lock differs")

    fox_ids, fox_truth, fox_pairs = fox8_raw_truth()
    bot_ids, bot_truth, bot_native = independently_recomputed_botsim()
    if (fox_pairs != binding["fox8_native_event_pairs_recomputed"]
            or bot_native["verified_fact_cells"] !=
                binding["botsim_native_fact_cells_recomputed"]
            or sha256_file(FOX8_EVENTS) != json.loads(
                FOX8_PRIOR.read_text(encoding="utf-8"))["native_event_index_sha256"]):
        raise ValueError("Independent raw graph source differs")
    population = {
        "fox8": (fox_ids, fox_truth, list(FOX8_RAW_NAMES)),
        "botsim": (bot_ids, bot_truth, list(BOTSIM_RAW_NAMES)),
    }
    results = {}
    for dataset, (ids, raw_truth, raw_names) in population.items():
        names = [item["fact"] for item in schema["schema"][dataset]["concepts"]]
        groups, indices = _source_groups(names, schema["schema"][dataset])
        selected = _missing_means(dataset, raw_truth, raw_names, ids)
        selected = selected[:, [raw_names.index(name) for name in names]]
        runs = []
        for seed, frozen in enumerate(lock["runs"][dataset]):
            loaded, _, loaded_ids, loaded_names, split, manifest = _source_facts(
                dataset, seed)
            if (loaded_ids != ids or manifest["split_sha256"] != frozen["split_sha256"]
                    or manifest["files_sha256"] != frozen["source_files_sha256"]):
                raise ValueError(f"{dataset} seed {seed} source or split differs")
            loaded = _missing_means(dataset, loaded, loaded_names, loaded_ids)
            loaded = loaded[:, [loaded_names.index(name) for name in names]]
            if not np.array_equal(np.isfinite(loaded), np.isfinite(selected)):
                raise ValueError(f"{dataset} seed {seed} model missingness differs")
            max_fact_difference = float(np.nanmax(np.abs(loaded - selected)))
            if max_fact_difference > 1e-6:
                raise ValueError(f"{dataset} seed {seed} model fact differs")

            reference_path = ROOT / frozen["reference_path"]
            direction_path = ROOT / frozen["direction_path"]
            model_paths = [ROOT / item for item in frozen["model_paths"]]
            if (sha256_file(reference_path) != frozen["reference_sha256"]
                    or sha256_file(direction_path) != frozen["direction_sha256"]
                    or [sha256_file(item) for item in model_paths] !=
                        frozen["model_sha256"]):
                raise ValueError("Frozen model, directions or reference differ")
            reference = LabelBlindReference.load(reference_path)
            directions = json.loads(direction_path.read_text(encoding="utf-8"))
            if reference.version != frozen["reference_version"]:
                raise ValueError("Frozen reference version differs")
            test = split["test"]
            evidence = reference.transform(
                selected[test], _test_context(selected[test], names, frozen))
            logits = []
            zero_logits = []
            effects = []
            for path, direction in zip(model_paths, directions, strict=True):
                model = joblib.load(path)
                x = independently_encode(evidence, direction, names)
                raw_score = np.asarray(model.booster_.predict(x, raw_score=True))
                zero_score = np.asarray(model.booster_.predict(
                    np.zeros_like(x), raw_score=True))
                group_effects = []
                for members in indices:
                    without = x.copy()
                    without[:, members] = 0
                    group_effects.append(raw_score - np.asarray(
                        model.booster_.predict(without, raw_score=True)))
                logits.append(raw_score)
                zero_logits.append(zero_score)
                effects.append(np.column_stack(group_effects))
            score = np.mean(logits, axis=0)
            zero = np.mean(zero_logits, axis=0)
            effect = np.mean(effects, axis=0)
            additivity_error = float(np.max(np.abs(score-zero-effect.sum(axis=1))))

            tested = fixed["runs"][dataset][seed]
            path = ROOT / tested["prediction_path"]
            if sha256_file(path) != tested["prediction_sha256"]:
                raise ValueError("Saved fixed-test predictions differ")
            predictions = json.loads(path.read_text(encoding="utf-8"))["predictions"]
            probability = expit(score)
            saved_probability = np.asarray([row["probability"] for row in predictions])
            probability_error = float(np.max(np.abs(probability-saved_probability)))
            threshold = float(frozen["classification_threshold_selected_on_dev"])
            predicted = (probability >= threshold).astype(int)
            if (additivity_error > 1e-9 or probability_error > 1e-12
                    or [ids[int(i)] for i in test] !=
                        [row["account_id"] for row in predictions]
                    or predicted.tolist() != [row["class"] for row in predictions]):
                raise ValueError("Independent raw facts do not replay decision")
            matrix_path = ROOT / dependence["runs"][dataset][seed]["account_matrix"]
            if sha256_file(matrix_path) != dependence["runs"][dataset][seed][
                    "account_matrix_sha256"]:
                raise ValueError("Frozen decision matrix differs")
            with np.load(matrix_path, allow_pickle=False) as prior:
                if (prior["ids"].astype(str).tolist() !=
                        [ids[int(i)] for i in test]
                        or prior["group_names"].astype(str).tolist() != list(groups)):
                    raise ValueError("Decision matrix accounts or groups differ")
                score_error = float(np.max(np.abs(score-prior["logit"])))
                group_error = float(np.max(np.abs(effect-prior["group_effects"])))
                zero_error = float(np.max(np.abs(zero-prior["zero_logit"])))
            if max(score_error, group_error, zero_error) > 1e-9:
                raise ValueError("Independent raw source differs from saved group effects")
            decision_margin = score - logit(threshold)
            runs.append({
                "seed": seed,
                "test_accounts": len(test),
                "max_raw_fact_to_model_input_difference": max_fact_difference,
                "max_raw_replay_to_saved_probability_difference": probability_error,
                "max_raw_replay_to_saved_logit_difference": score_error,
                "max_raw_replay_to_saved_group_effect_difference": group_error,
                "max_group_additivity_error": additivity_error,
                "positive_decision_margins": int(np.count_nonzero(decision_margin >= 0)),
                "negative_decision_margins": int(np.count_nonzero(decision_margin < 0)),
                "source_group_names": list(groups),
                "saved_decision_matrix_sha256": sha256_file(matrix_path),
            })
            print(dataset, seed, len(test), probability_error,
                  score_error, group_error, flush=True)
        results[dataset] = runs
    report = {
        "status": "independent_raw_fact_to_frozen_candidate_score_replay_verified",
        "scope": "independent_source_values_to_current_single_tail_candidate_logits",
        "schema_sha256": sha256_file(SCHEMA),
        "lock_sha256": sha256_file(LOCK),
        "binding_certificate_sha256": sha256_file(BINDING),
        "fixed_test_report_sha256": sha256_file(FIXED),
        "actual_decision_dependence_sha256": sha256_file(DEPENDENCE),
        "behavioral_semantics_or_real_world_causality_proven": False,
        "natural_source_correction_benefit_proven": False,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
