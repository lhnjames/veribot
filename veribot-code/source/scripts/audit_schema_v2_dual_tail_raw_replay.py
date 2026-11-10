"""Replay frozen dual-tail decisions from independently recomputed sources."""

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
from screen_schema_v2_dual_tail_neutral_dev import _context, _design
from audit_schema_v2_raw_fact_binding import (
    BOTSIM_RAW_NAMES, FOX8_RAW_NAMES,
    independently_recomputed_botsim, fox8_raw_truth,
)
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = ROOT / "output/schema_v2_dual_tail_neutral_candidate/selection_lock.json"
TEST = ROOT / "output/schema_v2_dual_tail_neutral_candidate/fixed_test_report.json"
BINDING = ROOT / "research/certificates/schema_v2_raw_fact_binding.json"
OUT = ROOT / "research/certificates/schema_v2_dual_tail_raw_replay.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    test_report = json.loads(TEST.read_text(encoding="utf-8"))
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    if (lock["schema_sha256"] != sha256_file(SCHEMA)
            or test_report["selection_lock_sha256"] != sha256_file(LOCK)
            or binding["schema_sha256"] != sha256_file(SCHEMA)):
        raise ValueError("Frozen schema, results or raw-source binding differs")
    fox_ids, fox_truth, _ = fox8_raw_truth()
    bot_ids, bot_truth, _ = independently_recomputed_botsim()
    source_populations = {
        "fox8": (fox_ids, fox_truth, list(FOX8_RAW_NAMES)),
        "botsim": (bot_ids, bot_truth, list(BOTSIM_RAW_NAMES)),
    }
    runs = {}
    for dataset, (ids, raw_facts, raw_names) in source_populations.items():
        names = [item["fact"] for item in schema["schema"][dataset]["concepts"]]
        named_groups, atomic_groups = _source_groups(
            names, schema["schema"][dataset])
        groups = [[2*j+k for j in members for k in (0, 1)]
                  for members in atomic_groups]
        selected = _missing_means(dataset, raw_facts, raw_names, ids)
        selected = selected[:, [raw_names.index(name) for name in names]]
        dataset_runs = []
        for seed, frozen in enumerate(lock["runs"][dataset]):
            loaded, _, loaded_ids, loaded_names, split, manifest = _source_facts(
                dataset, seed)
            if (loaded_ids != ids or manifest["split_sha256"] != frozen["split_sha256"]
                    or manifest["files_sha256"] != frozen["source_files_sha256"]):
                raise ValueError("Raw population or split differs")
            loaded = _missing_means(dataset, loaded, loaded_names, ids)
            loaded = loaded[:, [loaded_names.index(name) for name in names]]
            if (not np.array_equal(np.isfinite(loaded), np.isfinite(selected))
                    or float(np.nanmax(np.abs(loaded-selected))) > 1e-6):
                raise ValueError("Independent raw facts differ from model input")
            model_paths = [ROOT / item for item in frozen["model_paths"]]
            reference_path = ROOT / frozen["reference_path"]
            if (sha256_file(reference_path) != frozen["reference_sha256"]
                    or [sha256_file(item) for item in model_paths] !=
                        frozen["model_sha256"]):
                raise ValueError("Frozen artifacts differ")
            reference = LabelBlindReference.load(reference_path)
            observed_test = split["test"]
            context = _context(selected, names, frozen)
            evidence = reference.transform(
                selected[observed_test],
                None if context is None else context[observed_test])
            x = _design(evidence)
            fold_scores = []
            fold_effects = []
            fold_without = []
            for index, path in enumerate(model_paths):
                model = joblib.load(path)
                full = np.asarray(model.booster_.predict(x, raw_score=True))
                zero = float(model.booster_.predict(
                    np.zeros((1, x.shape[1])), raw_score=True)[0])
                if abs(zero-frozen["zero_input_logit_by_fold"][index]) > 1e-12:
                    raise ValueError("Origin score differs")
                effects = []
                withouts = []
                for members in groups:
                    masked = x.copy()
                    masked[:, members] = 0
                    without = np.asarray(model.booster_.predict(
                        masked, raw_score=True))
                    effects.append(full-without)
                    withouts.append(without-zero)
                fold_scores.append(full-zero)
                fold_effects.append(np.column_stack(effects))
                fold_without.append(np.column_stack(withouts))
            score = np.mean(fold_scores, axis=0)
            effects = np.mean(fold_effects, axis=0)
            without = np.mean(fold_without, axis=0)
            predictions_path = ROOT / test_report["results"][dataset][seed]["prediction_path"]
            matrix_path = ROOT / test_report["results"][dataset][seed]["group_matrix_path"]
            if (sha256_file(predictions_path) !=
                    test_report["results"][dataset][seed]["prediction_sha256"]
                    or sha256_file(matrix_path) !=
                    test_report["results"][dataset][seed]["group_matrix_sha256"]):
                raise ValueError("Saved test records differ")
            saved = json.loads(predictions_path.read_text(
                encoding="utf-8"))["predictions"]
            with np.load(matrix_path, allow_pickle=False) as matrix:
                if (matrix["ids"].astype(str).tolist() !=
                        [ids[int(i)] for i in observed_test]
                        or matrix["group_names"].astype(str).tolist() !=
                        list(named_groups)):
                    raise ValueError("Saved account/group ordering differs")
                score_error = float(np.max(np.abs(score-matrix["score"])))
                group_error = float(np.max(np.abs(effects-matrix["group_effects"])))
                without_error = float(np.max(np.abs(
                    without-matrix["without_group_scores"])))
            probability_error = float(np.max(np.abs(
                expit(score)-np.asarray([row["probability"] for row in saved]))))
            classes = (score > 0).astype(int).tolist()
            additivity = float(np.max(np.abs(score-effects.sum(axis=1))))
            if (max(score_error, group_error, without_error,
                    probability_error) > 1e-12
                    or additivity > 1e-9
                    or classes != [row["class"] for row in saved]):
                raise ValueError("Independent raw sources do not replay decision")
            dataset_runs.append({
                "seed": seed, "accounts": len(observed_test),
                "max_score_difference": score_error,
                "max_probability_difference": probability_error,
                "max_group_effect_difference": group_error,
                "max_group_mask_difference": without_error,
                "max_additivity_error": additivity,
                "predicted_bot": int(np.count_nonzero(score > 0)),
                "predicted_human": int(np.count_nonzero(score < 0)),
            })
            print(dataset, seed, score_error, group_error, flush=True)
        runs[dataset] = dataset_runs
    report = {
        "status": "dual_tail_frozen_decisions_replayed_from_independent_raw_facts",
        "schema_sha256": sha256_file(SCHEMA),
        "selection_lock_sha256": sha256_file(LOCK),
        "fixed_test_report_sha256": sha256_file(TEST),
        "raw_binding_certificate_sha256": sha256_file(BINDING),
        "natural_source_correction_or_behavioral_semantics_proven": False,
        "results": runs,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
