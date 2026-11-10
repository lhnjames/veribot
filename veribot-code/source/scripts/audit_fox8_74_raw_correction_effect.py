"""Compare frozen Fox8 74-fact decisions with independently parsed raw records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_twenty_raw_to_decision import EVENTS, RAW, source_population
from gate0_shallow_tree import evaluate, sha256_file
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.high_performance_fact_effects import evidence_design


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity"
OUT = ROOT / "research/certificates/fox8_74_raw_precision_and_decision_audit.json"


def encode(runtime: Fox8NeighborAtomicRuntime, facts: np.ndarray,
           indices: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts[indices])
    reliability = batch.reliability.copy()
    reliability[:, 43:] *= runtime.supports[indices] / (runtime.supports[indices] + 5.0)
    matrix = np.column_stack((np.log1p(batch.upper) * reliability,
                              np.log1p(batch.lower) * reliability))
    if not np.isfinite(matrix).all() or matrix.shape != (len(indices), 148):
        raise ValueError("corrected Fox8 evidence matrix differs")
    return matrix


def main() -> None:
    ids, raw74, event_pairs = source_population()
    if raw74.shape != (2279, 74):
        raise ValueError("raw Fox8 source population differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        if runtime.ids != ids or not np.array_equal(
                np.isfinite(runtime.facts), np.isfinite(raw74)):
            raise ValueError("raw and frozen fact availability differs")
        delta = np.abs(runtime.facts - raw74)
        finite = np.isfinite(delta)
        magnitude = np.maximum(np.abs(runtime.facts), np.abs(raw74))
        safe_magnitude = np.where(finite, magnitude, 0.0)
        ulp = np.abs(np.spacing(safe_magnitude.astype(np.float32)).astype(float))
        ulp = np.maximum(ulp, np.finfo(np.float32).tiny)
        ratio = np.where(finite, delta, 0.0) / ulp
        differing = []
        for j, name in enumerate(runtime.reference.names):
            if np.any(delta[:, j] > 1e-6):
                differing.append({
                    "fact": name,
                    "accounts_differing_gt_1e_6": int(np.count_nonzero(delta[:, j] > 1e-6)),
                    "max_absolute_difference": float(np.nanmax(delta[:, j])),
                    "max_float32_ulp_ratio": float(np.max(ratio[:, j])),
                    "accounts_differing_gt_one_ulp": int(np.count_nonzero(ratio[:, j] > 1.000001)),
                })
        saved_path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        test_ids = [row["account_id"] for row in saved]
        indices = np.asarray([runtime.index[uid] for uid in test_ids])
        y = np.asarray([row["label"] for row in saved], dtype=int)
        original_matrix = evidence_design(runtime, indices)
        raw_matrix = encode(runtime, raw74, indices)
        original_probability = runtime.model.predict_proba(original_matrix)[:, 1]
        raw_probability = runtime.model.predict_proba(raw_matrix)[:, 1]
        recorded_probability = np.asarray([row["probability"] for row in saved])
        if np.max(np.abs(original_probability - recorded_probability)) > 1e-12:
            raise ValueError("frozen prediction does not replay")
        original_class = original_probability >= runtime.threshold
        raw_class = raw_probability >= runtime.threshold
        runs.append({
            "seed": seed, "test_accounts": len(indices),
            "fact_cells_checked": int(runtime.facts.size),
            "max_absolute_fact_difference": float(np.nanmax(delta)),
            "max_float32_ulp_ratio": float(np.max(ratio)),
            "facts_differing_gt_1e_6": differing,
            "test_accounts_with_any_encoded_evidence_change": int(np.count_nonzero(
                np.any(original_matrix != raw_matrix, axis=1))),
            "test_accounts_with_probability_change_gt_1e_12": int(np.count_nonzero(
                np.abs(original_probability - raw_probability) > 1e-12)),
            "max_probability_change": float(np.max(np.abs(
                original_probability - raw_probability))),
            "test_class_changes": int(np.count_nonzero(original_class != raw_class)),
            "original_test": evaluate(y, original_probability, runtime.threshold),
            "raw_recomputed_test": evaluate(y, raw_probability, runtime.threshold),
            "test_predictions_sha256": sha256_file(saved_path),
            "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
        })
        print({"seed": seed, "max_ulp": runs[-1]["max_float32_ulp_ratio"],
               "changed_probabilities": runs[-1]["test_accounts_with_probability_change_gt_1e_12"],
               "changed_classes": runs[-1]["test_class_changes"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "fox8_all_74_frozen_facts_vs_independent_raw_parser",
        "raw_archive_sha256": sha256_file(RAW),
        "native_event_index_sha256": sha256_file(EVENTS),
        "typed_internal_event_pairs": event_pairs,
        "raw_correction_is_training_population_edit": False,
        "frozen_model_used_without_refit": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
