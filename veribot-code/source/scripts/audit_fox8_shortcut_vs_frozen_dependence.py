"""Separate Fox8 single-field predictive capacity from actual frozen use."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.high_performance_fact_effects import evidence_design


ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "research/certificates/fox8_single_source_shortcut_controls.json"
EXACT = ROOT / "research/certificates/high_performance_exact_fact_effects.json"
GROUP = ROOT / "research/certificates/best_source_group_effects.json"
OUTPUT = ROOT / "research/certificates/fox8_shortcut_vs_frozen_dependence.json"
FIELD = "native_interaction_neighbor_mean_unique_lang_ratio"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict:
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    exact = json.loads(EXACT.read_text(encoding="utf-8"))
    group = json.loads(GROUP.read_text(encoding="utf-8"))
    rows = []
    for seed in (0, 1, 2):
        d = next(r for r in dev["runs"] if r["seed"] == seed)
        e = next(r for r in exact["runs"]
                 if r["dataset"] == "fox8" and r["seed"] == seed)
        g = next(r for r in group["runs"]
                 if r["dataset"] == "fox8" and r["seed"] == seed)
        matrix_path = ROOT / e["effect_matrix_path"]
        group_path = Path(g["group_effect_path"])
        if (sha256(matrix_path) != e["effect_matrix_sha256"] or
                sha256(group_path) != g["group_effect_sha256"] or
                d["model_sha256"] != e["model_sha256"] or
                d["split_sha256"] != e["source_split_sha256"]):
            raise ValueError("control and frozen effect certificates differ")
        with np.load(matrix_path, allow_pickle=False) as z:
            ids = z["ids"].astype(str).tolist()
            names = z["fact_names"].astype(str).tolist()
            full = z["full_probability"].astype(float)
            without = z["without_probability"].astype(float)
            threshold = float(z["threshold"])
        with np.load(group_path, allow_pickle=False) as z:
            group_names = z["group_names"].astype(str).tolist()
            group_without = z["without_group_probability"].astype(float)
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        if (names != list(runtime.reference.names) or
                len(ids) != 343 or threshold != runtime.threshold):
            raise ValueError("frozen Fox8 population, vocabulary or threshold changed")
        j = names.index(FIELD)
        rows_index = np.asarray([runtime.index[uid] for uid in ids])
        design = evidence_design(runtime, rows_index)
        original = expit(np.asarray(runtime.model.booster_.predict(
            design, raw_score=True), dtype=float))
        single_masked = design.copy()
        single_masked[:, j] = 0
        single_masked[:, j+74] = 0
        single = expit(np.asarray(runtime.model.booster_.predict(
            single_masked, raw_score=True), dtype=float))
        native_masked = design.copy()
        native_masked[:, 31:43] = 0
        native_masked[:, 105:117] = 0
        native = expit(np.asarray(runtime.model.booster_.predict(
            native_masked, raw_score=True), dtype=float))
        native_col = group_names.index("native_interaction")
        max_replay = float(max(np.max(np.abs(original-full)),
                               np.max(np.abs(single-without[:, j])),
                               np.max(np.abs(native-group_without[:, native_col]))))
        if max_replay > 1e-12:
            raise ValueError("saved single/group effect differs from frozen model")
        row = {
            "seed": seed, "dev_accounts": d["dev_accounts"],
            "fixed_test_judgments": len(ids),
            "single_stump_dev_errors_neighbor_language": d["field_controls"][FIELD]["dev_errors"],
            "single_stump_dev_errors_own_language": d["field_controls"]["unique_lang_ratio"]["dev_errors"],
            "single_stump_dev_errors_neighbor_degree": d["field_controls"]["native_neighbor_degree"]["dev_errors"],
            "single_stump_dev_errors_neighbor_defined": d["field_controls"]["neighbor_language_defined_indicator"]["dev_errors"],
            "neighbor_language_stump_threshold": d["field_controls"][FIELD]["split_threshold"],
            "neighbor_language_dev_strata": d["field_controls"][FIELD]["dev_strata"],
            "frozen_single_neighbor_language_nonzero_score_responses": int(np.count_nonzero(
                np.abs(single-full) > 1e-10)),
            "frozen_single_neighbor_language_class_changes": int(np.count_nonzero(
                (single >= threshold) != (full >= threshold))),
            "frozen_native_interaction_group_nonzero_score_responses": int(np.count_nonzero(
                np.abs(native-full) > 1e-10)),
            "frozen_native_interaction_group_class_changes": int(np.count_nonzero(
                (native >= threshold) != (full >= threshold))),
            "frozen_native_neighbor_group_class_changes": g["source_groups"]["native_neighbor_means"]["class_changed_accounts"],
            "maximum_direct_model_replay_error": max_replay,
            "model_sha256": e["model_sha256"],
            "exact_effect_matrix_sha256": e["effect_matrix_sha256"],
            "source_group_matrix_sha256": g["group_effect_sha256"],
        }
        if (row["frozen_native_interaction_group_nonzero_score_responses"] !=
                g["source_groups"]["native_interaction"]["score_changed_accounts"] or
                row["frozen_native_interaction_group_class_changes"] !=
                g["source_groups"]["native_interaction"]["class_changed_accounts"]):
            raise ValueError("native interaction summary changed")
        rows.append(row)
        print(seed, row["single_stump_dev_errors_neighbor_language"],
              row["frozen_single_neighbor_language_class_changes"],
              row["frozen_native_interaction_group_class_changes"])
    result = {
        "status": "fox8_single_source_capacity_and_frozen_use_separated",
        "interpretation": "A one-field retrained dev stump measures dataset predictive capacity; frozen encoded withholding measures this model's conditional input response. They are different estimands.",
        "population_limit": "The field came from an earlier 74-feature dev screen; all dev splits and fixed test populations have been explored. This is a diagnostic, not new-population performance.",
        "fixed_test_labels_read": False,
        "dev_control_sha256": sha256(DEV),
        "frozen_exact_effect_certificate_sha256": sha256(EXACT),
        "frozen_group_effect_certificate_sha256": sha256(GROUP),
        "runs": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
