"""Propagate Fox8's alternative duplicate row through native facts and frozen models."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_adapter import (
    FEATURE_FIELDS, profile_features, tweet_features)
from verifiable_bot.fox8_native_interaction_facts import build as build_native
from verifiable_bot.fox8_native_neighbor_profile_facts import extract as extract_neighbors
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.high_performance_fact_effects import evidence_design


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_duplicate_source_sensitivity"
OUT = ROOT / "research/certificates/fox8_duplicate_source_sensitivity.json"
DUPLICATE = ROOT / "research/certificates/fox8_duplicate_source_row_audit.json"


def row_features(row: dict) -> np.ndarray:
    tweets = row["user_tweets"]
    values = {**profile_features(tweets[0]["user"]),
              **tweet_features(tweets)}
    return np.asarray([values[name] for name in FEATURE_FIELDS], dtype=np.float32)


def transformed_probability(runtime: Fox8NeighborAtomicRuntime,
                            facts: np.ndarray, supports: np.ndarray,
                            rows: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts[rows])
    reliability = batch.reliability.copy()
    reliability[:, 43:74] *= supports[rows] / (supports[rows] + 5.0)
    design = np.column_stack((np.log1p(batch.upper) * reliability,
                              np.log1p(batch.lower) * reliability))
    if design.shape != (len(rows), 148) or not np.isfinite(design).all():
        raise ValueError("alternative source did not yield model evidence")
    return np.asarray(runtime.model.predict_proba(design)[:, 1], dtype=float)


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    audit = json.loads(DUPLICATE.read_text(encoding="utf-8"))
    uid = audit["account_id"]
    first_line, later_line = audit["first_line"], audit["later_line"]
    if (first_line != 241 or later_line != 397 or
            audit["same_31_model_account_features"]):
        raise ValueError("Fox8 duplicate source audit differs")
    runtime0 = Fox8NeighborAtomicRuntime(ROOT, 0)
    frame = pd.read_csv(runtime0.features_path)
    if frame["user_id"].astype(str).tolist() != runtime0.ids:
        raise ValueError("Fox8 published account order differs")
    duplicate_row = runtime0.index[uid]
    first_raw = later_raw = None
    with gzip.open(runtime0.raw_path, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number == first_line:
                first_raw = line
            elif line_number == later_line:
                later_raw = line
    if first_raw is None or later_raw is None:
        raise ValueError("duplicate raw line is absent")
    if (hashlib.sha256(first_raw).hexdigest() != audit["first_row_sha256"] or
            hashlib.sha256(later_raw).hexdigest() != audit["later_row_sha256"]):
        raise ValueError("duplicate raw record hash differs")
    first_record, later_record = json.loads(first_raw), json.loads(later_raw)
    if str(first_record["user_id"]) != uid or str(later_record["user_id"]) != uid:
        raise ValueError("duplicate raw account differs")
    baseline_feature = frame.loc[duplicate_row, list(FEATURE_FIELDS)].to_numpy(
        dtype=np.float32)
    first_feature = row_features(first_record)
    later_feature = row_features(later_record)
    if not np.allclose(first_feature, baseline_feature, rtol=1e-5, atol=1e-5):
        raise ValueError("published feature row does not match first raw record")
    alternate_frame = frame.copy()
    alternate_frame.loc[duplicate_row, list(FEATURE_FIELDS)] = later_feature
    FOLDER.mkdir(parents=True, exist_ok=True)
    temporary_raw = FOLDER / "alternative_source_temp.ndjson.gz"
    native_path = FOLDER / "native_interaction_facts.npz"
    try:
        with gzip.open(runtime0.raw_path, "rb") as original, \
                temporary_raw.open("wb") as sink, gzip.GzipFile(
                    filename="", mode="wb", fileobj=sink, mtime=0,
                    compresslevel=6) as target:
            for line_number, line in enumerate(original, 1):
                target.write(later_raw if line_number == first_line else line)
        native_manifest = build_native(temporary_raw, runtime0.features_path,
                                       native_path)
    finally:
        temporary_raw.unlink(missing_ok=True)
    event_path = FOLDER / "interaction_event_index.json"
    alternate_events = json.loads(event_path.read_text(encoding="utf-8"))["events"]
    original_events = json.loads(runtime0.event_path.read_text(
        encoding="utf-8"))["events"]
    with np.load(native_path, allow_pickle=False) as native:
        if native["ids"].astype(str).tolist() != runtime0.ids:
            raise ValueError("alternative native account order differs")
        alternate_native = native["values"].astype(float)
    alternate_neighbor, alternate_support, names, graph_summary = \
        extract_neighbors(alternate_frame, alternate_events)
    if (alternate_neighbor.shape != (2279, 31) or
            names != list(runtime0.reference.names[43:74])):
        raise ValueError("alternative native neighbor facts differ")
    native_cell_changes = int(np.count_nonzero(~np.isclose(
        alternate_native, runtime0.facts[:, 31:43], rtol=0, atol=0,
        equal_nan=True)))
    neighbor_cell_changes = int(np.count_nonzero(~np.isclose(
        alternate_neighbor, runtime0.facts[:, 43:74], rtol=0, atol=0,
        equal_nan=True)))
    support_cell_changes = int(np.count_nonzero(
        alternate_support != runtime0.supports))
    runs = []
    for seed in (0, 1, 2):
        runtime = (runtime0 if seed == 0 else Fox8NeighborAtomicRuntime(ROOT, seed))
        if runtime.ids != runtime0.ids:
            raise ValueError("seed source account order differs")
        alternative = runtime.facts.copy()
        alternative[duplicate_row, :31] = later_feature
        alternative[:, 31:43] = alternate_native
        alternative[:, 43:74] = alternate_neighbor
        test_path = (ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity" /
                     f"seed{seed}/test_predictions.json")
        recorded = json.loads(test_path.read_text(encoding="utf-8"))["predictions"]
        ids = [str(row["account_id"]) for row in recorded]
        rows = np.asarray([runtime.index[account] for account in ids], dtype=int)
        baseline = np.asarray([row["probability"] for row in recorded], dtype=float)
        replay = runtime.model.predict_proba(evidence_design(runtime, rows))[:, 1]
        if np.max(np.abs(replay-baseline)) > 1e-12:
            raise ValueError("frozen baseline differs from saved test")
        altered = transformed_probability(runtime, alternative,
                                          alternate_support, rows)
        delta = altered - baseline
        baseline_class = baseline >= runtime.threshold
        altered_class = altered >= runtime.threshold
        changed_facts = ~np.isclose(alternative[rows], runtime.facts[rows],
                                    rtol=0, atol=0, equal_nan=True)
        affected = []
        for position in np.flatnonzero(np.any(changed_facts, axis=1)):
            affected.append({
                "account_id": ids[int(position)],
                "changed_source_fact_names": [runtime.reference.names[j]
                    for j in np.flatnonzero(changed_facts[position])],
                "frozen_probability": float(baseline[position]),
                "alternate_probability": float(altered[position]),
                "probability_difference": float(delta[position]),
                "frozen_class": int(baseline_class[position]),
                "alternate_class": int(altered_class[position]),
            })
        runs.append({
            "seed": seed, "fixed_test_accounts": len(ids),
            "test_accounts_with_changed_source_facts": len(affected),
            "test_accounts_with_changed_probability": int(np.count_nonzero(
                np.abs(delta) > 1e-12)),
            "test_accounts_with_changed_class": int(np.count_nonzero(
                baseline_class != altered_class)),
            "maximum_absolute_probability_difference": float(np.max(
                np.abs(delta))),
            "source_affected_accounts": affected,
            "saved_test_predictions_sha256": sha256_file(test_path),
        })
        print({key: runs[-1][key] for key in (
            "seed", "test_accounts_with_changed_source_facts",
            "test_accounts_with_changed_probability",
            "test_accounts_with_changed_class")}, flush=True)
    OUT.write_text(json.dumps({
        "status": "duplicate_record_alternative_source_propagated_through_frozen_fox8",
        "source_truth_status": "two conflicting published rows; no independent truth selecting either",
        "intervention": "replace first duplicate row with second, rebuild 31 account facts, native interaction facts, graph and neighbor means; keep model/reference/threshold frozen",
        "duplicate_source_audit_sha256": sha256_file(DUPLICATE),
        "original_raw_sha256": sha256_file(runtime0.raw_path),
        "alternative_native_manifest": native_manifest,
        "alternative_native_fact_sha256": sha256_file(native_path),
        "alternative_event_index_sha256": sha256_file(event_path),
        "alternative_graph_summary": graph_summary,
        "native_event_records_changed": bool(alternate_events != original_events),
        "native_fact_cells_changed": native_cell_changes,
        "neighbor_mean_cells_changed": neighbor_cell_changes,
        "neighbor_support_cells_changed": support_cell_changes,
        "test_labels_accessed": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
