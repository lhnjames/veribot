"""Independent full-source and frozen-decision replay of Fox8 neighbor evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def independent_neighbor_mean(features: pd.DataFrame, events: list[dict]
                              ) -> tuple[np.ndarray, np.ndarray]:
    ids = features["user_id"].astype(str).tolist()
    index = {uid: k for k, uid in enumerate(ids)}
    raw = features[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    neighbors: list[set[int]] = [set() for _ in ids]
    for event in events:
        u = index.get(str(event["source_id"]))
        v = index.get(str(event["target_id"]))
        if u is None or v is None or u == v:
            continue
        neighbors[u].add(v)
        neighbors[v].add(u)
    value = np.full((len(ids), len(FEATURE_FIELDS)), np.nan)
    support = np.zeros_like(value, dtype=np.int32)
    for i, adjacent in enumerate(neighbors):
        if not adjacent:
            continue
        selected = raw[sorted(adjacent)]
        support[i] = np.isfinite(selected).sum(axis=0)
        value[i] = np.nansum(selected, axis=0) / np.maximum(support[i], 1)
        value[i, support[i] == 0] = np.nan
    return value, support


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_neighbor_atomic_evidence_capacity"
    lock_path = folder / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report_path = folder / "fixed_test_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
    feature_path = source_dir / "features.csv"
    event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    cache_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    if (report["selection_lock_sha256"] != sha256_file(lock_path)
            or report["source_feature_sha256"] != sha256_file(feature_path)
            or report["native_event_index_sha256"] != sha256_file(event_path)
            or report["raw_neighbor_sha256"] != sha256_file(cache_path)):
        raise ValueError("Saved source or lock fingerprint differs")
    features = pd.read_csv(feature_path)
    events = json.loads(event_path.read_text(encoding="utf-8"))["events"]
    expected, expected_support = independent_neighbor_mean(features, events)
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str).tolist()
        cached = cache["values"].astype(float)
        support = cache["supports"]
    if ids != features["user_id"].astype(str).tolist():
        raise ValueError("Feature table account IDs differ")
    if not np.array_equal(support, expected_support) or not np.array_equal(
            np.isfinite(cached), np.isfinite(expected)):
        raise ValueError("Native neighbor support or missing pattern differs")
    exact_float32 = np.array_equal(cached[np.isfinite(cached)],
                                   expected.astype(np.float32)[np.isfinite(expected)])
    if not exact_float32:
        raise ValueError("All-account neighbor means differ from raw source")
    args = SimpleNamespace(
        dataset="fox8", data_dir=source_dir, account_file=None,
        native_graph_facts=None, structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    audited = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        row = report["runs"][seed]
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        test = split["test"]
        full = np.column_stack((x, expected.astype(np.float32)))
        reference = LabelBlindReference.load(run_dir / "label_blind_reference.json.gz")
        if (reference.version != lock["reference_version_by_seed"][str(seed)]
                or reference.names != tuple(names + [
                    f"native_interaction_neighbor_mean_{name}"
                    for name in FEATURE_FIELDS])):
            raise ValueError("Reference names or hash differ")
        fresh = LabelBlindReference(min_context_n=30).fit(
            full[split["train"]], reference.names)
        if fresh.version != reference.version:
            raise ValueError("Saved reference did not derive from train-only facts")
        batch = reference.transform(full[test])
        reliability = batch.reliability.copy()
        reliability[:, 43:] *= support[test] / (support[test] + 5.0)
        matrix = np.column_stack((np.log1p(batch.upper) * reliability,
                                  np.log1p(batch.lower) * reliability))
        with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as saved:
            if (saved["ids"].astype(str).tolist() != [ids[int(i)] for i in test]
                    or not np.array_equal(saved["label"], y[test])):
                raise ValueError("Test population or labels differ")
            fact_diff = float(np.nanmax(np.abs(saved["raw_values"] - full[test])))
            evidence_diff = float(np.max(np.abs(saved["evidence_matrix"] - matrix)))
            saved_probability = saved["probability"].copy()
            saved_logit = saved["logit"].copy()
            saved_shap = saved["shap"].copy()
            if fact_diff or evidence_diff:
                raise ValueError("Saved evidence differs from raw source replay")
        model = joblib.load(run_dir / "model.joblib")
        probability = model.predict_proba(matrix)[:, 1]
        logit = np.asarray(model.booster_.predict(matrix, raw_score=True))
        shap = np.asarray(model.booster_.predict(matrix, pred_contrib=True))
        max_probability_diff = float(np.max(np.abs(probability - saved_probability)))
        max_logit_diff = float(np.max(np.abs(logit - saved_logit)))
        max_shap_diff = float(np.max(np.abs(shap - saved_shap)))
        if (max_probability_diff > 1e-14 or max_logit_diff > 1e-13
                or max_shap_diff > 1e-13
                or np.max(np.abs(expit(logit) - probability)) > 1e-12
                or np.max(np.abs(shap.sum(axis=1) - logit)) > 1e-9):
            raise ValueError("Frozen model, reference or contributions differ")
        prediction_path = run_dir / "test_predictions.json"
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        if predictions["manifest"]["evidence_sha256"] != sha256_file(
                run_dir / "test_evidence.npz"):
            raise ValueError("Prediction manifest evidence hash differs")
        if any(item["account_id"] != ids[int(i)] or
               abs(item["probability"] - p) > 1e-14
               for item, i, p in zip(predictions["predictions"], test, probability)):
            raise ValueError("Prediction file differs from frozen model")
        card_count = 0
        max_card_logit_diff = 0.0
        with (run_dir / "test_evidence_cards.jsonl").open(encoding="utf-8") as stream:
            for j, line in enumerate(stream):
                card = json.loads(line)
                if card["account_id"] != ids[int(test[j])]:
                    raise ValueError("Evidence card account differs")
                card_contrib = card["all_fact_logit_contributions"]
                if list(card_contrib) != list(reference.names):
                    raise ValueError("Card contribution vocabulary differs")
                total = card["intercept_logit"] + sum(card_contrib.values())
                max_card_logit_diff = max(max_card_logit_diff,
                                          abs(total - card["logit"]))
                if abs(card["logit"] - logit[j]) > 1e-13:
                    raise ValueError("Card logit differs from model")
                card_count += 1
        if card_count != len(test) or max_card_logit_diff > 1e-9:
            raise ValueError("Evidence card count or arithmetic differs")
        audited.append({
            "seed": seed, "test_accounts": len(test), "atomic_facts_per_account": 74,
            "total_fact_cells": len(test) * 74,
            "max_raw_fact_difference": fact_diff,
            "max_encoded_evidence_difference": evidence_diff,
            "max_frozen_probability_difference": max_probability_diff,
            "max_frozen_shap_difference": max_shap_diff,
            "max_card_logit_difference": max_card_logit_diff,
            "cards_checked": card_count,
        })
        print(json.dumps(audited[-1]), flush=True)
    output = {
        "scope": "fox8_all_account_raw_neighbor_facts_and_all_test_decisions_independent_replay",
        "native_neighbor_accounts": len(ids),
        "native_neighbor_fact_cells": int(expected.size),
        "all_neighbor_means_match_raw_source_at_float32_precision": exact_float32,
        "interaction_event_index_sha256": sha256_file(event_path),
        "feature_table_sha256": sha256_file(feature_path),
        "runs": audited,
    }
    path = root / "research/certificates/fox8_neighbor_atomic_evidence_full_replay.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
