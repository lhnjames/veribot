"""Label-free full-source replay of all MGTAB evidence-only test decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.compact_source_population import load_source_population
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_full_atomic_evidence_lightgbm"
    report_path, lock_path = folder / "fixed_test_report.json", folder / "selection_lock.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if report["selection_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("MGTAB validation selection changed")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    with np.load(neighbor, allow_pickle=False) as cache:
        supports = cache["supports"].astype(float)
        source_ids = cache["ids"].astype(str).tolist()
    original_torch_load = torch.load

    def no_label_read(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels_bot.pt", "label.pt", "labels.pt"}:
            raise AssertionError("source inference attempted to read labels")
        return original_torch_load(path, *args, **kwargs)

    runs = []
    with patch.object(torch, "load", no_label_read):
        for seed in (0, 1, 2):
            run_dir = folder / f"seed{seed}"
            old_run_dir = root / f"output/v3_mgtab_ab_spline_content/selected/seed{seed}"
            x, names, ids = load_source_population(
                "mgtab", data_dir, old_run_dir,
                structural_facts=structural, neighbor_facts=neighbor)
            if ids != source_ids or x.shape != (10199, 150):
                raise ValueError("Published source population differs")
            with np.load(old_run_dir / "projection.npz", allow_pickle=False) as artifact:
                train = artifact["train_indices"].astype(int)
            reference_path = run_dir / "label_blind_reference.json.gz"
            reference = LabelBlindReference.load(reference_path)
            if reference.names != tuple(names):
                raise ValueError("Evidence reference vocabulary differs")
            fitted = LabelBlindReference(min_context_n=30).fit(x[train], names)
            if (fitted.version != reference.version or reference.version !=
                    lock["reference_version_by_seed"][str(seed)]):
                raise ValueError("Reference is not from train-only published facts")
            prediction_path = run_dir / "test_predictions.json"
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
            if (predictions["manifest"]["model_sha256"] !=
                    sha256_file(run_dir / "model.joblib")
                    or predictions["manifest"]["reference_sha256"] !=
                    sha256_file(reference_path)
                    or predictions["manifest"]["evidence_sha256"] !=
                    sha256_file(run_dir / "test_evidence.npz")):
                raise ValueError("Frozen inference artifacts differ")
            account_ids = [item["account_id"] for item in predictions["predictions"]]
            if len(account_ids) != 1020 or len(set(account_ids)) != len(account_ids):
                raise ValueError("MGTAB fixed test account list differs")
            indices = np.asarray([int(uid) for uid in account_ids], dtype=int)
            if any(ids[int(i)] != uid for i, uid in zip(indices, account_ids)):
                raise ValueError("MGTAB source row ID differs")
            batch = reference.transform(x[indices])
            reliability = batch.reliability.copy()
            reliability[:, 62:86] *= supports[indices] / (supports[indices] + 5.0)
            matrix = np.column_stack((np.log1p(batch.upper) * reliability,
                                      np.log1p(batch.lower) * reliability))
            with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as saved:
                source_diff = float(np.nanmax(np.abs(saved["raw_values"] - x[indices])))
                evidence_diff = float(np.max(np.abs(saved["evidence_matrix"] - matrix)))
                probability_saved = saved["probability"].copy()
                logit_saved = saved["logit"].copy()
                shap_saved = saved["shap"].copy()
                if (saved["ids"].astype(str).tolist() != account_ids
                        or not np.array_equal(np.isfinite(saved["raw_values"]),
                                              np.isfinite(x[indices]))
                        or source_diff > 1e-10 or evidence_diff > 1e-10):
                    raise ValueError("Frozen test evidence differs from source")
            model = joblib.load(run_dir / "model.joblib")
            probability = model.predict_proba(matrix)[:, 1]
            logit = np.asarray(model.booster_.predict(matrix, raw_score=True))
            shap = np.asarray(model.booster_.predict(matrix, pred_contrib=True))
            probability_diff = float(np.max(np.abs(probability - probability_saved)))
            logit_diff = float(np.max(np.abs(logit - logit_saved)))
            shap_diff = float(np.max(np.abs(shap - shap_saved)))
            if (shap.shape != (len(indices), 301)
                    or probability_diff > 1e-12 or logit_diff > 1e-11
                    or shap_diff > 1e-11
                    or np.max(np.abs(expit(logit) - probability)) > 1e-12
                    or np.max(np.abs(shap.sum(axis=1) - logit)) > 1e-9):
                raise ValueError("Frozen model fails source-to-decision replay")
            card_path = run_dir / "test_evidence_cards.jsonl"
            card_count = 0
            max_card_diff = 0.0
            with card_path.open(encoding="utf-8") as stream:
                for j, line in enumerate(stream):
                    card = json.loads(line)
                    if card["account_id"] != account_ids[j]:
                        raise ValueError("Evidence card account order differs")
                    contributions = card["all_fact_logit_contributions"]
                    if list(contributions) != names:
                        raise ValueError("Evidence card fact vocabulary differs")
                    full = card["intercept_logit"] + sum(contributions.values())
                    max_card_diff = max(max_card_diff, abs(full - card["logit"]))
                    if abs(card["logit"] - logit[j]) > 1e-11:
                        raise ValueError("Evidence card logit differs")
                    card_count += 1
            if card_count != len(indices) or max_card_diff > 1e-9:
                raise ValueError("Evidence card count or sums differ")
            if any(abs(item["probability"] - probability[j]) > 1e-12
                   for j, item in enumerate(predictions["predictions"])):
                raise ValueError("Saved probability differs from source replay")
            runs.append({
                "seed": seed, "test_accounts": len(indices),
                "source_fact_cells": len(indices) * 150,
                "max_source_fact_difference": source_diff,
                "max_encoded_evidence_difference": evidence_diff,
                "max_frozen_probability_difference": probability_diff,
                "max_frozen_logit_difference": logit_diff,
                "max_frozen_contribution_difference": shap_diff,
                "max_card_logit_difference": max_card_diff,
                "cards_checked": card_count,
                "label_file_read_attempts": 0,
            })
            print(json.dumps(runs[-1]), flush=True)
    output = {
        "scope": "mgtab_all_fixed_test_source_evidence_decision_label_free_replay",
        "published_source_depth": "numeric tensors and typed edges; no individual tweet text",
        "runs": runs,
    }
    path = root / "research/certificates/mgtab_full_atomic_evidence_lightgbm_replay.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
