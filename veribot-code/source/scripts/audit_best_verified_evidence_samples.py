"""Audit one selected evidence card per frozen dataset and seed."""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file
from predict_best_verified_evidence import predict


ROOT = Path(__file__).resolve().parents[2]
SAVED = {
    "botsim": "output/v3_botsim_observed_means_12",
    "fox8": "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": "output/v3_mgtab_ab_group_additive_tree",
}
COUNT = {"botsim": 12, "fox8": 74, "mgtab": 150}
OUT = ROOT / "research/certificates/best_verified_evidence_nine_samples.json"


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    original_text = Path.read_text
    original_open = Path.open
    original_csv = pd.read_csv
    original_torch = torch.load
    forbidden_reads = []

    def forbidden(path) -> bool:
        name = Path(path).name.lower()
        return ("test_predictions" in name or "test_evidence" in name or
                name in {"report.json", "fixed_test_report.json", "labels.csv",
                         "label.csv", "labels_bot.pt", "labels.pt"})

    def checked_text(path: Path, *args, **kwargs):
        if forbidden(path):
            forbidden_reads.append(str(path))
            raise AssertionError("selected evidence inference read a result or label")
        return original_text(path, *args, **kwargs)

    def checked_open(path: Path, *args, **kwargs):
        if forbidden(path):
            forbidden_reads.append(str(path))
            raise AssertionError("selected evidence inference opened a result or label")
        return original_open(path, *args, **kwargs)

    def checked_csv(path, *args, **kwargs):
        if forbidden(path):
            forbidden_reads.append(str(path))
            raise AssertionError("selected evidence inference read labels")
        return original_csv(path, *args, **kwargs)

    def checked_torch(path, *args, **kwargs):
        if forbidden(path):
            forbidden_reads.append(str(path))
            raise AssertionError("selected evidence inference loaded labels")
        return original_torch(path, *args, **kwargs)

    rows = []
    for dataset, folder in SAVED.items():
        for seed in (0, 1, 2):
            saved_path = ROOT / folder / f"seed{seed}/test_predictions.json"
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            target = saved["predictions"][0]
            uid = str(target["account_id"])
            before = len(forbidden_reads)
            with patch.object(Path, "read_text", checked_text), \
                    patch.object(Path, "open", checked_open), \
                    patch.object(pd, "read_csv", checked_csv), \
                    patch.object(torch, "load", checked_torch):
                card = predict(ROOT, dataset, seed, uid)
            if len(forbidden_reads) != before:
                raise ValueError("inference attempted a forbidden read")
            terms = card["evidence_terms"]
            if (card["account_id"] != uid or len(terms) != COUNT[dataset] or
                    card["classification_labels_accessed"] is not False or
                    card["caller_supplied_fact_matrix_accepted"] is not False or
                    abs(card["probability"] - float(target["probability"])) > 1e-12):
                raise ValueError("selected card differs from fixed test decision")
            binding = card["source_binding"]
            if dataset == "botsim":
                if binding["independent_decision_audit_sha256"] != sha256_file(
                        ROOT / "research/certificates/botsim_observed_means_source_to_decision.json"):
                    raise ValueError("BotSim source decision audit differs")
            elif (binding["independent_source_audit_sha256"] != sha256_file(
                    Path(binding["independent_source_audit"])) or
                  binding["inference_manifest_sha256"] != sha256_file(
                    Path(binding["inference_manifest"]))):
                raise ValueError("source or model audit digest differs")
            if dataset == "botsim":
                threshold = json.loads((ROOT / folder / "dev_only.json")
                                       .read_text(encoding="utf-8"))["runs"][seed]["threshold"]
            else:
                threshold = saved["manifest"]["threshold"]
            if (abs(card["threshold"] - threshold) > 1e-12 or
                    card["prediction"] != int(target["probability"] >= threshold)):
                raise ValueError("card threshold or class differs from fixed test")
            source_count = 0
            projection_count = 0
            for term in terms:
                query = term["source_query"]
                if term["source_fact_observed"]:
                    if query is None or not query["source_ids"]:
                        raise ValueError("observed model fact has no source")
                    for source in query["source_ids"]:
                        if not Path(source.split("#", 1)[0]).is_file():
                            raise ValueError("source query points to a missing file")
                    source_count += 1
                elif dataset == "botsim" and (
                        term["concept_available"] or
                        abs(term["logit_attribution"]) > 1e-12):
                    raise ValueError("unavailable BotSim concept affects decision")
                if term["source_level"] == (
                        "published_averaged_tweet_tensor_train_only_projection"):
                    projection_count += 1
                if term["attribution_method"] != (
                        "exact_additive_concept" if dataset == "botsim" else
                        "tree_shap_logit_allocation"):
                    raise ValueError("evidence attribution method misstated")
            reconstructed = card["intercept_logit"] + math.fsum(
                item["logit_attribution"] for item in terms)
            if (abs(reconstructed - card["logit"]) > 1e-9 or
                    abs(float(expit(card["logit"])) - card["probability"]) > 1e-12):
                raise ValueError("evidence logit or probability differs")
            if dataset == "mgtab" and projection_count != 64:
                raise ValueError("MGTAB numeric content source count differs")
            if dataset == "mgtab" and card[
                    "exact_frozen_model_dependency"][
                        "source_group_decision_certificate"] is None:
                raise ValueError("MGTAB exact source group decision certificate absent")
            rows.append({
                "dataset": dataset, "seed": seed, "account_id": uid,
                "evidence_terms": len(terms),
                "observed_terms_with_existing_source_file": source_count,
                "published_numeric_tweet_projection_terms": projection_count,
                "probability_difference_from_saved": card["probability"] -
                float(target["probability"]),
                "logit_reconstruction_difference": reconstructed - card["logit"],
                "evidence_changes_class_from_zero": card[
                    "exact_frozen_model_dependency"]["all_evidence_changes_class"],
                "forbidden_read_attempts": len(forbidden_reads) - before,
                "saved_predictions_sha256": sha256_file(saved_path),
            })
            print({key: rows[-1][key] for key in (
                "dataset", "seed", "evidence_terms",
                "probability_difference_from_saved",
                "logit_reconstruction_difference")}, flush=True)
    OUT.write_text(json.dumps({
        "status": "nine_selected_source_evidence_card_samples_verified",
        "scope": "one_fixed_test_account_per_dataset_seed",
        "selected_card_entry": str(ROOT / "source/scripts/predict_best_verified_evidence.py"),
        "selected_card_entry_sha256": sha256_file(
            ROOT / "source/scripts/predict_best_verified_evidence.py"),
        "full_population_batch_replay_sha256": sha256_file(
            ROOT / "research/certificates/best_evidence_batch_observed_means_full_test.json"),
        "guarded_label_or_test_result_read_attempts": len(forbidden_reads),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
