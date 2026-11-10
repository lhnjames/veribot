"""Recompute every frozen Fox8 or MGTAB test prediction through the live source entry."""

from __future__ import annotations

import argparse
import builtins
import json
import math
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.mgtab_ab_positive_human_runtime import MGTABABPositiveHumanRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {
    "fox8": ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": ROOT / "output/v3_mgtab_ab_group_additive_tree",
}


def run_dataset(dataset: str, *, positive_human: bool = False) -> dict:
    if positive_human and dataset != "mgtab":
        raise ValueError("positive Human variant is available for MGTAB")
    folder = (ROOT / "output/v3_mgtab_ab_group_positive_human"
              if positive_human else FOLDERS[dataset])
    original_text = Path.read_text
    original_path_open = Path.open
    original_open = builtins.open
    original_csv = pd.read_csv
    original_torch = torch.load
    forbidden = []

    def protected(path) -> bool:
        if not isinstance(path, (str, bytes, Path)):
            return False
        name = Path(path).name.lower()
        return ("test_predictions" in name or "test_evidence" in name or
                name in {"fixed_test_report.json", "report.json", "labels.csv",
                         "label.csv", "labels_bot.pt", "labels.pt",
                         "source_group_decision_certificates.jsonl"})

    def guard(original):
        def checked(path, *args, **kwargs):
            if protected(path):
                forbidden.append(str(path))
                raise AssertionError("live inference read protected evaluation data")
            return original(path, *args, **kwargs)
        return checked

    rows = []
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    for seed in (0, 1, 2):
        saved_path = folder / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        expected = saved["predictions"]
        threshold = float(saved["manifest"]["threshold"])
        before = len(forbidden)
        with patch.object(Path, "read_text", guard(original_text)), \
                patch.object(Path, "open", guard(original_path_open)), \
                patch.object(builtins, "open", guard(original_open)), \
                patch.object(pd, "read_csv", guard(original_csv)), \
                patch.object(torch, "load", guard(original_torch)):
            runtime = (Fox8NeighborAtomicRuntime(ROOT, seed)
                       if dataset == "fox8" else
                       MGTABABPositiveHumanRuntime(ROOT, seed)
                       if positive_human else
                       MGTABFullAtomicRuntime(ROOT, seed,
                                              variant="ab_group_additive"))
            if runtime.threshold != threshold:
                raise ValueError("live threshold differs from sealed test threshold")
            max_probability_difference = 0.0
            zero_evidence_class_changes = 0
            class_critical_native_or_relation = 0
            positive_human_only_mask_changes = 0
            max_logit_reconstruction_error = 0.0
            max_group_additivity_error = 0.0
            for j, item in enumerate(expected):
                actual = runtime.predict(item["account_id"])
                max_probability_difference = max(
                    max_probability_difference,
                    abs(actual["probability"]-float(item["probability"])))
                if (actual["account_id"] != item["account_id"] or
                        actual["prediction"] !=
                        int(float(item["probability"]) >= threshold) or
                        "label" in actual or
                        len(actual["evidence"]) !=
                        (74 if dataset == "fox8" else 150)):
                    raise ValueError("live account, class, or evidence ledger differs")
                terms = [float(term["logit_contribution"])
                         for term in actual["evidence"]]
                if positive_human:
                    if len(actual["positive_human_evidence"]) != 9:
                        raise ValueError("positive Human source groups differ")
                    positive_human_only_mask_changes += int(actual[
                        "without_positive_human_evidence_changes_class"])
                    terms += [float(value) for value in actual[
                        "positive_human_logit_attributions"]]
                error = abs(float(actual["intercept_logit"]) +
                            math.fsum(terms) - float(actual["logit"]))
                max_logit_reconstruction_error = max(
                    max_logit_reconstruction_error, error)
                if dataset == "fox8":
                    zero_evidence_class_changes += int(actual[
                        "evidence_dependency"]["all_evidence_changes_class"])
                    class_critical_native_or_relation += int(actual[
                        "evidence_dependency"][
                            "native_neighbor_evidence_changes_class"])
                else:
                    zero_evidence_class_changes += int(actual[
                        "all_evidence_changes_class"])
                    class_critical_native_or_relation += int(any(
                        actual["group_dependencies"][name][
                            "class_changes_without_group"]
                        for name in actual["group_dependencies"]
                        if name.startswith("published_relation_")))
                    max_group_additivity_error = max(
                        max_group_additivity_error,
                        float(actual["group_logit_additivity_error"]))
                    if (not positive_human and
                            actual["source_group_decision_certificate"] is None):
                        raise ValueError("MGTAB group decision certificate is absent")
                if j and j % 250 == 0:
                    print(dataset, seed, j, flush=True)
            test_ids = {item["account_id"] for item in expected}
            non_test = next(uid for uid in runtime.ids if uid not in test_ids)
            outside = runtime.predict(non_test)
        if (len(forbidden) != before or
                max_probability_difference > 1e-12 or
                max_logit_reconstruction_error > 1e-9 or
                max_group_additivity_error > 1e-9 or
                outside["account_id"] != non_test):
            raise ValueError("full-population live inference audit differs")
        row = {
            "dataset": dataset, "seed": seed,
            "fixed_test_accounts_checked": len(expected),
            "non_test_account_checked": non_test,
            "max_probability_difference": max_probability_difference,
            "max_logit_reconstruction_error":
                max_logit_reconstruction_error,
            "max_group_additivity_error": max_group_additivity_error,
            "zero_evidence_class_changes": zero_evidence_class_changes,
            "native_or_relation_class_critical_accounts":
                class_critical_native_or_relation,
            "protected_evaluation_reads": len(forbidden)-before,
        }
        if positive_human:
            row["positive_human_only_mask_changes"] = (
                positive_human_only_mask_changes)
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = (ROOT / "research/certificates" /
              ("mgtab_ab_positive_human_live_full_population.json"
               if positive_human else f"{dataset}_best_live_full_population.json"))
    payload = {
        "scope": "all_fixed_test_and_one_non_test_per_seed_live_source_to_evidence_to_decision",
        "model": ("raw_neighbor_atomic_74" if dataset == "fox8" else
                  "ab_group_additive_positive_human_309" if positive_human else
                  "ab_group_additive_150"),
        "runs": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FOLDERS), required=True)
    parser.add_argument("--positive-human", action="store_true")
    args = parser.parse_args()
    run_dataset(args.dataset, positive_human=args.positive_human)
