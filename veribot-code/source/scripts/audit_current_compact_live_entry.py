"""Check live readable decisions without access to test predictions or labels."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from predict_current_compact_verified import load_runtime, render_account


CONFIG = {
    "fox8": ("output/v3_fox8_targeted_neighbor_20_concepts",
             "ab_group_tree_test_predictions.json"),
    "mgtab": ("output/v3_mgtab_targeted_20_concept_ab_group_tree",
              "test_predictions.json"),
}


def _saved_reports(path: Path) -> dict[str, dict]:
    reports = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            uid = item["account_id"]
            if uid in reports:
                raise ValueError("duplicate fixed-test report account")
            reports[uid] = item
    return reports


def _check_same(live: dict, saved: dict) -> tuple[float, int]:
    if (live["account_id"] != saved["account_id"]
            or live["model_prediction"] != saved["model_prediction"]
            or len(live["source_groups"]) != 10):
        raise ValueError("live and saved report decision differ")
    maximum = abs(live["bot_probability"] - saved["bot_probability"])
    anchors = 0
    for actual, expected in zip(live["source_groups"],
                                saved["source_groups"], strict=True):
        if actual["group"] != expected["group"]:
            raise ValueError("live source group differs")
        maximum = max(maximum, abs(actual["probability_without_group"] -
                                   expected["probability_without_group"]))
        for fact, stored in zip(actual["facts"], expected["facts"], strict=True):
            if (fact["concept"] != stored["concept"]
                    or fact["winning_fact"] != stored["winning_fact"]
                    or fact["source_locator"] != stored["source_locator"]
                    or fact["claim"] != stored["claim"]):
                raise ValueError("live source locator or literal claim differs")
            anchors += fact["source_locator"] is not None
    if maximum > 1e-12:
        raise ValueError("live fixed-model probability differs")
    return maximum, anchors


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original_text = Path.read_text
    original_open = Path.open
    original_csv = pd.read_csv
    original_torch = torch.load
    forbidden_attempts = []

    def guarded_text(path, *args, **kwargs):
        name = path.name.lower()
        if ("test_predictions" in name or "test_evidence_cards" in name
                or name.startswith("fixed_test_")
                or name == "report.json"
                or name.startswith("labels")):
            forbidden_attempts.append(str(path))
            raise AssertionError("live entry attempted to read a test card or label")
        return original_text(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        name = path.name.lower()
        if ("test_predictions" in name or "test_evidence_cards" in name
                or name.startswith("fixed_test_") or name == "report.json"
                or name.startswith("labels")):
            forbidden_attempts.append(str(path))
            raise AssertionError("live entry attempted to open a test card or label")
        return original_open(path, *args, **kwargs)

    def guarded_csv(path, *args, **kwargs):
        if "label" in Path(path).name.lower():
            forbidden_attempts.append(str(path))
            raise AssertionError("live entry attempted to read label CSV")
        return original_csv(path, *args, **kwargs)

    def guarded_torch(path, *args, **kwargs):
        if "label" in Path(path).name.lower():
            forbidden_attempts.append(str(path))
            raise AssertionError("live entry attempted to read label tensor")
        return original_torch(path, *args, **kwargs)

    rows = []
    for dataset, (folder, prediction_name) in CONFIG.items():
        for seed in (0, 1, 2):
            saved_predictions = json.loads((root / folder / f"seed{seed}" /
                                           prediction_name).read_text(encoding="utf-8"))
            test_ids = {item["account_id"] for item in saved_predictions["predictions"]}
            test_order = [item["account_id"] for item in saved_predictions["predictions"]]
            if dataset == "fox8":
                population = original_csv(Path("D:/Code/SAHG/SAHG/code/data/fox8_23/features.csv"))[
                    "user_id"].astype(str).tolist()
            else:
                population = [str(i) for i in range(10199)]
            non_test_id = next(uid for uid in population if uid not in test_ids)
            saved_path = (root / "output/v3_current_compact_verified_reports" /
                          f"{dataset}_seed{seed}_verified_reports.jsonl.gz")
            stored = _saved_reports(saved_path)
            if set(stored) != test_ids or len(stored) != len(test_order):
                raise ValueError("fixed-test report population differs")
            with patch.object(Path, "read_text", guarded_text), \
                    patch.object(Path, "open", guarded_open), \
                    patch.object(pd, "read_csv", guarded_csv), \
                    patch.object(torch, "load", guarded_torch):
                runtime = load_runtime(root, dataset, seed)
                maximum = 0.0
                anchors = 0
                for test_id in test_order:
                    live = render_account(runtime, dataset, seed, test_id)
                    difference, count = _check_same(live, stored[test_id])
                    maximum = max(maximum, difference)
                    anchors += count
                    if "label" in live:
                        raise ValueError("live report contains a test label")
                other = render_account(runtime, dataset, seed, non_test_id)
            if ("label" in other or other["account_id"] != non_test_id
                    or len(other["source_groups"]) != 10):
                raise ValueError("live report contains a test label or lost an account")
            row = {"dataset": dataset, "seed": seed,
                   "fixed_test_accounts_checked": len(test_order),
                   "non_test_account_checked": non_test_id,
                   "source_locators_compared": anchors,
                   "max_fixed_test_probability_difference": maximum,
                   "blocked_test_card_or_label_reads": len(forbidden_attempts),
                   "saved_report_sha256": sha256_file(saved_path)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    path = root / "research/certificates/current_compact_live_entry_audit.json"
    path.write_text(json.dumps({
        "scope": "live_source_to_readable_decision_without_test_cards_or_labels",
        "runs": rows,
        "status": "all_fixed_test_and_six_non_test_live_reports_match_source_bound_contract",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
