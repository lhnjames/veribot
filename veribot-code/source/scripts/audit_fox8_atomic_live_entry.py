"""Audit Fox8 live atomic source effects against saved full-population ablations."""

from __future__ import annotations

import gzip
import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_twenty_concept_runtime import Fox8TwentyConceptRuntime

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_targeted_neighbor_20_concepts"
OUT = ROOT / "research/certificates/fox8_atomic_live_entry_audit.json"


def main() -> None:
    certificate_path = ROOT / "research/certificates/fox8_current_atomic_fact_withholding.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    original_text, original_open = Path.read_text, Path.open
    original_csv, original_torch = pd.read_csv, torch.load
    forbidden = []

    def blocked(path) -> bool:
        name = Path(path).name.lower()
        return ("test_predictions" in name or "test_evidence" in name or
                "test_cards" in name or "withholding_cards" in name or
                name.startswith("fixed_test") or name == "report.json" or
                name.startswith("labels") or name in {"label.pt", "label.csv"})

    def guarded_text(path, *args, **kwargs):
        if blocked(path):
            forbidden.append(str(path))
            raise AssertionError("live Fox8 path attempted to read test or label")
        return original_text(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        if blocked(path):
            forbidden.append(str(path))
            raise AssertionError("live Fox8 path attempted to open test or label")
        return original_open(path, *args, **kwargs)

    def guarded_csv(path, *args, **kwargs):
        if blocked(path):
            forbidden.append(str(path))
            raise AssertionError("live Fox8 path attempted label CSV")
        return original_csv(path, *args, **kwargs)

    def guarded_torch(path, *args, **kwargs):
        if blocked(path):
            forbidden.append(str(path))
            raise AssertionError("live Fox8 path attempted label tensor")
        return original_torch(path, *args, **kwargs)

    runs = []
    for seed in (0, 1, 2):
        saved_path = FOLDER / f"seed{seed}/atomic_fact_withholding_cards.jsonl.gz"
        expected = certificate["runs"][seed]
        if expected["withholding_cards_sha256"] != sha256_file(saved_path):
            raise ValueError("frozen atomic source card differs")
        with gzip.open(saved_path, "rt", encoding="utf-8") as stream:
            saved = [json.loads(line) for line in stream]
        ids = [item["account_id"] for item in saved]
        test_ids = set(ids)
        if len(test_ids) != len(saved):
            raise ValueError("duplicate test account")
        non_test = next(uid for uid in pd.read_csv(
            "D:/Code/SAHG/SAHG/code/data/fox8_23/features.csv")[
                "user_id"].astype(str) if uid not in test_ids)
        with patch.object(Path, "read_text", guarded_text), \
                patch.object(Path, "open", guarded_open), \
                patch.object(pd, "read_csv", guarded_csv), \
                patch.object(torch, "load", guarded_torch):
            runtime = Fox8TwentyConceptRuntime(ROOT, seed)
            max_probability_difference = 0.0
            effect_entries = 0
            class_critical = 0
            for j, item in enumerate(saved):
                live = runtime.predict_with_atomic_fact_effects(item["account_id"])
                if (live["account_id"] != item["account_id"] or
                        live["prediction"] != item["prediction"] or
                        "label" in live):
                    raise ValueError("live atomic decision differs")
                max_probability_difference = max(
                    max_probability_difference,
                    abs(live["probability"]-item["full_probability"]))
                actual = live["atomic_fact_effects"]
                frozen = item["cited_fact_withholding"]
                if len(actual) != len(frozen):
                    raise ValueError("live cited fact count differs")
                for a, b in zip(actual, frozen, strict=True):
                    if (a["fact"] != b["fact"] or
                            a["source_locator"] != b["source_locator"] or
                            a["replacement_concepts"] != b["replacement_concepts"] or
                            a["probability_changes_gt_1e_10"] !=
                            b["probability_changes_gt_1e_10"] or
                            a["class_changes"] != b["class_changes"]):
                        raise ValueError("live atomic fact or locator differs")
                    max_probability_difference = max(
                        max_probability_difference,
                        abs(a["probability_without_fact"]-
                            b["probability_without_fact"]))
                    effect_entries += 1
                    class_critical += a["class_changes"]
                if j and j % 250 == 0:
                    print("seed", seed, "checked", j, flush=True)
            other = runtime.predict_with_atomic_fact_effects(non_test)
        if (max_probability_difference > 1e-12 or
                effect_entries != expected["cited_unique_fact_entries"] or
                class_critical != expected["cited_entries_with_class_change"] or
                "label" in other or other["account_id"] != non_test or forbidden):
            raise ValueError("live atomic source audit differs")
        runs.append({
            "seed": seed,
            "fixed_test_accounts_checked": len(ids),
            "non_test_account_checked": non_test,
            "cited_fact_effects_checked": effect_entries,
            "class_critical_cited_facts": class_critical,
            "max_probability_difference": max_probability_difference,
            "blocked_test_or_label_reads": len(forbidden),
            "frozen_atomic_cards_sha256": sha256_file(saved_path),
        })
        print(json.dumps(runs[-1]), flush=True)
    OUT.write_text(json.dumps({
        "scope": "all_fox8_live_account_atomic_fact_effects_vs_independent_frozen_cards",
        "atomic_withholding_certificate_sha256": sha256_file(certificate_path),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
