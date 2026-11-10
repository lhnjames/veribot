"""Check every new MGTAB live inference without allowing label/test file reads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_positive_human_runtime import MGTABPositiveHumanRuntime
from verifiable_bot.mgtab_positive_human_report import render_positive_human


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_positive_human_evidence"
OUT = ROOT / "research/certificates/mgtab_positive_human_live_entry_audit.json"


def main() -> None:
    report = json.loads((FOLDER / "fixed_test_report.json")
                        .read_text(encoding="utf-8"))
    source_audit = json.loads((ROOT /
        "research/certificates/mgtab_positive_human_evidence_source_replay.json")
        .read_text(encoding="utf-8"))
    original_text = Path.read_text
    original_open = Path.open
    original_csv = pd.read_csv
    original_torch = torch.load
    forbidden = []

    def blocked_name(path) -> bool:
        name = Path(path).name.lower()
        return ("test_predictions" in name or "test_evidence" in name or
                name.startswith("fixed_test") or name == "report.json" or
                "label" in name)

    def guarded_text(path, *args, **kwargs):
        if blocked_name(path):
            forbidden.append(str(path))
            raise AssertionError("live path attempted test result or label read")
        return original_text(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        if blocked_name(path):
            forbidden.append(str(path))
            raise AssertionError("live path attempted test result or label read")
        return original_open(path, *args, **kwargs)

    def guarded_csv(path, *args, **kwargs):
        if blocked_name(path):
            forbidden.append(str(path))
            raise AssertionError("live path attempted label CSV read")
        return original_csv(path, *args, **kwargs)

    def guarded_torch(path, *args, **kwargs):
        if blocked_name(path):
            forbidden.append(str(path))
            raise AssertionError("live path attempted label tensor read")
        return original_torch(path, *args, **kwargs)

    rows = []
    for seed in (0, 1, 2):
        run_dir = FOLDER / f"seed{seed}"
        evidence_path = run_dir / "test_evidence.npz"
        with np.load(evidence_path, allow_pickle=False) as saved:
            ids = saved["ids"].astype(str).tolist()
            x = saved["model_input"].copy()
            probability = saved["probability"].copy()
            threshold = float(saved["threshold"])
            groups = saved["source_group_names"].astype(str).tolist()
        test_ids = set(ids)
        non_test = next(str(i) for i in range(10199) if str(i) not in test_ids)
        with patch.object(Path, "read_text", guarded_text), \
                patch.object(Path, "open", guarded_open), \
                patch.object(pd, "read_csv", guarded_csv), \
                patch.object(torch, "load", guarded_torch):
            runtime = MGTABPositiveHumanRuntime(ROOT, seed)
            max_error = 0.0
            max_human_error = 0.0
            locators = 0
            group_mask_changes = {g: 0 for g in groups}
            human_changes = 0
            for j, uid in enumerate(ids):
                live = runtime.predict(uid)
                readable = render_positive_human(live, seed)
                if (live["account_id"] != uid or
                        live["prediction"] != int(probability[j] >= threshold) or
                        "label" in live or
                        readable["account_id"] != uid or
                        readable["model_prediction"] != live["prediction"] or
                        readable["bot_probability"] != live["probability"] or
                        readable["without_positive_human_evidence_changes_class"] !=
                        live["without_positive_human_evidence_changes_class"] or
                        len(readable["source_groups"]) != 10 or
                        list(live["source_group_effects"]) != groups or
                        len(live["evidence"]) != 20 or
                        len(live["positive_human_evidence"]) != 10):
                    raise ValueError("live account, class, or evidence structure differs")
                max_error = max(max_error, abs(live["probability"]-probability[j]))
                for k, row in enumerate(live["positive_human_evidence"]):
                    max_human_error = max(max_human_error,
                                          abs(row["human_normality_value"]-x[j, 40+k]))
                    if (row["source_group"] != groups[k] or
                            any(fact["source"] is None for fact in row["source_facts"])):
                        raise ValueError("live positive Human source differs")
                    locators += len(row["source_facts"])
                    group_mask_changes[groups[k]] += int(
                        live["source_group_effects"][groups[k]]["class_changes_without_group"])
                human_changes += int(live[
                    "without_positive_human_evidence_changes_class"])
                if j and j % 250 == 0:
                    print("seed", seed, "checked", j, flush=True)
            other = runtime.predict(non_test)
            other_report = render_positive_human(other, seed)
        expected = report["runs"][seed]
        audited = source_audit["runs"][seed]
        if (max_error > 1e-12 or max_human_error > 1e-12 or
                locators != audited["source_locator_concepts"] or
                group_mask_changes != expected["group_mask_changed_classes"] or
                human_changes != audited["human_normality_only_mask_changes"] or
                "label" in other or other["account_id"] != non_test or
                other_report["account_id"] != non_test or
                forbidden):
            raise ValueError("live source or frozen decision audit differs")
        rows.append({
            "seed": seed, "fixed_test_accounts_checked": len(ids),
            "non_test_account_checked": non_test,
            "positive_human_source_locators": locators,
            "max_probability_difference": max_error,
            "max_human_evidence_value_difference": max_human_error,
            "positive_human_only_mask_changed_classes": human_changes,
            "source_group_mask_changes": group_mask_changes,
            "blocked_test_or_label_reads": len(forbidden),
            "source_bound_readable_reports_checked": len(ids) + 1,
            "inference_manifest_sha256": sha256_file(FOLDER / "inference_manifest.json"),
            "saved_evidence_sha256": sha256_file(evidence_path),
        })
        print(json.dumps(rows[-1]), flush=True)
    OUT.write_text(json.dumps({
        "scope": "all_fixed_test_and_three_non_test_live_positive_human_evidence",
        "status": "label_free_source_to_decision_live_entry_verified",
        "runs": rows}, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
