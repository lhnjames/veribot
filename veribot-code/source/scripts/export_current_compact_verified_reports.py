"""Bind selected compact-model reports to audited cards and fixed predictions."""

from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.current_compact_verified_report import render_card


CONFIG = {
    "fox8": {
        "folder": "output/v3_fox8_targeted_neighbor_20_concepts",
        "card_name": "ab_group_tree_test_cards.jsonl.gz",
        "prediction_name": "ab_group_tree_test_predictions.json",
        "audit": "research/certificates/fox8_targeted_20_concept_cards.json",
        "expected_status": "all_account_cards_match_sources_and_frozen_model",
        "count_field": "accounts_verified",
    },
    "mgtab": {
        "folder": "output/v3_mgtab_targeted_20_concept_ab_group_tree",
        "card_name": "test_evidence_cards.jsonl.gz",
        "prediction_name": "test_predictions.json",
        "audit": "research/certificates/mgtab_targeted_20_concept_cards.json",
        "expected_status": "all_account_cards_match_frozen_evidence_and_source_level",
        "count_field": "account_cards_checked",
    },
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output/v3_current_compact_verified_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset, config in CONFIG.items():
        folder = root / config["folder"]
        manifest_path = folder / "evidence_cards_manifest.json"
        audit_path = root / config["audit"]
        manifest, audit = read(manifest_path), read(audit_path)
        if (audit["status"] != config["expected_status"]
                or audit["manifest_sha256"] != sha256_file(manifest_path)):
            raise ValueError(f"{dataset} card audit is not bound to the manifest")
        if len(manifest["runs"]) != 3 or len(audit["runs"]) != 3:
            raise ValueError("three fixed seeds are required")
        for seed in (0, 1, 2):
            run_dir = folder / f"seed{seed}"
            card_path = run_dir / config["card_name"]
            prediction_path = run_dir / config["prediction_name"]
            card_sha = sha256_file(card_path)
            manifest_run, audit_run = manifest["runs"][seed], audit["runs"][seed]
            prediction = read(prediction_path)
            if (manifest_run["seed"] != seed or audit_run["seed"] != seed
                    or card_sha != manifest_run["sha256"]
                    or audit_run[config["count_field"]] !=
                    len(prediction["predictions"])):
                raise ValueError("fixed card, audit or prediction population differs")
            destination = out_dir / f"{dataset}_seed{seed}_verified_reports.jsonl.gz"
            count = located = unavailable = absent_shap = mask_flips = group_masks = 0
            with gzip.open(card_path, "rt", encoding="utf-8") as source, \
                    destination.open("wb") as raw, \
                    gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed, \
                    io.TextIOWrapper(packed, encoding="utf-8") as target:
                for line in source:
                    if count >= len(prediction["predictions"]):
                        raise ValueError("more cards than frozen predictions")
                    card = json.loads(line)
                    saved = prediction["predictions"][count]
                    report = render_card(card, dataset, seed)
                    if (report["account_id"] != saved["account_id"]
                            or abs(report["bot_probability"] -
                                   saved["probability"]) > 1e-12
                            or report["threshold"] !=
                            prediction["manifest"]["threshold"]):
                        raise ValueError("report differs from frozen prediction")
                    target.write(json.dumps(report, ensure_ascii=False) + "\n")
                    count += 1
                    for group in report["source_groups"]:
                        group_masks += 1
                        mask_flips += group["changes_class_without_group"]
                        for fact in group["facts"]:
                            located += fact["source_locator"] is not None
                            unavailable += fact["source_locator"] is None
                    absent_shap += report[
                        "unavailable_concepts_with_nonzero_shap_accounting"]
            if count != len(prediction["predictions"]):
                raise ValueError("fewer reports than frozen predictions")
            expected_located = manifest_run["source_locators"]
            if (located != expected_located or located + unavailable != count*20
                    or group_masks != 10*count):
                raise ValueError("source locator or group mask count differs")
            row = {
                "dataset": dataset, "seed": seed, "accounts": count,
                "literal_source_claims": located,
                "unavailable_concept_slots": unavailable,
                "unavailable_slots_with_nonzero_shap_accounting": absent_shap,
                "exact_group_mask_class_changes": mask_flips,
                "audited_card_sha256": card_sha,
                "fixed_predictions_sha256": sha256_file(prediction_path),
                "verified_report_sha256": sha256_file(destination),
                "report_file": str(destination.relative_to(root)).replace("\\", "/"),
            }
            summary.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "scope": "selected_Fox8_MGTAB_literal_source_and_exact_group_mask_reports",
        "reports": summary,
        "source_audit_sha256": {key: sha256_file(root / config["audit"])
                                for key, config in CONFIG.items()},
        "status": "all_fixed_test_reports_bound_to_independently_audited_cards",
    }
    (root / "research/certificates/current_compact_verified_reports.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
