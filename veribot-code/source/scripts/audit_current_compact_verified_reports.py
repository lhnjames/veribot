"""Independently compare current readable reports with audited source cards."""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_profile_schema import PROFILE_SCHEMA


CONFIG = {
    "fox8": ("output/v3_fox8_targeted_neighbor_20_concepts",
             "ab_group_tree_test_cards.jsonl.gz"),
    "mgtab": ("output/v3_mgtab_targeted_20_concept_ab_group_tree",
              "test_evidence_cards.jsonl.gz"),
}


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "research/certificates/current_compact_verified_reports.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["reports"]) != 6:
        raise ValueError("all six compact reports are required")
    results = []
    for row in manifest["reports"]:
        dataset, seed = row["dataset"], int(row["seed"])
        folder, card_name = CONFIG[dataset]
        card_path = root / folder / f"seed{seed}" / card_name
        report_path = root / row["report_file"]
        if (sha256_file(card_path) != row["audited_card_sha256"]
                or sha256_file(report_path) != row["verified_report_sha256"]):
            raise ValueError("audited card or rendered report changed")
        account_count = fact_count = no_fact_count = mask_count = 0
        with gzip.open(card_path, "rt", encoding="utf-8") as original, \
                gzip.open(report_path, "rt", encoding="utf-8") as displayed:
            for old_line, new_line in zip(original, displayed, strict=True):
                card, report = json.loads(old_line), json.loads(new_line)
                if ("label" in report or "SHAP" in json.dumps(report.get("source_groups"))
                        or report["account_id"] != card["account_id"]
                        or report["model_prediction"] != card["prediction"]
                        or not _close(report["bot_probability"], card["probability"])
                        or not _close(report["threshold"], card["threshold"])):
                    raise ValueError("readable report adds a label or changes a prediction")
                source = {item["concept"].split(":", 1)[0]: []
                          for item in card["evidence"]}
                for item in card["evidence"]:
                    source[item["concept"].split(":", 1)[0]].append(item)
                if len(source) != 10 or len(report["source_groups"]) != 10:
                    raise ValueError("source group count differs")
                observed_rank = []
                for group in report["source_groups"]:
                    name = group["group"]
                    terms = source[name]
                    if len(terms) != 2 or len(group["facts"]) != 2:
                        raise ValueError("source group does not contain two concepts")
                    old_effect = (card["source_group_zero_mask"][name]
                                  if dataset == "fox8" else card["group_effects"][name])
                    expected_mask = (old_effect["probability"] if dataset == "fox8"
                                     else old_effect["probability_without_group"])
                    expected_flip = (old_effect["changes_class"] if dataset == "fox8"
                                     else old_effect["changes_class_without_group"])
                    if (not _close(group["probability_without_group"], expected_mask)
                            or group["changes_class_without_group"] != expected_flip
                            or not _close(group["probability_difference"],
                                          card["probability"]-expected_mask)):
                        raise ValueError("exact frozen group mask differs")
                    mask_count += 1
                    if abs(group["probability_difference"]) > 1e-12:
                        observed_rank.append((name, abs(group["probability_difference"])))
                    for source_term, shown in zip(terms, group["facts"], strict=True):
                        old_fact = (source_term["winning_source_fact"] if dataset == "fox8"
                                    else source_term["winning_fact"])
                        old_value = (source_term["winning_source_value"] if dataset == "fox8"
                                     else source_term["winning_fact_value"])
                        locator = source_term["source_locator"]
                        if (shown["concept"] != source_term["concept"]
                                or shown["winning_fact"] != old_fact
                                or shown["source_locator"] != locator):
                            raise ValueError("displayed source fact differs")
                        if old_fact is None:
                            no_fact_count += 1
                            if (shown["source_value"] is not None
                                    or shown["source_tier"] != "unavailable"
                                    or not shown["claim"].startswith("No source fact")):
                                raise ValueError("missing fact is presented as evidence")
                        else:
                            fact_count += 1
                            if (not _close(shown["source_value"], old_value)
                                    or "is unavailable" in shown["claim"]):
                                raise ValueError("literal source value differs")
                            if dataset == "mgtab":
                                if shown["source_tier"] != locator["level"]:
                                    raise ValueError("MGTAB source tier differs")
                                if old_fact.startswith("published_profile_coordinate_"):
                                    column = locator["column"]
                                    if (locator["paper_table_a5_alias"] !=
                                            PROFILE_SCHEMA[column][0]
                                            or f"dimension {column+1}" not in shown["claim"]):
                                        raise ValueError("published profile field is misnamed")
                            elif shown["source_tier"] == "unavailable":
                                raise ValueError("present Fox8 source labeled unavailable")
                expected_rank = [name for name, _ in sorted(
                    observed_rank, key=lambda item: (-item[1], item[0]))[:3]]
                if report["largest_three_exact_group_changes"] != expected_rank:
                    raise ValueError("group mask ranking differs")
                expected_summary = " ".join(next(group["mask_claim"]
                                                 for group in report["source_groups"]
                                                 if group["group"] == name)
                                            for name in expected_rank)
                if (report["group_sensitivity_summary"] != expected_summary
                        or ("the Bot class" if card["prediction"] else "the Human class")
                        not in report["decision_statement"]):
                    raise ValueError("readable decision text differs from exact model effects")
                account_count += 1
        if (account_count != row["accounts"]
                or fact_count != row["literal_source_claims"]
                or no_fact_count != row["unavailable_concept_slots"]
                or mask_count != account_count*10):
            raise ValueError("report population or concept counts differ")
        results.append({"dataset": dataset, "seed": seed,
                        "accounts_checked": account_count,
                        "source_literals_checked": fact_count,
                        "missing_facts_checked": no_fact_count,
                        "exact_group_masks_checked": mask_count})
    out = {"report_manifest_sha256": sha256_file(manifest_path),
           "runs": results,
           "status": "all_literal_claims_source_tiers_and_exact_masks_verified"}
    path = root / "research/certificates/current_compact_verified_reports_audit.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
