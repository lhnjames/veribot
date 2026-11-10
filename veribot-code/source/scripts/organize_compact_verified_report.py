"""Display one checksum-locked compact report; optional API changes order only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_compact_verified_reports import CONFIG, SOURCE_BINDINGS, DEPENDENCE_AUDIT
from verifiable_bot.evidence_narrator import EvidenceItem, organize_evidence


STABILITY_AUDIT = "research/three_dataset_cross_seed_evidence_stability_2026-11-03.json"
CONTENT_DEPENDENCE_AUDIT = "research/mgtab_selected_content_projection_dependency_2026-11-03.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=("botsim", "fox8", "mgtab"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--use-api", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if sha256(args.project_root / DEPENDENCE_AUDIT) != summary.get(
            "decision_dependence_audit_sha256"):
        raise ValueError("independent decision-dependence audit differs")
    dataset = next(row for row in summary["datasets"]
                   if row["dataset"] == args.dataset)
    if args.dataset == "mgtab":
        dictionary_path = (args.project_root /
                           "research/certificates/mgtab_published_profile_dictionary_audit.json")
        if sha256(dictionary_path) != dataset.get(
                "published_profile_dictionary_audit_sha256"):
            raise ValueError("MGTAB published profile dictionary audit differs")
    run = next(row for row in dataset["runs"] if row["seed"] == args.seed)
    stability = json.loads((args.project_root / STABILITY_AUDIT).read_text(
        encoding="utf-8"))
    if stability.get("decision_dependence_audit_sha256") != summary.get(
            "decision_dependence_audit_sha256"):
        raise ValueError("cross-seed evidence audit differs from decision audit")
    stability_dataset = next(row for row in stability["datasets"]
                             if row["dataset"] == args.dataset)
    if any(stability_dataset["card_sha256_by_seed"][str(item["seed"])] !=
           item["model_cards_sha256"] for item in dataset["runs"]):
        raise ValueError("cross-seed evidence audit differs from verified cards")
    model_dir, card_filename, _ = CONFIG[args.dataset]
    model_path = args.project_root / model_dir
    verifier_path = model_path / "independent_verification.json"
    verified = json.loads(verifier_path.read_text(encoding="utf-8"))
    verified_run = next(row for row in verified["runs"] if row["seed"] == args.seed)
    source_pattern, source_field = SOURCE_BINDINGS[args.dataset]
    source_path = args.project_root / source_pattern.format(seed=args.seed)
    source_hash = sha256(source_path)
    card_path = model_path / f"seed{args.seed}" / card_filename
    report_path = model_path / f"seed{args.seed}" / "report.json"
    if (sha256(verifier_path) != run["independent_verification_sha256"]
            or sha256(card_path) != run["model_cards_sha256"]
            or sha256(card_path) != verified_run["cards_sha256"]
            or sha256(report_path) != verified_run["model_report_sha256"]
            or source_hash != run["source_binding_sha256"]
            or source_hash != verified_run[source_field]):
        raise ValueError("current source, model card, or independent audit differs")
    path = Path(run["output"])
    if sha256(path) != run["export_sha256"]:
        raise ValueError("verified report archive differs from its checksum")
    target = None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["account_id"] == args.account_id:
                target = row
                break
    if target is None or target["dataset"] != args.dataset or target["seed"] != args.seed:
        raise ValueError("requested account is absent from verified report archive")
    content_mask = None
    if args.dataset == "mgtab":
        content_audit = json.loads((args.project_root / CONTENT_DEPENDENCE_AUDIT).read_text(
            encoding="utf-8"))
        content_run = next(item for item in content_audit["runs"]
                           if item["seed"] == args.seed)
        content_account = next(item for item in content_run["accounts"]
                               if item["account_id"] == args.account_id)
        if (content_run["cards_sha256"] != run["model_cards_sha256"]
                or content_account["full_prediction"] != target["prediction"]):
            raise ValueError("MGTAB content dependence differs from locked report")
        named = [item for item in target["all_items"] if item["source_tier"] !=
                 "published_averaged_tweet_tensor_projection"]
        content = [item for item in target["all_items"] if item["source_tier"] ==
                   "published_averaged_tweet_tensor_projection"]
        if len(named) != 4 or len(content) != 16:
            raise ValueError("MGTAB source-tier evidence count differs")
        masked_probability = float(expit(target["intercept"] + math.fsum(
            item["logit_contribution"] for item in named)))
        masked_prediction = int(masked_probability >= target["threshold"])
        if (abs(masked_probability - content_account["named_only_probability"]) > 1e-10
                or masked_prediction != content_account["named_only_prediction"]):
            raise ValueError("MGTAB content mask differs from per-account audit")
        content_mask = {
            "scope": "same_frozen_model_and_threshold_with_16_content_terms_zeroed",
            "published_numeric_profile_or_typed_edge_terms": len(named),
            "published_mean_tweet_projection_terms": len(content),
            "content_masked_probability": masked_probability,
            "content_masked_prediction": masked_prediction,
            "content_projection_changes_class": masked_prediction != target["prediction"],
            "content_dependency_audit_sha256": sha256(
                args.project_root / CONTENT_DEPENDENCE_AUDIT),
        }
    items = [EvidenceItem(item["item_id"], item["claim"],
                          "verified_source_value_and_additive_contribution",
                          (item["item_id"],))
             for item in target["all_items"]]
    by_id = {item["item_id"]: item for item in target["all_items"]}
    dependence = target["decision_dependence"]
    basis = {
        "individually_pivotal": [by_id[item_id] for item_id in
                                  dependence["individually_pivotal_item_ids"]],
        "one_minimum_same_direction_sufficient_set": [by_id[item_id] for item_id in
            dependence["minimum_same_direction_sufficient_item_ids"]],
        "minimum_support_deletion_to_flip": (None if dependence[
            "minimum_support_deletion_to_flip_item_ids"] is None else
            [by_id[item_id] for item_id in dependence[
                "minimum_support_deletion_to_flip_item_ids"]]),
    }
    source_index = {item["item_id"]: item["source_locator"]
                    for item in target["all_items"]}
    organized = organize_evidence("bot" if target["prediction"] else "human",
                                  items, source_index, use_api=args.use_api)
    comparisons = []
    for pair in stability_dataset["pairs"]:
        if pair["account_id"] != args.account_id or args.seed not in pair["seed_pair"]:
            continue
        other_seed = next(seed for seed in pair["seed_pair"] if seed != args.seed)
        comparisons.append({
            "other_test_seed": other_seed,
            "same_prediction": pair["same_prediction"],
            "top_support_comparable": pair["comparable_top_support"],
            "same_comparable_top_support": (
                pair["same_top_support_concept"]
                if pair["comparable_top_support"] else None),
            "both_have_pivotal_term": pair["both_have_pivotal_term"],
            "pivotal_sets_comparable": pair["comparable_pivotal_sets"],
            "same_comparable_nonempty_pivotal_set": (
                pair["same_nonempty_pivotal_set"]
                if pair["both_have_pivotal_term"] and
                pair["comparable_pivotal_sets"] else None),
        })
    comparisons.sort(key=lambda row: row["other_test_seed"])
    basis_status = ("evidence_changes_class_from_intercept"
                    if dependence["evidence_changes_intercept_class"] else
                    "evidence_modifies_score_without_changing_intercept_class")
    output = {"dataset": args.dataset, "seed": args.seed,
              "account_id": args.account_id,
              "probability": target["probability"],
              "report_archive_sha256": run["export_sha256"],
              "source_level": target["source_level"],
              "decision_dependence": target["decision_dependence"],
              "decision_basis_status": basis_status,
              "published_content_projection_mask_audit": content_mask,
              "decision_basis_items": basis,
              "cross_seed_test_evidence": {
                  "scope": "same_account_in_other_frozen_test_splits_only",
                  "comparisons": comparisons,
                  "meaning": ("no repeated test appearance available"
                              if not comparisons else
                              "descriptive model-reason agreement; not external semantic truth"),
              },
              "largest_bot_support_item_ids": target["largest_bot_support_item_ids"],
              "largest_human_support_item_ids": target["largest_human_support_item_ids"],
              "ordered_source_claims": organized,
              "source_index": source_index}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
