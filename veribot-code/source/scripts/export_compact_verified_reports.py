"""Export immutable reports from all independently source-replayed compact cards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.compact_verified_report import render_verified_card
from verifiable_bot.mgtab_profile_dictionary import PAPER_URL


CONFIG = {
    "botsim": ("output/v3_computable/botsim_group_tail_12",
               "concept_evidence_cards.jsonl.gz",
               "all_source_to_concept_to_decision_cards_replayed"),
    "fox8": ("output/v3_fox8_native_spline_l2/group_tail_20_selected",
             "concept_evidence_cards.jsonl.gz",
             "all_source_to_concept_to_decision_cards_replayed"),
    "mgtab": ("output/v3_mgtab_ab_spline_content/direct_selected_20",
              "evidence_cards.jsonl.gz",
              "all_published_tensor_to_evidence_to_decision_cards_replayed"),
}

SOURCE_BINDINGS = {
    "botsim": ("output/v3_computable/botsim_native_graph/evidence_cards/"
               "seed{seed}_evidence_cards.jsonl.gz", "source_cards_sha256"),
    "fox8": ("output/v3_fox8_native_spline_l2/selected/evidence_cards/"
             "seed{seed}_evidence_cards.jsonl.gz", "source_cards_sha256"),
    "mgtab": ("research/certificates/mgtab_full_source_audit.json",
              "source_audit_sha256"),
}
DEPENDENCE_AUDIT = "research/three_dataset_compact_decision_dependence_2026-11-03.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_dependence(rendered: dict, prior: dict) -> None:
    if rendered["account_id"] != prior["account_id"]:
        raise ValueError("decision-dependence account order differs")
    dependence = rendered["decision_dependence"]
    by_name = {item["evidence_name"]: item["item_id"]
               for item in rendered["all_items"]}
    for new_key, old_key in (
        ("individually_pivotal_item_ids", "individual_pivotal_concepts"),
        ("minimum_support_deletion_to_flip_item_ids",
         "smallest_support_deletion_to_flip"),
        ("minimum_same_direction_sufficient_item_ids",
         "smallest_sufficient_same_direction_subset"),
    ):
        expected = prior[old_key]
        if dependence[new_key] != (None if expected is None else
                                   [by_name[name] for name in expected]):
            raise ValueError(f"independent decision-dependence audit differs: {new_key}")
    if (dependence["intercept_only_prediction"] != prior["intercept_only_prediction"]
            or abs(dependence["intercept_only_probability"] -
                   prior["intercept_only_probability"]) > 1e-11
            or dependence["evidence_changes_intercept_class"] !=
            prior["evidence_changes_intercept_class"]):
        raise ValueError("independent intercept-only audit differs")
    pivotal_ids = set(dependence["individually_pivotal_item_ids"])
    sufficient_ids = set(dependence["minimum_same_direction_sufficient_item_ids"])
    deletion_ids = set(dependence["minimum_support_deletion_to_flip_item_ids"] or ())
    for item in rendered["all_items"]:
        item_id = item["item_id"]
        if (item["individually_pivotal_within_fixed_model"] != (item_id in pivotal_ids)
                or item["in_one_minimum_same_direction_sufficient_set"] !=
                (item_id in sufficient_ids)
                or item["in_minimum_support_deletion_to_flip"] !=
                (item_id in deletion_ids)):
            raise ValueError("displayed decision-role flag differs from audited set")


def export(root: Path, output_dir: Path, dataset: str,
           dependence_audit: dict) -> dict:
    model_dir, filename, status = CONFIG[dataset]
    model = root / model_dir
    verifier_path = model / "independent_verification.json"
    verified = read(verifier_path)
    by_seed = {row["seed"]: row for row in verified["runs"]}
    dependence_dataset = next(row for row in dependence_audit["datasets"]
                              if row["dataset"] == dataset)
    dependence_by_seed = {row["seed"]: row for row in dependence_dataset["runs"]}
    profile_dictionary_hash = None
    if dataset == "mgtab":
        dictionary_path = root / "research/certificates/mgtab_published_profile_dictionary_audit.json"
        dictionary = read(dictionary_path)
        tensor_source = read(root / "research/certificates/mgtab_full_source_audit.json")
        if (dictionary["status"] != "published_profile_dimensions_match_table_a5_types"
                or dictionary["paper_table_a5"] != PAPER_URL
                or dictionary["features_sha256"] !=
                tensor_source["source_sha256"]["features.pt"]
                or len(dictionary["integer_normalization_checks"]) != 3):
            raise ValueError("MGTAB published profile dictionary differs")
        profile_dictionary_hash = digest(dictionary_path)
    rows = []
    for seed in (0, 1, 2):
        run_dir = model / f"seed{seed}"
        report = read(run_dir / "report.json")
        saved = read(run_dir / "test_predictions.json")
        cards_path = run_dir / filename
        cards_hash = digest(cards_path)
        dependence_run = dependence_by_seed[seed]
        source_pattern, source_hash_field = SOURCE_BINDINGS[dataset]
        source_path = root / source_pattern.format(seed=seed)
        source_hash = digest(source_path)
        if (cards_hash != report["cards_sha256"]
                or cards_hash != by_seed[seed].get("cards_sha256")
                or digest(run_dir / "report.json") !=
                by_seed[seed].get("model_report_sha256")
                or source_hash != by_seed[seed].get(source_hash_field)
                or by_seed[seed]["status"] != status
                or saved["manifest"] != report["manifest"]
                or dependence_run["cards_sha256"] != cards_hash
                or dependence_run["split_sha256"] != report["manifest"]["split_sha256"]
                or dependence_run["test_accounts"] != len(saved["predictions"])):
            raise ValueError(f"{dataset} seed{seed}: independent source audit differs")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{dataset}_seed{seed}_verified_reports.jsonl.gz"
        count = 0
        tier_counts: dict[str, int] = {}
        pivotal_accounts = evidence_changes_class_accounts = 0
        with gzip.open(cards_path, "rt", encoding="utf-8") as source, \
                destination.open("wb") as raw, \
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped, \
                io.TextIOWrapper(zipped, encoding="utf-8") as target:
            for line in source:
                if count >= len(saved["predictions"]):
                    raise ValueError("more cards than saved predictions")
                card = json.loads(line)
                expected = saved["predictions"][count]
                if (card["dataset"] != dataset or card["seed"] != seed
                        or card["account_id"] != expected["account_id"]
                        or abs(float(card["probability"]) -
                               float(expected["probability"])) > 1e-10
                        or card["threshold"] != report["manifest"]["threshold"]):
                    raise ValueError("card differs from fixed test prediction")
                rendered = render_verified_card(card)
                _compare_dependence(rendered, dependence_run["accounts"][count])
                pivotal_accounts += bool(rendered["decision_dependence"][
                    "individually_pivotal_item_ids"])
                evidence_changes_class_accounts += rendered["decision_dependence"][
                    "evidence_changes_intercept_class"]
                if len(rendered["all_items"]) != report["manifest"]["concept_count"]:
                    raise ValueError("rendered evidence count differs")
                for item in rendered["all_items"]:
                    tier = item["source_tier"]
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                target.write(json.dumps(rendered, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
                count += 1
        if count != len(saved["predictions"]) or count != by_seed[seed]["test_accounts"]:
            raise ValueError("exported report population differs")
        rows.append({"seed": seed, "accounts": count, "model_cards_sha256": cards_hash,
                     "source_binding_sha256": source_hash,
                     "independent_verification_sha256": digest(verifier_path),
                     "export_sha256": digest(destination), "output": str(destination),
                     "source_tier_item_counts": tier_counts,
                     "individually_pivotal_accounts": pivotal_accounts,
                     "evidence_changes_intercept_class_accounts":
                         evidence_changes_class_accounts})
    result = {"dataset": dataset, "source_audit": status, "runs": rows}
    if profile_dictionary_hash is not None:
        result["published_profile_dictionary_audit_sha256"] = profile_dictionary_hash
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    dependence_path = args.project_root / DEPENDENCE_AUDIT
    dependence = read(dependence_path)
    if dependence.get("scope") != "fixed_additive_model_per_account_evidence_deletion":
        raise ValueError("independent decision-dependence audit scope differs")
    result = {"scope": "all_fixed_test_accounts_immutable_source_and_model_claims",
              "decision_dependence_audit_sha256": digest(dependence_path),
              "datasets": [export(args.project_root, args.output_dir, name, dependence)
                           for name in CONFIG]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({row["dataset"]: [run["accounts"] for run in row["runs"]]
                      for row in result["datasets"]}))


if __name__ == "__main__":
    main()
