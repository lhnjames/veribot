"""Compute exact decision dependence from independently verified additive cards."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit

from gate0_shallow_tree import sha256_file


CONFIG = {
    "botsim": ("output/v3_computable/botsim_group_tail_12",
               "concept_evidence_cards.jsonl.gz", "concepts",
               "all_source_to_concept_to_decision_cards_replayed"),
    "fox8": ("output/v3_fox8_native_spline_l2/group_tail_20_selected",
             "concept_evidence_cards.jsonl.gz", "concepts",
             "all_source_to_concept_to_decision_cards_replayed"),
    "mgtab": ("output/v3_mgtab_ab_spline_content/direct_selected_20",
             "evidence_cards.jsonl.gz", "terms",
             "all_published_tensor_to_evidence_to_decision_cards_replayed"),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decision(logit: float, threshold: float) -> int:
    return int(expit(logit) >= threshold)


def smallest_deletion(logit: float, threshold: float,
                      contributions: list[tuple[str, float]]) -> list[str] | None:
    original = decision(logit, threshold)
    candidates = sorted(
        ((name, value) for name, value in contributions
         if (value > 0 if original else value < 0)),
        key=lambda item: abs(item[1]), reverse=True)
    remaining = logit
    deleted = []
    for name, value in candidates:
        remaining -= value
        deleted.append(name)
        if decision(remaining, threshold) != original:
            return deleted
    return None


def smallest_sufficient(intercept: float, threshold: float, target: int,
                        contributions: list[tuple[str, float]]) -> list[str]:
    if decision(intercept, threshold) == target:
        return []
    candidates = sorted(
        ((name, value) for name, value in contributions
         if (value > 0 if target else value < 0)),
        key=lambda item: abs(item[1]), reverse=True)
    score = intercept
    selected = []
    for name, value in candidates:
        score += value
        selected.append(name)
        if decision(score, threshold) == target:
            return selected
    raise ValueError("full prediction has no same-direction sufficient coalition")


def audit_dataset(root: Path, name: str, config: tuple[str, str, str, str]) -> dict:
    directory_name, card_name, term_key, expected_status = config
    directory = root / directory_name
    verified = read(directory / "independent_verification.json")
    verified_by_seed = {row["seed"]: row for row in verified["runs"]}
    runs = []
    for seed in (0, 1, 2):
        run_dir = directory / f"seed{seed}"
        report = read(run_dir / "report.json")
        saved = read(run_dir / "test_predictions.json")
        cards_path = run_dir / card_name
        if (sha256_file(cards_path) != report["cards_sha256"]
                or verified_by_seed[seed]["status"] != expected_status
                or saved["manifest"] != report["manifest"]):
            raise ValueError(f"{name} seed{seed}: cards or verifier changed")
        predictions = saved["predictions"]
        account_rows = []
        pivotal_counts = Counter()
        with gzip.open(cards_path, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= len(predictions):
                    raise ValueError("more cards than predictions")
                card = json.loads(line)
                prediction = predictions[index]
                if (card["account_id"] != prediction["account_id"]
                        or card["dataset"] != name or card["seed"] != seed
                        or card["threshold"] != report["manifest"]["threshold"]):
                    raise ValueError("card or prediction order differs")
                contributions = [(term["name"], float(term["logit_contribution"]))
                                 for term in card[term_key]]
                if len(contributions) != report["manifest"]["concept_count"]:
                    raise ValueError("evidence count differs")
                intercept = float(card["intercept"])
                logit = intercept + sum(value for _, value in contributions)
                probability = float(expit(logit))
                threshold = float(card["threshold"])
                if (abs(logit - card["logit"]) > 1e-11
                        or abs(probability - card["probability"]) > 1e-11
                        or abs(probability - prediction["probability"]) > 1e-11
                        or decision(logit, threshold) != card["prediction"]):
                    raise ValueError("card does not reconstruct saved decision")
                pivotal = [concept for concept, value in contributions
                           if decision(logit - value, threshold) != card["prediction"]]
                pivotal_counts.update(pivotal)
                deletion = smallest_deletion(logit, threshold, contributions)
                sufficient = smallest_sufficient(intercept, threshold,
                                                  card["prediction"], contributions)
                contribution_map = dict(contributions)
                if deletion is not None and (
                        decision(logit - sum(contribution_map[name] for name in deletion),
                                 threshold) == card["prediction"]
                        or decision(logit - sum(contribution_map[name] for name in deletion[:-1]),
                                    threshold) != card["prediction"]):
                    raise ValueError("support deletion certificate is inconsistent")
                if (decision(intercept + sum(contribution_map[name] for name in sufficient),
                             threshold) != card["prediction"]
                        or (sufficient and decision(
                            intercept + sum(contribution_map[name] for name in sufficient[:-1]),
                            threshold) == card["prediction"])):
                    raise ValueError("sufficient subset certificate is inconsistent")
                account_rows.append({
                    "account_id": card["account_id"],
                    "prediction": card["prediction"],
                    "probability": probability,
                    "intercept_only_prediction": decision(intercept, threshold),
                    "intercept_only_probability": float(expit(intercept)),
                    "evidence_changes_intercept_class": (
                        card["prediction"] != decision(intercept, threshold)),
                    "individual_pivotal_concepts": pivotal,
                    "smallest_support_deletion_to_flip": deletion,
                    "smallest_sufficient_same_direction_subset": sufficient,
                })
        if len(account_rows) != len(predictions) or len(account_rows) != verified_by_seed[seed]["test_accounts"]:
            raise ValueError("card population differs from verified test accounts")
        runs.append({
            "seed": seed, "split_sha256": report["manifest"]["split_sha256"],
            "cards_sha256": report["cards_sha256"], "test_accounts": len(account_rows),
            "individual_pivotal_account_count": sum(
                bool(row["individual_pivotal_concepts"]) for row in account_rows),
            "evidence_changes_intercept_class_count": sum(
                row["evidence_changes_intercept_class"] for row in account_rows),
            "intercept_already_predicts_same_class_count": sum(
                row["prediction"] == row["intercept_only_prediction"]
                for row in account_rows),
            "pivotal_concept_account_counts": dict(pivotal_counts),
            "minimum_support_deletion_to_flip_distribution": dict(Counter(
                str(len(row["smallest_support_deletion_to_flip"]))
                if row["smallest_support_deletion_to_flip"] is not None else "unflippable_by_support_deletion"
                for row in account_rows)),
            "minimum_sufficient_subset_size_distribution": dict(Counter(
                str(len(row["smallest_sufficient_same_direction_subset"]))
                for row in account_rows)),
            "accounts": account_rows,
        })
    return {"dataset": name, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "fixed_additive_model_per_account_evidence_deletion",
        "meaning": "exact input dependence of the fixed classifier, not a causal claim about natural account behavior",
        "datasets": [audit_dataset(args.project_root, name, config)
                     for name, config in CONFIG.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for dataset in result["datasets"]:
        print(dataset["dataset"], [
            {"seed": row["seed"], "accounts": row["test_accounts"],
             "individual_pivotal": row["individual_pivotal_account_count"],
             "deletion_to_flip": row["minimum_support_deletion_to_flip_distribution"]}
            for row in dataset["runs"]])


if __name__ == "__main__":
    main()
