"""Evaluate a validation-locked, source-addressable twelve-concept BotSim model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def read_verified_source_cards(source_dir: Path, seed: int, ids: list[str]) -> dict[str, dict]:
    manifest_path = source_dir / "evidence_cards_manifest.json"
    verification_path = source_dir / "verification.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    source_path = source_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    expected = next(row for row in manifest["runs"] if row["seed"] == seed)
    audited = next(row for row in verification["runs"] if row["seed"] == seed)
    if (expected["cards_sha256"] != sha256_file(source_path)
            or audited["status"] != "verified_all_source_to_logit_cards"
            or audited["test_accounts"] != len(ids)
            or audited["fact_cells"] != 22 * len(ids)):
        raise ValueError("full-population raw-source certificate differs")
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        cards = [json.loads(line) for line in stream]
    if [card["account_id"] for card in cards] != ids:
        raise ValueError("source-card order differs from fixed test population")
    return {card["account_id"]: card for card in cards}


def masked(evidence: EvidenceBatch, columns: list[int]) -> EvidenceBatch:
    upper = evidence.upper.copy()
    lower = evidence.lower.copy()
    upper[:, columns] = 0.0
    lower[:, columns] = 0.0
    return EvidenceBatch(upper, lower, evidence.reliability, evidence.available,
                         evidence.reference_version, evidence.names)


def run(args: argparse.Namespace, seed: int, lock: dict, screen: dict) -> dict:
    original_dir = args.full_model_dir / f"seed{seed}"
    original = json.loads((original_dir / "report.json").read_text(encoding="utf-8"))
    x, y, ids, names, split, source = load(
        "botsim", args.data_dir, args.account_file, seed, args.native_graph_facts)
    if (source["split_sha256"] != original["manifest"]["split_sha256"]
            or source["files_sha256"] != original["manifest"]["files_sha256"]
            or names != original["manifest"]["feature_names"]):
        raise ValueError("raw facts or fixed population differ from audited model")
    reference_path = original_dir / "reference.json.gz"
    if sha256_file(reference_path) != original["manifest"]["reference_artifact_sha256"]:
        raise ValueError("reference artifact differs")
    reference = LabelBlindReference.load(reference_path)
    if reference.version != original["manifest"]["reference_version"]:
        raise ValueError("reference version differs")
    train, dev, test = split["train"], split["val"], split["test"]
    atomic_train = reference.transform(x[train])
    atomic_dev = reference.transform(x[dev])
    atomic_test = reference.transform(x[test])
    groups = original["reasoner"]["groups"]
    concept_train, _ = collapse_group_tails(atomic_train, groups)
    concept_dev, _ = collapse_group_tails(atomic_dev, groups)
    concept_test, winners = collapse_group_tails(atomic_test, groups)
    if len(concept_train.names) != 12 or not (
            concept_train.reference_version == concept_dev.reference_version ==
            concept_test.reference_version):
        raise ValueError("twelve-concept contract differs")
    concept_groups = {group: [f"{group}:high_tail_max", f"{group}:low_tail_max"]
                      for group in groups}
    a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                            stratify=y[train], random_state=seed)
    model = MonotoneSplineReasoner(concept_groups, l2=lock["selected_l2"])
    model.fit(rows(concept_train, a), y[train][a],
              rows(concept_train, b), y[train][b])
    dev_probability = model.predict_proba(concept_dev)
    threshold = best_validation_threshold(dev_probability, y[dev])
    validation = evaluate(y[dev], dev_probability, threshold)
    frozen_run = next(row for row in screen["runs"] if row["seed"] == seed)
    expected = frozen_run["alternatives"][str(lock["selected_l2"])]
    for key in ("acc", "macro_f1", "auroc"):
        if abs(validation[key] - expected["validation"][key]) > 1e-12:
            raise ValueError(f"validation lock mismatch: {key}")
    if abs(threshold - expected["threshold"]) > 1e-12:
        raise ValueError("validation threshold differs from lock")

    test_probability = model.predict_proba(concept_test)
    design = model._design(concept_test)
    logit = model.intercept + design @ model.coef
    if np.max(np.abs(expit(logit) - test_probability)) > 1e-12:
        raise ValueError("concepts do not reconstruct test probabilities")
    by_concept = np.column_stack([
        design[:, model._slices[name]] @ model.coef[model._slices[name]]
        for name in concept_test.names
    ])
    if np.max(np.abs(model.intercept + by_concept.sum(axis=1) - logit)) > 1e-12:
        raise ValueError("concept logit decomposition differs")
    contribution = model.contributions(concept_test)
    masked_changes = {}
    for group, members in concept_groups.items():
        columns = [concept_test.names.index(name) for name in members]
        masked_prediction = model.predict_proba(masked(concept_test, columns)) >= threshold
        masked_changes[group] = int(np.sum(masked_prediction != (test_probability >= threshold)))
    no_evidence = model.predict_proba(masked(concept_test,
                                              list(range(len(concept_test.names)))))
    if np.max(np.abs(no_evidence - expit(model.intercept))) > 1e-12:
        raise ValueError("hidden bypass of the evidence bottleneck")

    out = args.output_dir / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    test_ids = [ids[int(i)] for i in test]
    source_cards = read_verified_source_cards(args.source_cards_dir, seed, test_ids)
    names_to_index = {name: j for j, name in enumerate(names)}
    cards_path = out / "concept_evidence_cards.jsonl.gz"
    predictions = []
    with cards_path.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                for offset, index in enumerate(test):
                    account_id = ids[int(index)]
                    source_card = source_cards[account_id]
                    atomic_rows = source_card["terms"]
                    if [term["name"] for term in atomic_rows] != names:
                        raise ValueError("audited atomic vocabulary differs")
                    for j, term in enumerate(atomic_rows):
                        if abs(term["source_value"] - float(x[int(index), j])) > 1e-12:
                            raise ValueError("audited source value differs")
                    concepts = []
                    for j, name in enumerate(concept_test.names):
                        winning_index = int(winners[name][offset])
                        source_term = atomic_rows[winning_index] if winning_index >= 0 else None
                        selected_value = float(concept_test.upper[offset, j])
                        if winning_index >= 0:
                            tail = "high" if ":high_tail_" in name else "low"
                            atomic_score = (atomic_test.upper if tail == "high"
                                            else atomic_test.lower)[offset, winning_index]
                            expected_value = np.log1p(atomic_score) * atomic_test.reliability[
                                offset, winning_index]
                            if abs(selected_value - expected_value) > 1e-12:
                                raise ValueError("winning source does not yield concept")
                        concepts.append({
                            "name": name, "available": bool(concept_test.available[offset, j]),
                            "winning_atomic_fact": (names[winning_index]
                                                    if winning_index >= 0 else None),
                            "source_locator": (source_term["source_locator"]
                                               if source_term is not None else None),
                            "raw_source_value": (source_term["source_value"]
                                                 if source_term is not None else None),
                            "concept_value": selected_value,
                            "fixed_direction": int(model.sign[j]),
                            "spline_coefficients": model.coef[model._slices[name]].tolist(),
                            "logit_contribution": float(by_concept[offset, j]),
                        })
                    card = {
                        "dataset": "botsim", "seed": seed, "account_id": account_id,
                        "source_level": source_card["source_level"],
                        "source_hashes": source_card["source_hashes"],
                        "source_certificate_sha256": sha256_file(
                            args.source_cards_dir / f"seed{seed}_evidence_cards.jsonl.gz"),
                        "concept_version": concept_test.reference_version,
                        "intercept": float(model.intercept), "concepts": concepts,
                        "group_contributions": {group: float(value[offset])
                                                for group, value in contribution.items()},
                        "logit": float(logit[offset]),
                        "probability": float(test_probability[offset]),
                        "threshold": float(threshold),
                        "prediction": int(test_probability[offset] >= threshold),
                    }
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
                    predictions.append({
                        "account_id": account_id, "label": int(y[index]),
                        "probability": float(test_probability[offset]), "split": "test",
                        "group_contributions": card["group_contributions"],
                    })
    result = {
        "manifest": {
            "dataset": "botsim", "seed": seed,
            "split_sha256": source["split_sha256"],
            "source_files_sha256": source["files_sha256"],
            "reference_version": reference.version,
            "concept_version": concept_test.reference_version,
            "concept_count": len(concept_test.names),
            "concept_names": list(concept_test.names),
            "selection_lock_sha256": sha256_file(args.lock),
            "validation_screen_sha256": sha256_file(args.screen),
            "threshold": float(threshold),
            "threshold_selected_on": "validation_macro_f1",
            "prior_test_results_on_same_population_already_observed": True,
        },
        "reasoner": model.manifest(),
        "validation": validation,
        "test": evaluate(y[test], test_probability, threshold),
        "group_mask_prediction_changes": masked_changes,
        "all_evidence_mask_probability": float(expit(model.intercept)),
        "test_accounts": len(test),
        "cards_sha256": sha256_file(cards_path),
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": result["manifest"], "predictions": predictions},
        indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "validation": validation, "test": result["test"],
            "group_mask_prediction_changes": masked_changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--account-file", required=True, type=Path)
    parser.add_argument("--native-graph-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--source-cards-dir", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if (screen["dataset"] != "botsim" or lock["dataset"] != "botsim"
            or sha256_file(args.screen) != lock["validation_screen_sha256"]
            or lock["selected_l2"] != 0.01 or screen["test_predictions_accessed"]):
        raise ValueError("validation-only concept selection was not locked")
    print(json.dumps([run(args, seed, lock, screen) for seed in (0, 1, 2)], indent=2))


if __name__ == "__main__":
    main()
