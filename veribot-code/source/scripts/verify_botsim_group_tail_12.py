"""Independently replay BotSim raw facts through twelve concepts to the decision."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def spline_term(value: float, sign: int, knots: list[float],
                coefficients: list[float]) -> float:
    magnitude = float(np.log1p(value))
    previous = 0.0
    total = 0.0
    for knot, coefficient in zip(knots, coefficients):
        total += min(max(magnitude - previous, 0.0), knot - previous) * sign * coefficient
        previous = knot
    return total + max(magnitude - previous, 0.0) * sign * coefficients[-1]


def verify_seed(args: argparse.Namespace, seed: int) -> dict:
    original = json.loads((args.full_model_dir / f"seed{seed}" / "report.json")
                          .read_text(encoding="utf-8"))
    directory = args.output_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    if saved["manifest"] != report["manifest"]:
        raise ValueError("prediction manifest differs from model")
    model = report["reasoner"]
    x, y, ids, names, split, source = load(
        "botsim", args.data_dir, args.account_file, seed, args.native_graph_facts)
    if (source["split_sha256"] != report["manifest"]["split_sha256"]
            or source["files_sha256"] != report["manifest"]["source_files_sha256"]
            or report["manifest"]["concept_count"] != 12
            or sha256_file(args.lock) != report["manifest"]["selection_lock_sha256"]):
        raise ValueError("source, split, selection lock, or concept count differs")
    reference_path = args.full_model_dir / f"seed{seed}" / "reference.json.gz"
    reference = LabelBlindReference.load(reference_path)
    if (sha256_file(reference_path) != original["manifest"]["reference_artifact_sha256"]
            or reference.version != report["manifest"]["reference_version"]):
        raise ValueError("source reference differs")
    test = split["test"]
    atomic = reference.transform(x[test])
    name_index = {name: j for j, name in enumerate(names)}
    groups = original["reasoner"]["groups"]
    card_path = directory / "concept_evidence_cards.jsonl.gz"
    if sha256_file(card_path) != report["cards_sha256"]:
        raise ValueError("concept cards changed")
    with gzip.open(card_path, "rt", encoding="utf-8") as stream:
        cards = [json.loads(line) for line in stream]
    source_card_path = args.source_cards_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    source_manifest = json.loads((args.source_cards_dir / "evidence_cards_manifest.json")
                                 .read_text(encoding="utf-8"))
    source_verification = json.loads((args.source_cards_dir / "verification.json")
                                     .read_text(encoding="utf-8"))
    source_entry = next(row for row in source_manifest["runs"] if row["seed"] == seed)
    source_audit = next(row for row in source_verification["runs"] if row["seed"] == seed)
    if (sha256_file(source_card_path) != source_entry["cards_sha256"]
            or source_audit["status"] != "verified_all_source_to_logit_cards"
            or source_audit["fact_cells"] != 22 * len(test)):
        raise ValueError("audited source-card file differs")
    with gzip.open(source_card_path, "rt", encoding="utf-8") as stream:
        source_cards = [json.loads(line) for line in stream]
    predictions = saved["predictions"]
    if len(cards) != len(test) or len(source_cards) != len(test) or len(predictions) != len(test):
        raise ValueError("card population differs")
    max_concept_difference = 0.0
    max_logit_difference = 0.0
    max_probability_difference = 0.0
    checked_concepts = 0
    for offset, index in enumerate(test):
        card, source_card, prediction = cards[offset], source_cards[offset], predictions[offset]
        uid = ids[int(index)]
        if (uid != card["account_id"] or uid != source_card["account_id"]
                or uid != prediction["account_id"]):
            raise ValueError("test account order differs")
        if card["source_certificate_sha256"] != source_entry["cards_sha256"]:
            raise ValueError("source certificate hash differs")
        if [term["name"] for term in source_card["terms"]] != names:
            raise ValueError("audited atomic vocabulary differs")
        if len(card["concepts"]) != 12:
            raise ValueError("missing concept term")
        expected_logit = float(model["intercept"])
        for concept in card["concepts"]:
            concept_name = concept["name"]
            group, suffix = concept_name.split(":", 1)
            tail = suffix.split("_", 1)[0]
            if group not in groups or tail not in ("high", "low"):
                raise ValueError("concept group or tail differs")
            members = [name_index[name] for name in groups[group]]
            scores = atomic.upper if tail == "high" else atomic.lower
            values = np.log1p(scores[offset, members]) * atomic.reliability[offset, members]
            values = np.where(atomic.available[offset, members], values, -np.inf)
            if np.isfinite(values).any():
                winner = members[int(np.argmax(values))]
                value = float(values.max())
                term = source_card["terms"][winner]
                if (concept["winning_atomic_fact"] != names[winner]
                        or concept["source_locator"] != term["source_locator"]
                        or abs(concept["raw_source_value"] - term["source_value"]) > 1e-12
                        or abs(term["source_value"] - x[int(index), winner]) > 1e-12):
                    raise ValueError("winning atomic source differs")
            else:
                value = 0.0
                if concept["winning_atomic_fact"] is not None:
                    raise ValueError("unavailable concept has a winner")
            difference = abs(value - concept["concept_value"])
            max_concept_difference = max(max_concept_difference, difference)
            if difference > 1e-12:
                raise ValueError("concept value differs")
            direction = model["directions"][concept_name]
            expected_contribution = spline_term(
                value, direction["sign"], model["knots"][concept_name],
                model["coefficients"][concept_name])
            if abs(expected_contribution - concept["logit_contribution"]) > 1e-12:
                raise ValueError("concept contribution differs")
            expected_logit += expected_contribution
            checked_concepts += 1
        logit_difference = abs(expected_logit - card["logit"])
        probability_difference = abs(expit(expected_logit) - card["probability"])
        max_logit_difference = max(max_logit_difference, logit_difference)
        max_probability_difference = max(max_probability_difference, probability_difference)
        if (logit_difference > 1e-12 or probability_difference > 1e-12
                or abs(card["probability"] - prediction["probability"]) > 1e-12
                or int(y[index]) != int(prediction["label"])
                or int(card["probability"] >= report["manifest"]["threshold"]) !=
                card["prediction"]):
            raise ValueError("card does not reconstruct fixed prediction")
    return {"seed": seed, "test_accounts": len(test),
            "cards_sha256": sha256_file(card_path),
            "model_report_sha256": sha256_file(directory / "report.json"),
            "source_cards_sha256": sha256_file(source_card_path),
            "concept_terms": checked_concepts,
            "max_concept_difference": max_concept_difference,
            "max_logit_difference": max_logit_difference,
            "max_probability_difference": max_probability_difference,
            "status": "all_source_to_concept_to_decision_cards_replayed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--account-file", required=True, type=Path)
    parser.add_argument("--native-graph-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--source-cards-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = {"dataset": "botsim", "scope": "independent_twelve_concept_replay",
              "runs": [verify_seed(args, seed) for seed in (0, 1, 2)]}
    path = args.output_dir / "independent_verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
