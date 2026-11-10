"""Independently replay all Fox8 source facts through 20 concepts to decisions."""

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
from verify_botsim_group_tail_12 import spline_term
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def verify_seed(args: argparse.Namespace, seed: int, groups: dict) -> dict:
    directory = args.output_dir / f"seed{seed}"
    original_dir = args.full_model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    original = json.loads((original_dir / "report.json").read_text(encoding="utf-8"))
    prediction_file = json.loads((directory / "test_predictions.json")
                                 .read_text(encoding="utf-8"))
    if prediction_file["manifest"] != report["manifest"]:
        raise ValueError("saved predictions and report differ")
    x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
    with np.load(args.native_facts, allow_pickle=False) as cache:
        if cache["ids"].astype(str).tolist() != ids:
            raise ValueError("native account population differs")
        x = np.column_stack((x, cache["values"].astype(float)))
        names += cache["names"].astype(str).tolist()
    if (source["split_sha256"] != report["manifest"]["split_sha256"]
            or source["files_sha256"] != report["manifest"]["source_files_sha256"]
            or sha256_file(args.native_facts) != report["manifest"]["native_facts_sha256"]
            or report["manifest"]["concept_count"] != 20
            or sha256_file(args.lock) != report["manifest"]["selection_lock_sha256"]):
        raise ValueError("frozen source, split, selection or concept count differs")
    reference_path = original_dir / "reference.json.gz"
    reference = LabelBlindReference.load(reference_path)
    if (sha256_file(reference_path) != original["manifest"]["reference_artifact_sha256"]
            or reference.version != report["manifest"]["reference_version"]):
        raise ValueError("label-blind reference differs")
    test = split["test"]
    atomic = reference.transform(x[test])
    name_index = {name: j for j, name in enumerate(names)}
    model = report["reasoner"]
    card_path = directory / "concept_evidence_cards.jsonl.gz"
    if sha256_file(card_path) != report["cards_sha256"]:
        raise ValueError("concept cards changed")
    with gzip.open(card_path, "rt", encoding="utf-8") as stream:
        cards = [json.loads(line) for line in stream]
    source_manifest = json.loads((args.source_cards_dir / "evidence_cards_manifest.json")
                                 .read_text(encoding="utf-8"))
    source_verification = json.loads((args.source_cards_dir / "verification.json")
                                     .read_text(encoding="utf-8"))
    source_entry = next(row for row in source_manifest["runs"] if row["seed"] == seed)
    source_audit = next(row for row in source_verification["runs"] if row["seed"] == seed)
    source_path = args.source_cards_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    if (sha256_file(source_path) != source_entry["cards_sha256"]
            or source_audit["status"] != "verified_all_source_to_logit_cards"
            or source_audit["fact_cells"] != 43 * len(test)):
        raise ValueError("original raw-source audit differs")
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        source_cards = [json.loads(line) for line in stream]
    predictions = prediction_file["predictions"]
    if not (len(cards) == len(source_cards) == len(predictions) == len(test)):
        raise ValueError("test account population differs")
    max_concept = max_logit = max_probability = 0.0
    terms_checked = 0
    for offset, index in enumerate(test):
        uid = ids[int(index)]
        card, raw, saved = cards[offset], source_cards[offset], predictions[offset]
        if (uid != card["account_id"] or uid != raw["account_id"]
                or uid != saved["account_id"]
                or card["source_certificate_sha256"] != source_entry["cards_sha256"]
                or [term["name"] for term in raw["terms"]] != names
                or [term["name"] for term in card["concepts"]] !=
                report["manifest"]["concept_names"]):
            raise ValueError("account or concept vocabulary differs")
        logit = float(model["intercept"])
        for concept in card["concepts"]:
            name = concept["name"]
            group, suffix = name.split(":", 1)
            tail = suffix.split("_", 1)[0]
            if group not in groups or tail not in ("high", "low"):
                raise ValueError("source group or concept tail differs")
            members = [name_index[fact] for fact in groups[group]]
            scores = atomic.upper if tail == "high" else atomic.lower
            values = np.log1p(scores[offset, members]) * atomic.reliability[offset, members]
            values = np.where(atomic.available[offset, members], values, -np.inf)
            if np.isfinite(values).any():
                winner = members[int(np.argmax(values))]
                expected_value = float(np.max(values))
                winning_source = raw["terms"][winner]
                if (concept["winning_atomic_fact"] != names[winner]
                        or concept["source_locator"] != winning_source["source_locator"]
                        or winning_source["source_value"] is None
                        or abs(concept["raw_source_value"] -
                               winning_source["source_value"]) > 1e-12
                        or abs(winning_source["source_value"] -
                               x[int(index), winner]) > 1e-12):
                    raise ValueError("winning raw source differs")
            else:
                expected_value = 0.0
                if concept["winning_atomic_fact"] is not None:
                    raise ValueError("unavailable concept has a winner")
            concept_difference = abs(expected_value - concept["concept_value"])
            max_concept = max(max_concept, concept_difference)
            if concept_difference > 1e-12:
                raise ValueError("concept computation differs")
            direction = model["directions"][name]
            contribution = spline_term(expected_value, direction["sign"],
                                       model["knots"][name],
                                       model["coefficients"][name])
            if abs(contribution - concept["logit_contribution"]) > 1e-12:
                raise ValueError("concept logit contribution differs")
            logit += contribution
            terms_checked += 1
        logit_difference = abs(logit - card["logit"])
        probability_difference = abs(expit(logit) - card["probability"])
        max_logit = max(max_logit, logit_difference)
        max_probability = max(max_probability, probability_difference)
        if (logit_difference > 1e-12 or probability_difference > 1e-12
                or abs(saved["probability"] - card["probability"]) > 1e-12
                or saved["label"] != int(y[index])
                or card["prediction"] != int(card["probability"] >= card["threshold"])):
            raise ValueError("concepts do not reconstruct fixed decision")
    return {"seed": seed, "test_accounts": len(test),
            "cards_sha256": sha256_file(card_path),
            "model_report_sha256": sha256_file(directory / "report.json"),
            "source_cards_sha256": sha256_file(source_path),
            "concept_terms": terms_checked,
            "max_concept_difference": max_concept,
            "max_logit_difference": max_logit,
            "max_probability_difference": max_probability,
            "status": "all_source_to_concept_to_decision_cards_replayed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--native-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--source-cards-dir", required=True, type=Path)
    parser.add_argument("--groups-json", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    groups = json.loads(args.groups_json.read_text(encoding="utf-8"))
    result = {"dataset": "fox8", "scope": "independent_twenty_concept_replay",
              "runs": [verify_seed(args, seed, groups) for seed in (0, 1, 2)]}
    path = args.output_dir / "independent_verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
