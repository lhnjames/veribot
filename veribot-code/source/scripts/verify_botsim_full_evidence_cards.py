"""Independently verify all BotSim cards from raw users, activity and replies."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.botsim_adapter import CSV_KEYS, account_features
from verifiable_bot.botsim_native_facts import FACT_NAMES, extract
from verifiable_bot.evidence_v2 import SourceRecord
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def raw_matrix(args: argparse.Namespace) -> tuple[list[str], np.ndarray]:
    users = pd.read_csv(args.account_file)
    with args.account_file.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_users = list(csv.DictReader(stream))
    activity = json.loads(args.activity.read_text(encoding="utf-8"))
    ids, graph_values, trace, counts = extract(raw_users, activity)
    if users["user_id"].astype(str).tolist() != ids:
        raise ValueError("raw users differ from native reply account order")
    with np.load(args.native_graph_facts, allow_pickle=False) as saved:
        if (saved["ids"].astype(str).tolist() != ids
                or saved["names"].astype(str).tolist() != list(FACT_NAMES)
                or not np.array_equal(saved["values"], graph_values)):
            raise ValueError("raw reply graph differs from published fact cache")
    base = []
    for index, uid in enumerate(ids):
        row = users.iloc[index]
        payload = {name: row[name] for name in CSV_KEYS if not pd.isna(row[name])}
        records = {"user": SourceRecord("user", uid, "user_field", payload,
                                         f"Users.csv#row={index + 2}")}
        for kind in ("posts", "comment_1", "comment_2"):
            for ordinal, item in enumerate(activity[uid].get(kind, [])):
                key = f"{kind}:{ordinal}"
                records[key] = SourceRecord(key, uid,
                                            "post" if kind == "posts" else kind,
                                            item, f"user_post_comment.json#{uid}/{kind}/{ordinal}")
        base.append(account_features(records))
    return ids, np.column_stack((np.stack(base), graph_values))


def verify_seed(args: argparse.Namespace, seed: int, ids: list[str],
                reconstructed: np.ndarray, card_manifest: dict) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json")
                       .read_text(encoding="utf-8"))["predictions"]
    cached, y, loaded_ids, names, split, source = load(
        "botsim", args.data_dir, args.account_file, seed,
        args.native_graph_facts)
    if (loaded_ids != ids or source["split_sha256"] !=
            report["manifest"]["split_sha256"]
            or not np.allclose(reconstructed, cached, rtol=1e-5, atol=1e-5)):
        raise ValueError("raw account facts or split differ from selected model")
    reference = LabelBlindReference.load(directory / "reference.json.gz")
    refit = LabelBlindReference(min_context_n=reference.min_context_n,
                                support_prior=reference.support_prior).fit(
                                    reconstructed[split["train"]], names)
    if (refit.version != reference.version
            or reference.version != report["manifest"]["reference_version"]):
        raise ValueError("label-blind reference does not refit from raw train facts")
    reasoner = SignConstrainedReasoner.from_manifest(report["reasoner"])
    test = split["test"]
    evidence = reference.transform(reconstructed[test])
    terms = reasoner._design(evidence) * reasoner.coef
    logits = float(reasoner.intercept) + terms.sum(axis=1)
    probabilities = expit(logits)
    card_path = args.cards_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    if sha256_file(card_path) != card_manifest["runs"][seed]["cards_sha256"]:
        raise ValueError("evidence card archive differs from manifest")
    groups = {name: group for group, members in reasoner.groups.items()
              for name in members}
    max_value = max_evidence = max_term = max_probability = 0.0
    count = 0
    with gzip.open(card_path, "rt", encoding="utf-8") as stream:
        for offset, line in enumerate(stream):
            if offset >= len(test):
                raise ValueError("too many evidence cards")
            card = json.loads(line)
            index = int(test[offset])
            uid = ids[index]
            recorded = saved[offset]
            if (card["account_id"] != uid or card["seed"] != seed
                    or card["evaluation_label"] != int(y[index])
                    or recorded["account_id"] != uid
                    or card["source_hashes"] != card_manifest["source_hashes"]
                    or card["source_level"] !=
                    "raw_user_post_comment_and_native_reply_graph"
                    or card["report_sha256"] != sha256_file(report_path)
                    or card["reference_version"] != reference.version
                    or abs(card["threshold"] - report["manifest"]["threshold"]) > 1e-12
                    or len(card["terms"]) != len(names)):
                raise ValueError("card population, model or vocabulary differs")
            for j, term in enumerate(card["terms"]):
                locator = term["source_locator"]
                if (term["name"] != names[j] or term["group"] != groups[names[j]]
                        or locator["activity_account_key"] != uid
                        or locator["fact_name"] != names[j]
                        or (j < 17 and locator["users_csv_row"] != index + 2)
                        or (j >= 17 and locator["trace_sha256"] !=
                            card_manifest["source_hashes"]["native_trace_sha256"])
                        or term["selected_tail"] != reasoner.tail[j]
                        or term["fixed_direction"] != int(reasoner.sign[j])
                        or abs(term["nonnegative_coefficient"] - reasoner.coef[j]) > 1e-12):
                    raise ValueError("card source locator or trained term differs")
                max_value = max(max_value, abs(term["source_value"] -
                                               reconstructed[index, j]))
                max_evidence = max(max_evidence,
                                   abs(term["upper_tail_score"] - evidence.upper[offset, j]),
                                   abs(term["lower_tail_score"] - evidence.lower[offset, j]),
                                   abs(term["reliability"] - evidence.reliability[offset, j]))
                if term["available"] != bool(evidence.available[offset, j]):
                    raise ValueError("card evidence availability differs")
                max_term = max(max_term,
                               abs(term["logit_contribution"] - terms[offset, j]))
            group_values = {group: float(sum(terms[offset, j] for j, name in enumerate(names)
                                             if groups[name] == group))
                            for group in reasoner.groups}
            if any(abs(card["group_contributions"][group] - group_values[group]) > 1e-12
                   or abs(recorded["group_contributions"][group] - group_values[group]) > 1e-12
                   for group in group_values):
                raise ValueError("card or saved group contribution differs")
            max_probability = max(max_probability,
                                  abs(card["probability"] - probabilities[offset]),
                                  abs(recorded["probability"] - probabilities[offset]))
            if (abs(card["intercept"] - reasoner.intercept) > 1e-12
                    or abs(card["logit"] - logits[offset]) > 1e-12
                    or card["prediction"] !=
                    int(probabilities[offset] >= report["manifest"]["threshold"])):
                raise ValueError("card logit or decision differs")
            count += 1
    if count != len(test) or max(max_value, max_evidence, max_term,
                                 max_probability) > 1e-12:
        raise ValueError("card count or numeric reconstruction differs")
    return {"seed": seed, "test_accounts": count,
            "fact_cells": count * len(names),
            "max_raw_fact_difference": max_value,
            "max_reference_evidence_difference": max_evidence,
            "max_term_contribution_difference": max_term,
            "max_probability_difference": max_probability,
            "reference_refit_from_raw_train": True,
            "status": "verified_all_source_to_logit_cards"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--native-graph-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, required=True)
    args = parser.parse_args()
    card_manifest = json.loads((args.cards_dir / "evidence_cards_manifest.json")
                               .read_text(encoding="utf-8"))
    if (card_manifest["dataset"] != "botsim"
            or card_manifest["source_hashes"]["users_sha256"] !=
            sha256_file(args.account_file)
            or card_manifest["source_hashes"]["activity_sha256"] !=
            sha256_file(args.activity)
            or card_manifest["source_hashes"]["native_facts_sha256"] !=
            sha256_file(args.native_graph_facts)):
        raise ValueError("card manifest and raw source hashes differ")
    ids, matrix = raw_matrix(args)
    result = {"dataset": "botsim", "scope": "all_test_cards_raw_source_recomputation",
              "runs": [verify_seed(args, seed, ids, matrix, card_manifest)
                       for seed in (0, 1, 2)]}
    output = args.cards_dir / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
