"""Trace one frozen MGTAB pair contribution to published fact queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from train_v3_computable import attach_mgtab_structural
from verifiable_bot.mgtab_source_certificate import certificate as base_certificate
from verifiable_bot.mgtab_structural_certificate import certificate as structural_certificate
from verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


def trace(data_dir: Path, facts_path: Path, model_dir: Path, seed: int,
          account_index: int, pair_id: str) -> dict:
    directory = model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(
        encoding="utf-8"))
    x, y, ids, names, split, source = load("mgtab", data_dir, None, seed)
    x, names, source = attach_mgtab_structural(x, names, ids, source,
                                              data_dir, facts_path)
    if source["files_sha256"] != report["manifest"]["files_sha256"] or \
            source["split_sha256"] != report["manifest"]["split_sha256"]:
        raise ValueError("published source or fixed split differs from model")
    if account_index not in set(map(int, split["test"])):
        raise ValueError("requested account is not in saved test split")
    row = next((item for item in saved["predictions"]
                if item["account_id"] == ids[account_index]), None)
    if row is None or pair_id not in row.get("pair_contributions", {}):
        raise ValueError("saved test row or pair contribution is absent")
    reference_path = directory / "reference.json.gz"
    if sha256_file(reference_path) != report["manifest"]["reference_artifact_sha256"]:
        raise ValueError("saved reference differs")
    reference = LabelBlindReference.load(reference_path)
    reasoner = PairwiseEvidenceReasoner.from_manifest(report["reasoner"])
    if reference.version != reasoner.reference_version or tuple(names) != reference.names:
        raise ValueError("reference or vocabulary differs")
    pair_index = int(pair_id.removeprefix("pair_"))
    if pair_id != f"pair_{pair_index:02d}" or not 0 <= pair_index < len(reasoner.pairs):
        raise ValueError("pair ID is not in the fixed model vocabulary")
    pair = reasoner.pairs[pair_index]
    if saved["pair_terms"][pair_id]["source_fact_names"] != list(pair):
        raise ValueError("saved pair sources differ from model")
    structural_names = set(json.loads(facts_path.with_suffix(".json").read_text(
        encoding="utf-8"))["feature_names"])
    certificates = []
    for name in pair:
        cert = (structural_certificate(data_dir, facts_path, account_index, name)
                if name in structural_names else
                base_certificate(data_dir, account_index, name))
        if cert["value"] != float(x[account_index, names.index(name)]):
            raise ValueError("source certificate differs from model fact")
        certificates.append(cert)
    evidence = reference.transform(x[[account_index]])
    design = reasoner._design(evidence)
    contribution = float(design[0, len(names) + pair_index] *
                         reasoner.coef[len(names) + pair_index])
    probability = float(reasoner.predict_proba(evidence)[0])
    if abs(contribution - row["pair_contributions"][pair_id]) > 1e-12 or \
            abs(probability - row["probability"]) > 1e-12:
        raise ValueError("published-source pair term differs from saved model output")
    group = reasoner.pair_indices[pair_index][2]
    def without(fact_names: tuple[str, ...]) -> float:
        upper = evidence.upper.copy()
        lower = evidence.lower.copy()
        reliability = evidence.reliability.copy()
        available = evidence.available.copy()
        for name in fact_names:
            column = names.index(name)
            upper[:, column] = 0
            lower[:, column] = 0
            reliability[:, column] = 0
            available[:, column] = False
        masked = EvidenceBatch(upper, lower, reliability, available,
                               evidence.reference_version, evidence.names)
        return float(reasoner.predict_proba(masked)[0])

    removal = {"without_first_fact": without((pair[0],)),
               "without_second_fact": without((pair[1],)),
               "without_both_facts": without(pair)}
    return {
        "dataset": "mgtab", "account_index": account_index,
        "seed": seed, "pair_id": pair_id, "source_fact_names": list(pair),
        "source_certificates": certificates,
        "reference_version": reference.version,
        "directions": {name: {"tail": reasoner.tail[names.index(name)],
                              "sign": int(reasoner.sign[names.index(name)])}
                       for name in pair},
        "signed_singleton_design_values": {
            name: float(design[0, names.index(name)]) for name in pair},
        "pair_basis": float(design[0, len(names) + pair_index]),
        "nonnegative_pair_coefficient": float(reasoner.coef[len(names) + pair_index]),
        "pair_logit_contribution": contribution,
        "group": group,
        "group_logit_contribution": float(row["group_contributions"][group]),
        "model_probability": probability,
        "fixed_model_fact_removal_probabilities": removal,
        "fact_removal_interpretation": "masking evidence inputs in the frozen model; not a source-data causal intervention",
        "status": "source_to_pair_term_to_probability_verified",
        "provenance_limit": "published tensors and paper dimension mapping; no raw tweet/profile semantic verification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account-index", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = trace(args.data_dir, args.structural_facts, args.model_dir,
                   args.seed, args.account_index, args.pair_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"account_index": result["account_index"],
                      "pair_id": result["pair_id"],
                      "pair_logit_contribution": result["pair_logit_contribution"],
                      "model_probability": result["model_probability"],
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
