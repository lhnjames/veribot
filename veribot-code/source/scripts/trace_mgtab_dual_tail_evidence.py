"""Trace a frozen dual-tail MGTAB decision from source facts to logit terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from train_v3_computable import attach_mgtab_structural, attach_mgtab_neighbor
from verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.mgtab_source_certificate import certificate as base_certificate
from verifiable_bot.mgtab_structural_certificate import certificate as structural_certificate
from verifiable_bot.mgtab_neighbor_certificate import certificate as neighbor_certificate
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def trace(data_dir: Path, structural_path: Path, model_dir: Path, seed: int,
          account_index: int, source_top_k: int = 5,
          neighbor_path: Path | None = None) -> dict:
    directory = model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    x, y, ids, names, split, current = load("mgtab", data_dir, None, seed)
    x, names, current = attach_mgtab_structural(
        x, names, ids, current, data_dir, structural_path)
    x, names, neighbor_supports, current = attach_mgtab_neighbor(
        x, names, ids, current, data_dir, neighbor_path)
    manifest = report["manifest"]
    if current["files_sha256"] != manifest["files_sha256"] or \
            current["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("published source or split differs from saved model")
    if account_index not in set(map(int, split["test"])):
        raise ValueError("account is absent from frozen test split")
    row = next((item for item in saved["predictions"]
                if item["account_id"] == ids[account_index]), None)
    if row is None or row["label"] != int(y[account_index]):
        raise ValueError("saved account row differs from fixed split")
    reference_path = directory / "reference.json.gz"
    if sha256_file(reference_path) != manifest["reference_artifact_sha256"]:
        raise ValueError("reference artifact differs from saved manifest")
    reference = LabelBlindReference.load(reference_path)
    reasoner = DualTailEvidenceReasoner.from_manifest(report["reasoner"])
    if reference.version != reasoner.reference_version or tuple(names) != reference.names:
        raise ValueError("reference, reasoner or fact vocabulary differs")
    evidence = evidence_with_neighbor_support(
        reference, x[[account_index]], None if neighbor_supports is None else
        neighbor_supports[[account_index]])
    design = reasoner._design(evidence)
    width = len(names)
    terms = []
    for j, name in enumerate(names):
        high = float(design[0, j] * reasoner.coef[j])
        low = float(design[0, width + j] * reasoner.coef[width + j])
        terms.append({
            "fact_name": name, "source_value": (
                float(x[account_index, j]) if np.isfinite(x[account_index, j]) else None),
            "available": bool(evidence.available[0, j]),
            "reliability": float(evidence.reliability[0, j]),
            "upper_tail_score": float(evidence.upper[0, j]),
            "lower_tail_score": float(evidence.lower[0, j]),
            "upper_direction": int(reasoner.tail_sign[j, 0]),
            "lower_direction": int(reasoner.tail_sign[j, 1]),
            "upper_nonnegative_coefficient": float(reasoner.coef[j]),
            "lower_nonnegative_coefficient": float(reasoner.coef[width + j]),
            "upper_logit_contribution": high,
            "lower_logit_contribution": low,
            "total_logit_contribution": high + low,
        })
    logit_from_terms = float(reasoner.intercept + sum(
        item["total_logit_contribution"] for item in terms))
    probability = float(reasoner.predict_proba(evidence)[0])
    if abs(probability - float(row["probability"])) > 1e-12 or \
            abs(logit_from_terms - float(reasoner.logits(evidence)[0])) > 1e-12:
        raise ValueError("fact terms do not reconstruct saved model prediction")
    groups = reasoner.contributions(evidence)
    if any(abs(float(groups[group][0]) - saved_value) > 1e-12
           for group, saved_value in row["group_contributions"].items()):
        raise ValueError("fact terms disagree with saved group contributions")
    if source_top_k < 0:
        raise ValueError("source_top_k must be nonnegative")
    structural_names = set(json.loads(structural_path.with_suffix(".json").read_text(
        encoding="utf-8"))["feature_names"])
    neighbor_names = (set(json.loads(neighbor_path.with_suffix(".json").read_text(
        encoding="utf-8"))["feature_names"]) if neighbor_path is not None else set())
    ranking = sorted(range(width), key=lambda j: (
        -abs(terms[j]["total_logit_contribution"]), names[j]))[:source_top_k]
    source_certificates = []
    for j in ranking:
        name = names[j]
        cert = (neighbor_certificate(data_dir, neighbor_path, account_index, name)
                if name in neighbor_names else
                structural_certificate(data_dir, structural_path, account_index, name)
                if name in structural_names else
                base_certificate(data_dir, account_index, name))
        if (cert["value"] is None) != (terms[j]["source_value"] is None) or \
                (cert["value"] is not None and not np.isclose(
                    cert["value"], terms[j]["source_value"], rtol=0, atol=1e-7)):
            raise ValueError(f"published source differs from model fact: {name}")
        source_certificates.append(cert)
    return {
        "dataset": "mgtab", "seed": seed, "account_index": account_index,
        "account_id": ids[account_index],
        "source_level": "published_feature_and_typed_edge_tensors",
        "source_files_sha256": {name: sha256_file(data_dir / name)
                                for name in ("features.pt", "edge_index.pt", "edge_type.pt")},
        "reference_version": reference.version,
        "intercept": float(reasoner.intercept),
        "fact_terms": terms,
        "source_certificates_for_largest_absolute_terms": source_certificates,
        "group_logit_contributions": {name: float(value[0])
                                      for name, value in groups.items()},
        "reconstructed_logit": logit_from_terms,
        "reconstructed_probability": probability,
        "saved_probability": float(row["probability"]),
        "threshold": float(manifest["threshold"]),
        "status": "published_source_to_fact_to_reference_to_logit_verified",
        "scope": "source and fixed-model arithmetic; no independent behavior-semantic label",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account-index", type=int, required=True)
    parser.add_argument("--source-top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = trace(args.data_dir, args.structural_facts, args.model_dir,
                   args.seed, args.account_index, args.source_top_k,
                   args.neighbor_facts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"account_index": result["account_index"],
                      "fact_terms": len(result["fact_terms"]),
                      "source_certificates": len(result[
                          "source_certificates_for_largest_absolute_terms"]),
                      "probability": result["reconstructed_probability"],
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
