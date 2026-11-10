"""Trace one locked TwiBot-20 test decision from official ZIP to logit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.twibot20_source_certificate import certificate
from verifiable_bot.twibot20_neighbor_profile_certificate import certificates as neighbor_certificates
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def trace(archive: Path, facts_path: Path, audit_path: Path,
          lock_path: Path, model_dir: Path, seed: int, account_id: str,
          source_top_k: int, neighbor_path: Path | None = None,
          neighbor_audit_path: Path | None = None) -> dict:
    if source_top_k < 0:
        raise ValueError("source_top_k must be nonnegative")
    directory = model_dir / f"seed{seed}"
    report = json.loads((directory / "evidence_report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (manifest["facts_sha256"] != sha256(facts_path)
            or manifest["source_audit_sha256"] != sha256(audit_path)
            or manifest["selection_lock_sha256"] != sha256(lock_path)
            or manifest["source_archive_sha256"] != sha256(archive)
            or audit["status"] != "verified_all_labeled_account_fact_cells_against_official_zip"
            or lock["selected_model"] not in ("dual_tail_22", "dual_tail_32_neighbor")):
        raise ValueError("locked model, source archive, or population audit differs")
    with np.load(facts_path, allow_pickle=False) as cached:
        ids = cached["ids"].astype(str)
        matches = np.flatnonzero(ids == str(account_id))
        if len(matches) != 1:
            raise ValueError("account absent or duplicated")
        index = int(matches[0])
        if str(cached["split"][index]) != "test":
            raise ValueError("account is outside official test")
        raw = cached["values"][index:index + 1].astype(float)
        names = cached["names"].astype(str).tolist()
        context = cached["contexts"][index:index + 1].astype(str)
        label = int(cached["labels"][index])
    if neighbor_path is not None:
        if neighbor_audit_path is None:
            raise ValueError("neighbor source audit is required")
        with np.load(neighbor_path, allow_pickle=False) as cache:
            if str(cache["ids"][index]) != str(account_id):
                raise ValueError("neighbor fact account differs")
            raw = np.column_stack((raw, cache["values"][index:index + 1].astype(float)))
            names += cache["names"].astype(str).tolist()
        if (manifest.get("neighbor_facts_sha256") != sha256(neighbor_path)
                or manifest.get("neighbor_source_audit_sha256") !=
                sha256(neighbor_audit_path)):
            raise ValueError("neighbor facts or source audit differ")
    elif lock["selected_model"] != "dual_tail_22":
        raise ValueError("selected model requires neighbor facts")
    row = next((item for item in saved["predictions"]
                if item["account_id"] == str(account_id)), None)
    if row is None or row["label"] != label:
        raise ValueError("saved prediction differs from official test row")
    reference_path = directory / "reference.json.gz"
    if sha256(reference_path) != manifest["reference_artifact_sha256"]:
        raise ValueError("reference artifact differs")
    reference = LabelBlindReference.load(reference_path)
    reasoner = DualTailEvidenceReasoner.from_manifest(report["reasoner"])
    if tuple(names) != reference.names or reference.version != reasoner.reference_version:
        raise ValueError("reference or vocabulary differs")
    evidence = reference.transform(raw, context)
    design = reasoner._design(evidence)
    width = len(names)
    terms = []
    for j, name in enumerate(names):
        high = float(design[0, j] * reasoner.coef[j])
        low = float(design[0, width + j] * reasoner.coef[width + j])
        terms.append({
            "fact_name": name,
            "source_value": float(raw[0, j]) if np.isfinite(raw[0, j]) else None,
            "available": bool(evidence.available[0, j]),
            "upper_tail_score": float(evidence.upper[0, j]),
            "lower_tail_score": float(evidence.lower[0, j]),
            "reliability": float(evidence.reliability[0, j]),
            "upper_direction": int(reasoner.tail_sign[j, 0]),
            "lower_direction": int(reasoner.tail_sign[j, 1]),
            "upper_nonnegative_coefficient": float(reasoner.coef[j]),
            "lower_nonnegative_coefficient": float(reasoner.coef[width + j]),
            "upper_logit_contribution": high,
            "lower_logit_contribution": low,
            "total_logit_contribution": high + low,
        })
    reconstructed_logit = float(reasoner.intercept + sum(
        term["total_logit_contribution"] for term in terms))
    probability = float(reasoner.predict_proba(evidence)[0])
    groups = {name: float(value[0])
              for name, value in reasoner.contributions(evidence).items()}
    if (abs(reconstructed_logit - float(reasoner.logits(evidence)[0])) > 1e-12
            or abs(probability - row["probability"]) > 1e-12
            or any(abs(value - row["group_contributions"][name]) > 1e-12
                   for name, value in groups.items())):
        raise ValueError("fact terms do not reconstruct saved decision")
    ranking = sorted(range(width), key=lambda j: (
        -abs(terms[j]["total_logit_contribution"]), names[j]))[:source_top_k]
    neighbor_names = [names[j] for j in ranking if names[j] not in names[:22]]
    neighbor_by_name = {item["fact_name"]: item for item in
                        neighbor_certificates(archive, neighbor_path,
                                              str(account_id), neighbor_names)} \
        if neighbor_names else {}
    certificates = []
    for j in ranking:
        cert = (neighbor_by_name[names[j]] if names[j] in neighbor_by_name
                else certificate(archive, facts_path, str(account_id), names[j]))
        if cert["status"] != "verified" or cert["source_value"] != terms[j]["source_value"]:
            raise ValueError(f"raw ZIP fact differs: {names[j]}")
        certificates.append(cert)
    return {
        "dataset": "twibot20", "seed": seed, "account_id": str(account_id),
        "source_level": "official_raw_zip_record_and_native_neighbor_relation",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "fact_cache_sha256": manifest["facts_sha256"],
        "full_population_source_audit_sha256": manifest["source_audit_sha256"],
        "reference_version": reference.version,
        "intercept": float(reasoner.intercept),
        "fact_terms": terms,
        "source_certificates_for_largest_absolute_terms": certificates,
        "group_logit_contributions": groups,
        "reconstructed_logit": reconstructed_logit,
        "reconstructed_probability": probability,
        "saved_probability": row["probability"],
        "threshold": float(manifest["threshold"]),
        "predicted_label": int(probability >= manifest["threshold"]),
        "true_label": label,
        "status": "official_source_to_fact_to_reference_to_logit_verified",
        "scope": "source and fixed-model arithmetic; semantic truth needs external assessment",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--source-top-k", type=int, default=5)
    parser.add_argument("--neighbor-facts", type=Path)
    parser.add_argument("--neighbor-source-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = trace(args.zip, args.facts, args.source_audit, args.selection_lock,
                   args.model_dir, args.seed, args.account_id, args.source_top_k,
                   args.neighbor_facts, args.neighbor_source_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"account_id": result["account_id"],
                      "fact_terms": len(result["fact_terms"]),
                      "source_certificates": len(result[
                          "source_certificates_for_largest_absolute_terms"]),
                      "probability": result["reconstructed_probability"],
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
