"""Recompute TwiBot-20 candidate test scores from saved facts and parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import (EvidenceBatch, LabelBlindReference,
                                                SignConstrainedReasoner)
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--neighbor-facts", type=Path)
    parser.add_argument("--neighbor-source-audit", type=Path)
    parser.add_argument("--token-facts", type=Path)
    parser.add_argument("--token-source-audit", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    with np.load(args.facts, allow_pickle=False) as cached:
        x = cached["values"]
        ids = cached["ids"].astype(str)
        y = cached["labels"]
        split = cached["split"].astype(str)
        contexts = cached["contexts"].astype(str)
        names = cached["names"].astype(str).tolist()
        positions = cached["positions"].astype(int)
    if args.neighbor_facts is not None:
        with np.load(args.neighbor_facts, allow_pickle=False) as cached:
            if (not np.array_equal(ids, cached["ids"].astype(str))
                    or not np.array_equal(split, cached["split"].astype(str))
                    or not np.array_equal(positions, cached["positions"].astype(int))):
                raise ValueError("neighbor evidence account order differs")
            x = np.column_stack((x, cached["values"].astype(float)))
            names += cached["names"].astype(str).tolist()
    if args.token_facts is not None:
        with np.load(args.token_facts, allow_pickle=False) as cached:
            if (not np.array_equal(ids, cached["ids"].astype(str))
                    or not np.array_equal(split, cached["split"].astype(str))
                    or not np.array_equal(positions, cached["positions"].astype(int))):
                raise ValueError("token evidence account order differs")
            x = np.column_stack((x, cached["values"].astype(float)))
            names += cached["names"].astype(str).tolist()
    index = np.flatnonzero(split == "test")
    results = []
    for seed in args.seeds:
        directory = args.runs / f"seed{seed}"
        report = json.loads((directory / "evidence_report.json").read_text(encoding="utf-8"))
        saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
        manifest = report["manifest"]
        if sha256(args.facts) != manifest.get("facts_sha256", manifest.get("base_facts_sha256")):
            raise ValueError("fact table changed after training")
        if "token_facts_sha256" in manifest:
            if (args.token_facts is None or args.token_source_audit is None
                    or sha256(args.token_facts) != manifest["token_facts_sha256"]
                    or sha256(args.token_source_audit) !=
                    manifest["token_source_audit_sha256"]):
                raise ValueError("token fact table or source audit differs")
        if "neighbor_facts_sha256" in manifest:
            if (args.neighbor_facts is None or args.neighbor_source_audit is None
                    or sha256(args.neighbor_facts) != manifest["neighbor_facts_sha256"]
                    or sha256(args.neighbor_source_audit) !=
                    manifest["neighbor_source_audit_sha256"]):
                raise ValueError("neighbor fact table or source audit differs")
        if "selection_lock_sha256" in manifest:
            if args.selection_lock is None or args.source_audit is None or \
                    sha256(args.selection_lock) != manifest["selection_lock_sha256"] or \
                    sha256(args.source_audit) != manifest.get(
                        "source_audit_sha256", manifest.get("token_source_audit_sha256")):
                raise ValueError("dev lock or source audit differs from selected run")
        reference_path = directory / "reference.json.gz"
        if sha256(reference_path) != manifest["reference_artifact_sha256"]:
            raise ValueError("reference artifact changed after training")
        reference = LabelBlindReference.load(reference_path)
        reasoner_payload = report["reasoner"]
        reasoner = (DualTailEvidenceReasoner.from_manifest(reasoner_payload)
                    if reasoner_payload.get("type") == "dual_tail_sign_constrained_additive"
                    else SignConstrainedReasoner.from_manifest(reasoner_payload))
        if reference.version != reasoner.reference_version or \
                reference.names != tuple(names):
            raise ValueError("reference, reasoner or fact vocabulary differs")
        evidence = reference.transform(x[index], contexts[index])
        probability = reasoner.predict_proba(evidence)
        recorded = saved["predictions"]
        if [row["account_id"] for row in recorded] != ids[index].tolist() or \
                [row["label"] for row in recorded] != y[index].tolist():
            raise ValueError("test accounts or labels changed")
        difference = float(np.max(np.abs(probability - np.asarray(
            [row["probability"] for row in recorded]))))
        if difference > 1e-12:
            raise ValueError("saved probability differs from reconstructed score")
        groups = reasoner.contributions(evidence)
        if any(set(row.get("group_contributions", {})) != set(groups)
               for row in recorded):
            raise ValueError("saved evidence group vocabulary differs")
        group_difference = float(max(np.max(np.abs(value - np.asarray([
            row["group_contributions"][name] for row in recorded])))
            for name, value in groups.items()))
        if group_difference > 1e-12:
            raise ValueError("saved evidence group contributions differ")
        saved_logits = float(reasoner.intercept) + sum(np.asarray([
            row["group_contributions"][name] for row in recorded])
            for name in groups)
        logit_difference = float(np.max(np.abs(saved_logits - reasoner.logits(evidence))))
        if logit_difference > 1e-12:
            raise ValueError("saved evidence contributions do not reconstruct logits")
        masked = EvidenceBatch(np.zeros_like(evidence.upper), np.zeros_like(evidence.lower),
                               np.zeros_like(evidence.reliability),
                               np.zeros_like(evidence.available, dtype=bool),
                               evidence.reference_version, evidence.names)
        masked_difference = float(np.max(np.abs(reasoner.logits(masked) -
                                                 reasoner.intercept)))
        if masked_difference > 1e-12:
            raise ValueError("classifier retains a path after all evidence is masked")
        results.append({"seed": seed, "n": len(index),
                        "max_probability_difference": difference,
                        "max_group_contribution_difference": group_difference,
                        "max_logit_reconstruction_difference": logit_difference,
                        "max_all_evidence_masked_logit_difference": masked_difference,
                        "status": "verified"})
    (args.runs / "artifact_verification.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
