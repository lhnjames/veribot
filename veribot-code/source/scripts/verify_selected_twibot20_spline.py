"""Independently reconstruct selected spline test probabilities from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner
from gate0_shallow_tree import sha256_file


def verify(facts_path: Path, screen_dir: Path, model_dir: Path, seed: int) -> dict:
    selected = model_dir / f"seed{seed}"
    source = screen_dir / f"seed{seed}"
    report = json.loads((selected / "report.json").read_text(encoding="utf-8"))
    predictions = json.loads((selected / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if manifest["facts_sha256"] != sha256_file(facts_path) or \
            manifest["validation_report_sha256"] != sha256_file(source / "validation_report.json") or \
            manifest["reference_artifact_sha256"] != sha256_file(source / "reference.json.gz"):
        raise ValueError("facts, validation selection, or reference changed")
    reference = LabelBlindReference.load(source / "reference.json.gz")
    model = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    if model.reference_version != reference.version:
        raise ValueError("model reference version differs")
    with np.load(facts_path, allow_pickle=False) as cache:
        x = cache["values"].astype(float)
        y = cache["labels"].astype(int)
        ids = cache["ids"].astype(str)
        split = cache["split"].astype(str)
        contexts = cache["contexts"].astype(str)
    test = np.flatnonzero(split == "test")
    evidence = reference.transform(x[test], contexts=contexts[test])
    actual = model.predict_proba(evidence)
    expected = predictions["predictions"]
    if len(expected) != len(test) or any(
        row["account_id"] != str(ids[i]) or row["label"] != int(y[i])
        for row, i in zip(expected, test)
    ):
        raise ValueError("test account order or labels differ")
    difference = float(np.max(np.abs(actual - np.array(
        [row["probability"] for row in expected]))))
    if difference > 1e-12:
        raise ValueError("saved test probability differs from reconstructed model")
    groups = model.contributions(evidence)
    if any(set(row.get("group_contributions", {})) != set(groups)
           for row in expected):
        raise ValueError("saved evidence group vocabulary differs")
    group_difference = float(max(np.max(np.abs(value - np.asarray([
        row["group_contributions"][name] for row in expected])))
        for name, value in groups.items()))
    if group_difference > 1e-12:
        raise ValueError("saved evidence groups differ from model")
    saved_logits = float(model.intercept) + sum(np.asarray([
        row["group_contributions"][name] for row in expected]) for name in groups)
    logit_difference = float(np.max(np.abs(saved_logits - model.logits(evidence))))
    if logit_difference > 1e-12:
        raise ValueError("saved evidence groups do not reconstruct logits")
    masked = EvidenceBatch(np.zeros_like(evidence.upper), np.zeros_like(evidence.lower),
                           np.zeros_like(evidence.reliability),
                           np.zeros_like(evidence.available, dtype=bool),
                           evidence.reference_version, evidence.names)
    masked_difference = float(np.max(np.abs(model.logits(masked) - model.intercept)))
    if masked_difference > 1e-12:
        raise ValueError("classifier retains a path after all evidence is masked")
    return {"seed": seed, "test_accounts": len(test),
            "max_probability_difference": difference,
            "max_group_contribution_difference": group_difference,
            "max_logit_reconstruction_difference": logit_difference,
            "max_all_evidence_masked_logit_difference": masked_difference,
            "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    result = [verify(args.facts, args.screen_dir, args.model_dir, seed)
              for seed in args.seeds]
    (args.model_dir / "artifact_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
