"""Add exact pair-term contributions to frozen MGTAB v3 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from train_v3_computable import attach_mgtab_structural
from verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def enrich(data_dir: Path, facts_path: Path, model_dir: Path, seed: int) -> dict:
    directory = model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    prediction_path = directory / "test_predictions.json"
    saved = json.loads(prediction_path.read_text(encoding="utf-8"))
    if report["manifest"]["model"] != "computable_evidence_pairwise_same_direction":
        raise ValueError("saved model is not the fixed pairwise reasoner")
    x, y, ids, names, split, current = load("mgtab", data_dir, None, seed)
    x, names, current = attach_mgtab_structural(x, names, ids, current,
                                                data_dir, facts_path)
    if current["files_sha256"] != report["manifest"]["files_sha256"] or \
            current["split_sha256"] != report["manifest"]["split_sha256"]:
        raise ValueError("current source files or split differ from saved model")
    ref_path = directory / "reference.json.gz"
    if sha256_file(ref_path) != report["manifest"]["reference_artifact_sha256"]:
        raise ValueError("saved reference checksum differs")
    reference = LabelBlindReference.load(ref_path)
    reasoner = PairwiseEvidenceReasoner.from_manifest(report["reasoner"])
    if reference.version != reasoner.reference_version or \
            tuple(names) != reference.names:
        raise ValueError("reference or vocabulary differs from saved reasoner")
    test = split["test"]
    evidence = reference.transform(x[test])
    probabilities = reasoner.predict_proba(evidence)
    groups = reasoner.contributions(evidence)
    pairs = reasoner.pair_contributions(evidence)
    recorded = saved["predictions"]
    if len(recorded) != len(test):
        raise ValueError("saved test population differs")
    max_probability_difference = 0.0
    max_group_difference = 0.0
    for row_index, (row, account_index) in enumerate(zip(recorded, test)):
        if row["account_id"] != ids[int(account_index)] or \
                row["label"] != int(y[account_index]):
            raise ValueError("saved test account order or label differs")
        difference = abs(row["probability"] - float(probabilities[row_index]))
        max_probability_difference = max(max_probability_difference, difference)
        for group, contribution in groups.items():
            group_difference = abs(row["group_contributions"][group] -
                                   float(contribution[row_index]))
            max_group_difference = max(max_group_difference, group_difference)
        row["pair_contributions"] = {name: float(column[row_index])
                                      for name, column in pairs.items()}
    if max_probability_difference > 1e-12 or max_group_difference > 1e-12:
        raise ValueError("saved probabilities or group contributions do not replay")
    saved["pair_terms"] = {
        f"pair_{index:02d}": {"source_fact_names": list(pair)}
        for index, pair in enumerate(reasoner.pairs)
    }
    temp_path = prediction_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(prediction_path)
    return {
        "seed": seed, "n_test": len(test),
        "pair_terms": len(pairs),
        "nonzero_pair_coefficients": int(np.count_nonzero(
            reasoner.coef[len(names):] > 0)),
        "max_probability_difference": max_probability_difference,
        "max_group_difference": max_group_difference,
        "prediction_file_sha256": sha256_file(prediction_path),
        "status": "verified_pairwise_trace",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    result = [enrich(args.data_dir, args.structural_facts, args.model_dir, seed)
              for seed in args.seeds]
    (args.model_dir / "pair_trace_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
