"""Apply validation-selected TwiBot-20 spline artifacts to official test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner
from train_v3_twibot20 import metric
from gate0_shallow_tree import sha256_file


def run(facts_path: Path, screen_dir: Path, output_dir: Path, seed: int) -> dict:
    source_dir = screen_dir / f"seed{seed}"
    screen_path = source_dir / "validation_report.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    reference_path = source_dir / "reference.json.gz"
    if screen["manifest"]["screen_scope"] != "validation_only" or \
            screen["manifest"]["test_predictions_accessed"] or \
            screen["manifest"]["facts_sha256"] != sha256_file(facts_path) or \
            screen["manifest"]["reference_artifact_sha256"] != sha256_file(reference_path):
        raise ValueError("validation-selected artifacts or source facts changed")
    reference = LabelBlindReference.load(reference_path)
    model = MonotoneSplineReasoner.from_manifest(screen["reasoner"])
    if reference.version != model.reference_version:
        raise ValueError("reference and spline reasoner disagree")
    with np.load(facts_path, allow_pickle=False) as cache:
        x = cache["values"].astype(float)
        y = cache["labels"].astype(int)
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        contexts = cache["contexts"].astype(str)
    test = np.flatnonzero(parts == "test")
    if len(test) != 1183:
        raise ValueError("official TwiBot-20 test size differs")
    evidence = reference.transform(x[test], contexts=contexts[test])
    probability = model.predict_proba(evidence)
    contribution = model.contributions(evidence)
    threshold = float(screen["manifest"]["threshold"])
    report = {
        "manifest": {**screen["manifest"], "evaluation_scope": "official_test",
                     "selected_by": "three_seed_validation_macro_f1",
                     "validation_report_sha256": sha256_file(screen_path)},
        "reasoner": screen["reasoner"], "validation": screen["validation"],
        "test": metric(y[test], probability, threshold),
    }
    out = output_dir / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": report["manifest"],
        "predictions": [{"account_id": str(ids[i]), "label": int(y[i]),
                         "probability": float(probability[k]),
                         "group_contributions": {name: float(value[k])
                                                 for name, value in contribution.items()}}
                        for k, i in enumerate(test)]}, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "validation": screen["validation"], "test": report["test"],
            "report": str(out / "report.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    print(json.dumps({"runs": [run(args.facts, args.screen_dir, args.output, seed)
                                for seed in args.seeds]}, indent=2))


if __name__ == "__main__":
    main()
