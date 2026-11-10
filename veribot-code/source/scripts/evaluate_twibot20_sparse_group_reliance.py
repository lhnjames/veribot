"""Measure fixed sparse model reliance on each source-backed evidence group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import groups as base_groups, sha256
from verifiable_bot.group_reliance import evaluate_group_reliance
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--active-cache", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.model_dir / "report.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    saved = json.loads((args.model_dir / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    if (manifest["base_facts_sha256"] != sha256(args.base_facts)
            or manifest["model_artifact_sha256"] != sha256(args.artifact)
            or manifest["reference_artifact_sha256"] != sha256(args.reference)
            or manifest["active_source_cache_sha256"] != sha256(args.active_cache)
            or manifest["active_source_audit_sha256"] != sha256(args.active_audit)):
        raise ValueError("source-backed model artifact differs")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        split = cache["split"].astype(str)
        labels = cache["labels"].astype(int)
        contexts = cache["contexts"].astype(str)
        values = cache["values"].astype(float)
        names = cache["names"].astype(str).tolist()
    test = np.flatnonzero(split == "test")
    with np.load(args.artifact, allow_pickle=False) as model:
        coefficient = model["coefficient"].astype(float)
        intercept = float(model["intercept"][0])
        threshold = float(model["threshold"][0])
    with np.load(args.active_cache, allow_pickle=False) as cache:
        if not np.array_equal(ids, cache["ids"].astype(str)):
            raise ValueError("active source account population differs")
        active = cache["active_indices"].astype(int)
        lexical = cache["values"][test].astype(float) @ coefficient[44 + active]
    reference = LabelBlindReference.load(args.reference)
    evidence = reference.transform(values[test], contexts=contexts[test])
    upper = np.log1p(evidence.upper) * evidence.reliability
    lower = np.log1p(evidence.lower) * evidence.reliability
    columns = {}
    for group, members in base_groups(names).items():
        index = [names.index(name) for name in members]
        columns[group] = (upper[:, index] @ coefficient[index] +
                          lower[:, index] @ coefficient[np.asarray(index) + 22])
    columns["unigram_evidence"] = lexical
    if (len(saved) != len(test)
            or [row["account_id"] for row in saved] != ids[test].tolist()
            or any(abs(sum(columns[name][k] for name in columns if name !=
                           "unigram_evidence") - row["group_contributions"][
                               "structured_evidence"]) > 1e-10
                   or abs(lexical[k] - row["group_contributions"][
                       "unigram_evidence"]) > 1e-10
                   for k, row in enumerate(saved))):
        raise ValueError("fine source groups differ from saved prediction")
    result = evaluate_group_reliance(
        labels[test], np.asarray([row["probability"] for row in saved]),
        columns, intercept, threshold)
    output = args.model_dir / "group_reliance.json"
    output.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"dataset": "twibot20", "n": result["n"],
                      "original_acc": result["original_acc"],
                      "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
