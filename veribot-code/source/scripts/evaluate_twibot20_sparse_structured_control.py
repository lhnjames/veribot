"""Same L1 reasoner on structured evidence only for input ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    if (audit["status"] !=
            "verified_all_labeled_account_fact_cells_against_official_zip"
            or audit["fact_cache_sha256"] != sha256(args.base_facts)):
        raise ValueError("base source audit differs")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        labels = cache["labels"].astype(int)
        values = cache["values"].astype(float)
        contexts = cache["contexts"].astype(str)
        names = cache["names"].astype(str).tolist()
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    test = np.flatnonzero(parts == "test")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("structured reference vocabulary differs")
    evidence = reference.transform(values, contexts=contexts)
    structured = np.column_stack((
        np.log1p(evidence.upper) * evidence.reliability,
        np.log1p(evidence.lower) * evidence.reliability))
    model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                               max_iter=1000, random_state=0)
    model.fit(structured[train], labels[train])
    dev_p = expit(float(model.intercept_[0]) +
                  structured[dev] @ model.coef_.ravel())
    threshold = threshold_on_dev(dev_p, labels[dev])
    test_p = expit(float(model.intercept_[0]) +
                   structured[test] @ model.coef_.ravel())
    manifest = {
        "dataset": "twibot20", "official_split": True,
        "source_archive_sha256": audit["source_archive_sha256"],
        "base_facts_sha256": sha256(args.base_facts),
        "base_source_audit_sha256": sha256(args.base_source_audit),
        "reference_version": reference.version,
        "reference_artifact_sha256": sha256(args.reference),
        "reasoner": "l1_logistic_C1_same_as_sparse_unigram_model",
        "input": "44_upper_lower_scores_from_22_structured_facts_only",
        "threshold": threshold,
    }
    report = {"manifest": manifest,
              "dev": metric(labels[dev], dev_p, threshold),
              "test": metric(labels[test], test_p, threshold),
              "intercept": float(model.intercept_[0]),
              "coefficients": model.coef_.ravel().tolist()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "test_predictions.json").write_text(
        json.dumps({"manifest": manifest,
                    "predictions": [
                        {"account_id": ids[int(i)], "label": int(labels[i]),
                         "probability": float(test_p[k])}
                        for k, i in enumerate(test)]}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"dev": report["dev"], "test": report["test"]},
                     indent=2))


if __name__ == "__main__":
    main()
