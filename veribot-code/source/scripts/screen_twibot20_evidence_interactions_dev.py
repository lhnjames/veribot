"""Dev-only screen of source-traceable pair interactions in TwiBot evidence.

Each interaction is the product of two existing evidence terms. The lexical
vocabulary and label-blind reference are frozen from the selected sparse model;
official test rows are never opened here.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--sparse-artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (audit["source_archive_sha256"] != sha256(args.zip)
            or audit["fact_cache_sha256"] != sha256(args.base_facts)):
        raise ValueError("official source or base facts differ")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        values = cache["values"].astype(float)
        contexts = cache["contexts"].astype(str)
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if (not np.array_equal(train, np.arange(len(train)))
            or not np.array_equal(dev, np.arange(len(train), len(train) + len(dev)))):
        raise ValueError("official train/dev order differs")
    documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if (ids[index] != str(row["ID"]) or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("official source position differs")
                    documents.append(" ".join(str(tweet or "")
                                              for tweet in row.get("tweet") or []))
    if len(documents) != len(train) + len(dev):
        raise ValueError("train/dev source count differs")
    reference = LabelBlindReference.load(args.reference)
    evidence = reference.transform(values[:len(documents)],
                                   contexts=contexts[:len(documents)])
    structured = np.column_stack((
        np.log1p(evidence.upper) * evidence.reliability,
        np.log1p(evidence.lower) * evidence.reliability))
    if structured.shape[1] != 44:
        raise ValueError("structured evidence width differs")
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True, dtype=np.float64)
    text_train = vectorizer.fit_transform([documents[i] for i in train])
    text_dev = vectorizer.transform([documents[i] for i in dev])
    with np.load(args.sparse_artifact, allow_pickle=False) as frozen:
        vocabulary = frozen["vocabulary"].astype(str)
        idf = frozen["idf"].astype(float)
        coef = frozen["coefficient"].astype(float)
        intercept = float(frozen["intercept"][0])
        old_threshold = float(frozen["threshold"][0])
    if (not np.array_equal(vectorizer.get_feature_names_out(), vocabulary)
            or not np.array_equal(vectorizer.idf_, idf)
            or coef.size != structured.shape[1] + len(vocabulary)):
        raise ValueError("frozen lexical evidence transform differs")
    active = np.flatnonzero(coef[44:] != 0)
    if len(active) != 109:
        raise ValueError("frozen active lexical term count differs")
    raw_dev = np.asarray(csr_matrix(structured[dev]) @ coef[:44] +
                         text_dev @ coef[44:] + intercept).ravel()
    old_probability = 1 / (1 + np.exp(-raw_dev))
    old_metrics = metric(labels[dev], old_probability, old_threshold)
    if (old_metrics["errors"] != 337 or abs(old_metrics["acc"] -
                                          0.8575052854122621) > 1e-12):
        raise ValueError("selected sparse model failed validation replay")

    lexical = vstack((text_train[:, active], text_dev[:, active]), format="csr")
    structured_pairs = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False).fit_transform(
            csr_matrix(structured))[:, 44:]
    cross_pairs = hstack([
        lexical.multiply(structured[:, j].reshape(-1, 1))
        for j in range(44)
    ], format="csr")
    design = hstack((csr_matrix(structured), lexical,
                     structured_pairs, cross_pairs), format="csr")
    if design.shape[1] != 44 + 109 + 946 + 44 * 109:
        raise ValueError("interaction design width differs")
    scaler = MaxAbsScaler().fit(design[train])
    train_x = scaler.transform(design[train])
    dev_x = scaler.transform(design[dev])
    if (train_x.shape[1] != design.shape[1]
            or not np.isfinite(train_x.data).all()
            or not np.isfinite(dev_x.data).all()):
        raise ValueError("nonfinite interaction design")
    runs = []
    for strength in (0.1, 0.3, 1.0):
        model = LogisticRegression(C=strength, penalty="l1", solver="liblinear",
                                   max_iter=1000, random_state=0)
        model.fit(train_x, labels[train])
        probability = model.predict_proba(dev_x)[:, 1]
        threshold = threshold_on_dev(probability, labels[dev])
        weight = model.coef_.ravel()
        runs.append({
            "C": strength, "threshold": threshold,
            "validation": metric(labels[dev], probability, threshold),
            "active_atomic_terms": int(np.count_nonzero(weight[:153])),
            "active_structured_pairs": int(np.count_nonzero(weight[153:1099])),
            "active_structured_lexical_pairs": int(np.count_nonzero(weight[1099:])),
        })
        print(runs[-1], flush=True)
    result = {
        "scope": "twibot20_official_train_dev_only_source_traceable_interactions",
        "test_rows_accessed": False,
        "source_archive_sha256": sha256(args.zip),
        "base_facts_sha256": sha256(args.base_facts),
        "source_audit_sha256": sha256(args.source_audit),
        "sparse_artifact_sha256": sha256(args.sparse_artifact),
        "reference_version": reference.version,
        "input_rule": "44 structured evidence tails, 109 frozen source-local unigrams, and pairwise products of structured with structured or lexical terms",
        "baseline_validation": old_metrics,
        "design_width": int(design.shape[1]),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
