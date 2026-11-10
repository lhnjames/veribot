"""Dev-only sparse reasoner over verified structured and lexical evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    if (audit["status"] !=
            "verified_all_labeled_account_fact_cells_against_official_zip"
            or audit["source_archive_sha256"] != sha256(args.zip)
            or audit["fact_cache_sha256"] != sha256(args.base_facts)):
        raise ValueError("base source audit differs")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        values = cache["values"].astype(float)
        contexts = cache["contexts"].astype(str)
        names = cache["names"].astype(str).tolist()
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = len(documents)
                    if (ids[i] != str(row["ID"]) or parts[i] != part
                            or positions[i] != position):
                        raise ValueError("official source order differs")
                    documents.append(" ".join(
                        str(text or "") for text in row.get("tweet") or []))
    if len(documents) != len(train) + len(dev):
        raise ValueError("official train/dev population differs")
    reference = LabelBlindReference(min_context_n=30).fit(
        values[train], names, contexts=contexts[train])
    evidence = reference.transform(values[:len(documents)],
                                   contexts=contexts[:len(documents)])
    structured = np.column_stack((
        np.log1p(evidence.upper) * evidence.reliability,
        np.log1p(evidence.lower) * evidence.reliability))
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True,
        dtype=np.float64)
    train_text = vectorizer.fit_transform([documents[i] for i in train])
    dev_text = vectorizer.transform([documents[i] for i in dev])
    train_input = hstack((csr_matrix(structured[train]), train_text), format="csr")
    dev_input = hstack((csr_matrix(structured[dev]), dev_text), format="csr")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference_path = args.output.parent / "reference.json.gz"
    reference.save(reference_path)
    selected_artifact = args.output.parent / "sparse_l1_c1_model.npz"
    runs = []
    for penalty, strength in (("l2", 0.1), ("l2", 1.0),
                              ("l1", 0.1), ("l1", 1.0)):
        model = LogisticRegression(C=strength, penalty=penalty,
                                   solver="liblinear", max_iter=1000,
                                   random_state=0)
        model.fit(train_input, labels[train])
        probability = model.predict_proba(dev_input)[:, 1]
        threshold = threshold_on_dev(probability, labels[dev])
        coefficient = model.coef_.ravel()
        runs.append({
            "penalty": penalty, "C": strength,
            "validation": metric(labels[dev], probability, threshold),
            "threshold": threshold,
            "nonzero_structured_terms": int(np.count_nonzero(coefficient[:44])),
            "nonzero_lexical_terms": int(np.count_nonzero(coefficient[44:])),
            "max_absolute_coefficient": float(np.max(np.abs(coefficient))),
        })
        if penalty == "l1" and strength == 1.0:
            np.savez_compressed(
                selected_artifact,
                vocabulary=vectorizer.get_feature_names_out().astype(str),
                idf=vectorizer.idf_, coefficient=coefficient,
                intercept=np.asarray([float(model.intercept_[0])]),
                threshold=np.asarray([threshold]),
                structured_names=np.asarray(
                    [f"{name}:upper" for name in names] +
                    [f"{name}:lower" for name in names]),
            )
    report = {
        "dataset": "twibot20", "scope": "official_dev_only_evidence_reasoner_screen",
        "test_predictions_accessed": False,
        "source_archive_sha256": sha256(args.zip),
        "base_facts_sha256": sha256(args.base_facts),
        "base_source_audit_sha256": sha256(args.base_source_audit),
        "reference_version": reference.version,
        "reference_artifact_sha256": sha256(reference_path),
        "sparse_l1_c1_artifact_sha256": sha256(selected_artifact),
        "reference_used_labels": False,
        "text_vocabulary_selection": "official_train_only_min_df_10_max_20000",
        "text_evidence": "source-addressable train-IDF-normalized unigram TF-IDF",
        "structured_evidence": "44 label-blind upper/lower tail scores from 22 source-audited facts",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "runs": runs,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"scope": report["scope"],
                      "vocabulary_size": report["vocabulary_size"],
                      "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
