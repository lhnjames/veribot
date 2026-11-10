"""Dev-only tree capacity screen over the existing source-audited evidence terms."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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
        raise ValueError("official ZIP or base facts differ from source audit")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        values = cache["values"].astype(float)
        contexts = cache["contexts"].astype(str)
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if not np.array_equal(train, np.arange(len(train))) or not np.array_equal(
            dev, np.arange(len(train), len(train) + len(dev))):
        raise ValueError("official train/dev ordering differs")
    documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if (ids[index] != str(row["ID"]) or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("official source row differs")
                    documents.append(" ".join(str(item or "") for item in row.get("tweet") or []))

    reference = LabelBlindReference.load(args.reference)
    evidence = reference.transform(values[:len(documents)], contexts=contexts[:len(documents)])
    structured = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                  np.log1p(evidence.lower) * evidence.reliability))
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True, dtype=np.float64)
    text_train = vectorizer.fit_transform([documents[i] for i in train])
    text_dev = vectorizer.transform([documents[i] for i in dev])
    with np.load(args.sparse_artifact, allow_pickle=False) as existing:
        vocabulary = existing["vocabulary"].astype(str)
        idf = existing["idf"].astype(float)
        coefficient = existing["coefficient"].astype(float)
        if (not np.array_equal(vectorizer.get_feature_names_out(), vocabulary)
                or not np.array_equal(vectorizer.idf_, idf)
                or coefficient.size != 44 + vocabulary.size):
            raise ValueError("text transform differs from selected, audited model")
    active = np.flatnonzero(coefficient[44:] != 0)
    if len(active) != 109:
        raise ValueError("selected lexical evidence count differs")
    train_input = np.column_stack((structured[train], text_train[:, active].toarray()))
    dev_input = np.column_stack((structured[dev], text_dev[:, active].toarray()))
    if not np.isfinite(train_input).all() or not np.isfinite(dev_input).all():
        raise ValueError("nonfinite evidence term")
    runs = []
    for max_leaf_nodes, max_iter, min_samples_leaf, l2_regularization in (
        (15, 150, 30, 1.0),
        (15, 250, 30, 10.0),
        (31, 150, 30, 10.0),
        (31, 250, 50, 10.0),
    ):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=max_leaf_nodes,
            max_iter=max_iter, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, random_state=0)
        model.fit(train_input, labels[train])
        prob = model.predict_proba(dev_input)[:, 1]
        threshold = threshold_on_dev(prob, labels[dev])
        runs.append({
            "max_leaf_nodes": max_leaf_nodes, "max_iter": max_iter,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "threshold": threshold,
            "validation": metric(labels[dev], prob, threshold),
        })
    result = {
        "scope": "twibot20_official_train_dev_only_tree_on_verified_evidence",
        "source_archive_sha256": sha256(args.zip),
        "base_facts_sha256": sha256(args.base_facts),
        "selected_sparse_artifact_sha256": sha256(args.sparse_artifact),
        "reference_version": reference.version,
        "evidence_input": "44 source-audited structured tails plus 109 source-audited lexical terms",
        "test_rows_accessed": False,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_terms": train_input.shape[1], "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
