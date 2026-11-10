"""Official-dev-only diagnostic of source-addressable lexical content capacity."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        base = cache["values"].astype(float)
    documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = len(documents)
                    if (str(row["ID"]) != ids[i] or parts[i] != part
                            or positions[i] != position):
                        raise ValueError("official account order differs")
                    documents.append(" ".join(str(t or "")
                                              for t in row.get("tweet") or []))
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if len(documents) != len(train) + len(dev):
        raise ValueError("official train/dev population differs")
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True,
        dtype=np.float64)
    text_train = vectorizer.fit_transform([documents[i] for i in train])
    text_dev = vectorizer.transform([documents[i] for i in dev])
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    base_train = imputer.fit_transform(base[train])
    base_dev = imputer.transform(base[dev])
    scaler = StandardScaler()
    base_train = scaler.fit_transform(base_train)
    base_dev = scaler.transform(base_dev)
    matrices = {
        "text_only": (text_train, text_dev),
        "text_plus_22": (
            hstack((text_train, csr_matrix(base_train)), format="csr"),
            hstack((text_dev, csr_matrix(base_dev)), format="csr")),
    }
    runs = []
    for input_name, (fit_x, eval_x) in matrices.items():
        for strength in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=strength, solver="liblinear",
                                       max_iter=1000)
            model.fit(fit_x, labels[train])
            probability = model.predict_proba(eval_x)[:, 1]
            threshold = threshold_on_dev(probability, labels[dev])
            runs.append({"input": input_name, "C": strength,
                         "validation": metric(labels[dev], probability, threshold),
                         "threshold": threshold,
                         "nonzero_coefficients": int(np.count_nonzero(model.coef_))})
    report = {
        "dataset": "twibot20", "scope": "official_dev_only_capacity_probe",
        "test_predictions_accessed": False,
        "source_archive_sha256": sha256(args.zip),
        "base_facts_sha256": sha256(args.base_facts),
        "train_accounts": len(train), "dev_accounts": len(dev),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "feature_type": "train-only unigram TF-IDF; direct source-addressable text, no hidden embeddings",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"scope": report["scope"],
                      "vocabulary_size": report["vocabulary_size"],
                      "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
