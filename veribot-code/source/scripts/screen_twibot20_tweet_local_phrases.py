"""Dev-only screen of source-local unigram and adjacent bigram evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


TOKEN = re.compile(r"(?u)\b[a-zA-Z]{3,}\b")


def phrases(tweets: list[str]) -> list[str]:
    result = []
    for tweet in tweets:
        normalized = unicodedata.normalize("NFKD", str(tweet or "")).encode(
            "ascii", "ignore").decode("ascii").lower()
        words = TOKEN.findall(normalized)
        result.extend(word for word in words if word not in ENGLISH_STOP_WORDS)
        result.extend(f"bi:{a}_{b}" for a, b in zip(words, words[1:])
                      if a not in ENGLISH_STOP_WORDS or b not in ENGLISH_STOP_WORDS)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_labeled_account_fact_cells_against_official_zip"
            or audit["source_archive_sha256"] != sha256(args.zip)
            or audit["fact_cache_sha256"] != sha256(args.base_facts)):
        raise ValueError("official base facts or source audit differ")
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
    with zipfile.ZipFile(args.zip) as archive:
        for part in ("train", "dev"):
            with archive.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if (ids[index] != str(row["ID"]) or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("official source order differs")
                    documents.append([str(tweet or "") for tweet in row.get("tweet") or []])
    if len(documents) != len(train) + len(dev):
        raise ValueError("train/dev population differs")
    reference = LabelBlindReference(min_context_n=30).fit(
        values[train], names, contexts=contexts[train])
    evidence = reference.transform(values[:len(documents)],
                                   contexts=contexts[:len(documents)])
    structured = np.column_stack((
        np.log1p(evidence.upper) * evidence.reliability,
        np.log1p(evidence.lower) * evidence.reliability))
    runs = []
    for min_df, max_features in ((10, 20000), (5, 30000)):
        vectorizer = TfidfVectorizer(
            analyzer=phrases, min_df=min_df, max_features=max_features,
            sublinear_tf=True, dtype=np.float64)
        lexical_train = vectorizer.fit_transform([documents[i] for i in train])
        lexical_dev = vectorizer.transform([documents[i] for i in dev])
        train_input = hstack((csr_matrix(structured[train]), lexical_train), format="csr")
        dev_input = hstack((csr_matrix(structured[dev]), lexical_dev), format="csr")
        for strength in (0.3, 1.0):
            model = LogisticRegression(C=strength, penalty="l1", solver="liblinear",
                                       max_iter=1000, random_state=0)
            model.fit(train_input, labels[train])
            probability = model.predict_proba(dev_input)[:, 1]
            threshold = threshold_on_dev(probability, labels[dev])
            coefficients = model.coef_.ravel()
            runs.append({"min_df": min_df, "max_features": max_features,
                         "C": strength, "vocabulary_size": len(vectorizer.vocabulary_),
                         "active_phrases": int(np.count_nonzero(coefficients[44:])),
                         "active_bigrams": int(sum(term.startswith("bi:") and
                                                   coefficients[44 + idx] != 0
                                                   for term, idx in vectorizer.vocabulary_.items())),
                         "validation": metric(labels[dev], probability, threshold),
                         "threshold": threshold})
    result = {"dataset": "twibot20", "scope": "official_dev_only",
              "test_predictions_accessed": False,
              "source_archive_sha256": sha256(args.zip),
              "base_facts_sha256": sha256(args.base_facts),
              "base_source_audit_sha256": sha256(args.base_source_audit),
              "reference_version": reference.version,
              "feature_rule": "unigram or adjacent bigram within one source tweet",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
