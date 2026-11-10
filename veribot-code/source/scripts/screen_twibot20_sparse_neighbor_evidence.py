"""Dev-only screen of source-audited structured, neighbor, and unigram evidence."""

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


def _tails(reference: LabelBlindReference, values: np.ndarray,
           contexts: np.ndarray) -> np.ndarray:
    evidence = reference.transform(values, contexts=contexts)
    return np.column_stack((
        np.log1p(evidence.upper) * evidence.reliability,
        np.log1p(evidence.lower) * evidence.reliability))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--neighbor-source-audit", type=Path, required=True)
    parser.add_argument("--baseline-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    neighbor_audit = json.loads(args.neighbor_source_audit.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_screen.read_text(encoding="utf-8"))
    archive_hash = sha256(args.zip)
    if (base_audit["status"] !=
            "verified_all_labeled_account_fact_cells_against_official_zip"
            or neighbor_audit["status"] !=
            "verified_all_follow_neighbor_profile_fact_cells_against_official_zip"
            or base_audit["source_archive_sha256"] != archive_hash
            or neighbor_audit["source_archive_sha256"] != archive_hash
            or base_audit["fact_cache_sha256"] != sha256(args.base_facts)
            or neighbor_audit["fact_cache_sha256"] != sha256(args.neighbor_facts)
            or baseline["source_archive_sha256"] != archive_hash
            or baseline["base_facts_sha256"] != sha256(args.base_facts)):
        raise ValueError("official source audits or baseline binding differ")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        values = cache["values"].astype(float)
        contexts = cache["contexts"].astype(str)
        names = cache["names"].astype(str).tolist()
    with np.load(args.neighbor_facts, allow_pickle=False) as cache:
        if (not np.array_equal(ids, cache["ids"].astype(str))
                or not np.array_equal(parts, cache["split"].astype(str))
                or not np.array_equal(positions, cache["positions"].astype(int))):
            raise ValueError("neighbor evidence account order differs")
        neighbor_values = cache["values"].astype(float)
        neighbor_names = cache["names"].astype(str).tolist()
    if len(names) != 22 or len(neighbor_names) != 10:
        raise ValueError("expected 22 base and 10 neighbor concepts")
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if (len(train), len(dev)) != (8278, 2365):
        raise ValueError("official train/dev population differs")
    documents = []
    profile_documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = len(documents)
                    if (ids[i] != str(row["ID"]) or parts[i] != part
                            or positions[i] != position):
                        raise ValueError("official source account order differs")
                    documents.append(" ".join(
                        str(text or "") for text in row.get("tweet") or []))
                    profile_documents.append(str(
                        (row.get("profile") or {}).get("description") or ""))
    if len(documents) != len(train) + len(dev):
        raise ValueError("official train/dev population differs")
    base_reference = LabelBlindReference(min_context_n=30).fit(
        values[train], names, contexts=contexts[train])
    if base_reference.version != baseline["reference_version"]:
        raise ValueError("baseline reference differs")
    neighbor_reference = LabelBlindReference(min_context_n=30).fit(
        neighbor_values[train], neighbor_names, contexts=contexts[train])
    base_tails = _tails(base_reference, values[:len(documents)],
                        contexts[:len(documents)])
    neighbor_tails = _tails(neighbor_reference,
                            neighbor_values[:len(documents)],
                            contexts[:len(documents)])
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True,
        dtype=np.float64)
    train_text = vectorizer.fit_transform([documents[i] for i in train])
    dev_text = vectorizer.transform([documents[i] for i in dev])
    profile_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=5, max_features=10000, sublinear_tf=True,
        dtype=np.float64)
    train_profile = profile_vectorizer.fit_transform(
        [profile_documents[i] for i in train])
    dev_profile = profile_vectorizer.transform(
        [profile_documents[i] for i in dev])
    if len(vectorizer.vocabulary_) != baseline["vocabulary_size"]:
        raise ValueError("baseline text vocabulary differs")
    runs = {}
    for label, structured, include_profile in (
            ("base22_plus_unigram", base_tails, False),
            ("base22_neighbor10_plus_unigram",
             np.column_stack((base_tails, neighbor_tails)), False),
            ("base22_plus_unigram_plus_profile", base_tails, True),
            ("base22_neighbor10_plus_unigram_plus_profile",
             np.column_stack((base_tails, neighbor_tails)), True)):
        train_parts = [csr_matrix(structured[train]), train_text]
        dev_parts = [csr_matrix(structured[dev]), dev_text]
        if include_profile:
            train_parts.append(train_profile)
            dev_parts.append(dev_profile)
        train_input = hstack(train_parts, format="csr")
        dev_input = hstack(dev_parts, format="csr")
        model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                   max_iter=1000, random_state=0)
        model.fit(train_input, labels[train])
        probability = model.predict_proba(dev_input)[:, 1]
        threshold = threshold_on_dev(probability, labels[dev])
        coefficient = model.coef_.ravel()
        runs[label] = {
            "validation": metric(labels[dev], probability, threshold),
            "threshold": threshold,
            "nonzero_base_tails": int(np.count_nonzero(coefficient[:44])),
            "nonzero_neighbor_tails": int(np.count_nonzero(coefficient[44:64]))
                if structured.shape[1] == 64 else 0,
            "nonzero_lexical_terms": int(np.count_nonzero(
                coefficient[structured.shape[1]:structured.shape[1]+train_text.shape[1]])),
            "nonzero_profile_terms": int(np.count_nonzero(
                coefficient[structured.shape[1]+train_text.shape[1]:])),
        }
    expected = next(run for run in baseline["runs"]
                    if run["penalty"] == "l1" and run["C"] == 1.0)
    control = runs["base22_plus_unigram"]
    if (abs(control["threshold"] - expected["threshold"]) > 1e-12
            or any(abs(control["validation"][key] -
                       expected["validation"][key]) > 1e-9
                   for key in ("acc", "macro_f1", "auroc"))):
        raise ValueError("dev baseline does not replay")
    result = {
        "dataset": "twibot20", "scope": "official_dev_only",
        "test_predictions_accessed": False,
        "source_archive_sha256": archive_hash,
        "base_facts_sha256": sha256(args.base_facts),
        "base_source_audit_sha256": sha256(args.base_source_audit),
        "neighbor_facts_sha256": sha256(args.neighbor_facts),
        "neighbor_source_audit_sha256": sha256(args.neighbor_source_audit),
        "baseline_screen_sha256": sha256(args.baseline_screen),
        "base_reference_version": base_reference.version,
        "neighbor_reference_version": neighbor_reference.version,
        "text_vocabulary_size": len(vectorizer.vocabulary_),
        "profile_vocabulary_size": len(profile_vectorizer.vocabulary_),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
