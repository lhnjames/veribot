"""Official-dev screen of eight source-local behavior facts, with locked control."""

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
from verifiable_bot.twibot20_behavior_facts import NAMES, extract
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def _tails(reference: LabelBlindReference, values: np.ndarray,
           contexts: np.ndarray) -> np.ndarray:
    evidence = reference.transform(values, contexts=contexts)
    return np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                            np.log1p(evidence.lower) * evidence.reliability))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--baseline-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_screen.read_text(encoding="utf-8"))
    archive_hash = sha256(args.zip)
    if (audit["status"] != "verified_all_labeled_account_fact_cells_against_official_zip"
            or audit["source_archive_sha256"] != archive_hash
            or audit["fact_cache_sha256"] != sha256(args.base_facts)
            or baseline["source_archive_sha256"] != archive_hash
            or baseline["base_facts_sha256"] != sha256(args.base_facts)):
        raise ValueError("source audits or baseline do not bind these inputs")
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
    if (len(train), len(dev)) != (8278, 2365):
        raise ValueError("official split counts differ")
    documents: list[str] = []
    behavior: list[np.ndarray] = []
    with zipfile.ZipFile(args.zip) as archive:
        for part in ("train", "dev"):
            with archive.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if (ids[index] != str(row["ID"]) or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("official account or source location differs")
                    tweets = [str(item or "") for item in row.get("tweet") or []]
                    behavior.append(extract(tweets))
                    documents.append(" ".join(tweets))
    if len(documents) != len(train) + len(dev):
        raise ValueError("train/dev population differs")
    behavior_values = np.asarray(behavior, dtype=float)
    base_reference = LabelBlindReference(min_context_n=30).fit(
        values[train], names, contexts=contexts[train])
    if base_reference.version != baseline["reference_version"]:
        raise ValueError("base evidence reference differs")
    behavior_reference = LabelBlindReference(min_context_n=30).fit(
        behavior_values[train], NAMES, contexts=contexts[train])
    base_evidence = _tails(base_reference, values[:len(documents)],
                           contexts[:len(documents)])
    behavior_evidence = _tails(behavior_reference, behavior_values,
                               contexts[:len(documents)])
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True, dtype=np.float64)
    train_text = vectorizer.fit_transform([documents[i] for i in train])
    dev_text = vectorizer.transform([documents[i] for i in dev])
    if len(vectorizer.vocabulary_) != baseline["vocabulary_size"]:
        raise ValueError("baseline unigram vocabulary differs")
    runs = {}
    for label, extra in (("locked_input_control", None),
                         ("plus_eight_source_local_facts", behavior_evidence)):
        structured = base_evidence if extra is None else np.column_stack(
            (base_evidence, extra))
        x_train = hstack((csr_matrix(structured[train]), train_text), format="csr")
        x_dev = hstack((csr_matrix(structured[dev]), dev_text), format="csr")
        model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                   max_iter=1000, random_state=0)
        model.fit(x_train, labels[train])
        probability = model.predict_proba(x_dev)[:, 1]
        threshold = threshold_on_dev(probability, labels[dev])
        coef = model.coef_.ravel()
        runs[label] = {
            "validation": metric(labels[dev], probability, threshold),
            "threshold": threshold,
            "active_new_tail_terms": int(np.count_nonzero(coef[44:60])) if extra is not None else 0,
            "active_unigrams": int(np.count_nonzero(coef[structured.shape[1]:])),
        }
    expected = next(item for item in baseline["runs"]
                    if item["penalty"] == "l1" and item["C"] == 1.0)
    control = runs["locked_input_control"]
    if (abs(control["threshold"] - expected["threshold"]) > 1e-12
            or any(abs(control["validation"][key] - expected["validation"][key]) > 1e-9
                   for key in ("acc", "macro_f1", "auroc"))):
        raise ValueError("locked baseline did not exactly replay")
    result = {
        "dataset": "twibot20", "scope": "official_train_dev_only",
        "test_rows_accessed": False,
        "source_archive_sha256": archive_hash,
        "base_facts_sha256": sha256(args.base_facts),
        "base_source_audit_sha256": sha256(args.base_source_audit),
        "baseline_screen_sha256": sha256(args.baseline_screen),
        "new_fact_names": NAMES,
        "new_fact_denominators": "source tweet sample; zero denominator yields missing",
        "new_reference_version": behavior_reference.version,
        "new_fact_nonmissing_train_counts": np.sum(
            np.isfinite(behavior_values[train]), axis=0).astype(int).tolist(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
