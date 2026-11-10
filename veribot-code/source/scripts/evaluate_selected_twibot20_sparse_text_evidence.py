"""Evaluate locked, source-backed unigram evidence on official test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--active-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    if (lock["selected_model"] != "source_backed_unigram_l1_c1"
            or lock["candidate_test_predictions_accessed"] is not False
            or lock["screen_sha256"] != sha256(args.screen)
            or lock["model_artifact_sha256"] != sha256(args.artifact)
            or lock["reference_artifact_sha256"] != sha256(args.reference)
            or screen["source_archive_sha256"] != sha256(args.zip)
            or screen["base_facts_sha256"] != sha256(args.base_facts)
            or screen["base_source_audit_sha256"] != sha256(args.base_source_audit)
            or audit["status"] !=
            "verified_all_labeled_account_fact_cells_against_official_zip"):
        raise ValueError("locked model or official source audit differs")
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
    test = np.flatnonzero(parts == "test")
    documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = len(documents)
                    if (ids[i] != str(row["ID"]) or parts[i] != part
                            or positions[i] != position):
                        raise ValueError("official source account order differs")
                    documents.append(" ".join(
                        str(text or "") for text in row.get("tweet") or []))
    if len(documents) != len(ids) or len(test) != 1183:
        raise ValueError("official train/dev/test population differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.version != screen["reference_version"] or reference.names != tuple(names):
        raise ValueError("label-blind reference differs")
    with np.load(args.artifact, allow_pickle=False) as saved:
        vocabulary = saved["vocabulary"].astype(str)
        idf = saved["idf"].astype(float)
        coefficient = saved["coefficient"].astype(float)
        intercept = float(saved["intercept"][0])
        threshold = float(saved["threshold"][0])
        structured_names = saved["structured_names"].astype(str).tolist()
    if structured_names != ([f"{name}:upper" for name in names] +
                            [f"{name}:lower" for name in names]):
        raise ValueError("structured evidence names differ")
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True,
        dtype=np.float64)
    vectorizer.fit([documents[i] for i in train])
    if (not np.array_equal(vectorizer.get_feature_names_out(), vocabulary)
            or not np.array_equal(vectorizer.idf_, idf)
            or len(coefficient) != 44 + len(vocabulary)):
        raise ValueError("train-only unigram vocabulary or IDF differs")
    active_audit = json.loads(args.active_audit.read_text(encoding="utf-8"))
    if (active_audit["status"] !=
            "verified_train_idf_and_recomputed_all_active_unigram_values"
            or active_audit["source_archive_sha256"] != sha256(args.zip)
            or active_audit["base_facts_sha256"] != sha256(args.base_facts)
            or active_audit["model_artifact_sha256"] != sha256(args.artifact)
            or active_audit["active_cache_sha256"] != sha256(args.active_cache)):
        raise ValueError("independent active-unigram source audit differs")
    with np.load(args.active_cache, allow_pickle=False) as checked:
        active = checked["active_indices"].astype(int)
        if (not np.array_equal(checked["ids"].astype(str), ids)
                or not np.array_equal(active, np.flatnonzero(coefficient[44:] != 0))
                or not np.array_equal(checked["active_terms"].astype(str),
                                      vocabulary[active])):
            raise ValueError("active-unigram audit population or vocabulary differs")
        audited_values = checked["values"].astype(float)
    actual_active = vectorizer.transform(documents)[:, active].toarray()
    active_difference = float(np.max(np.abs(actual_active - audited_values)))
    if active_difference > 1e-10:
        raise ValueError("sklearn active unigram values differ from independent raw recomputation")
    evidence = reference.transform(values, contexts=contexts)
    structured = np.column_stack((
        np.log1p(evidence.upper) * evidence.reliability,
        np.log1p(evidence.lower) * evidence.reliability))
    dev_text = vectorizer.transform([documents[i] for i in dev])
    dev_input = hstack((csr_matrix(structured[dev]), dev_text), format="csr")
    dev_probability = expit(intercept + dev_input @ coefficient)
    selected = next(run for run in screen["runs"]
                    if run["penalty"] == "l1" and run["C"] == 1.0)
    dev_metric = metric(labels[dev], dev_probability, threshold)
    if (abs(threshold - selected["threshold"]) > 1e-12
            or any(abs(dev_metric[key] - selected["validation"][key]) > 1e-9
                   for key in ("acc", "macro_f1", "auroc"))):
        raise ValueError("saved dev model does not match locked selection")
    test_text = vectorizer.transform([documents[i] for i in test])
    test_input = hstack((csr_matrix(structured[test]), test_text), format="csr")
    structured_contribution = structured[test] @ coefficient[:44]
    text_contribution = np.asarray(test_text @ coefficient[44:]).ravel()
    logits = intercept + structured_contribution + text_contribution
    probability = expit(logits)
    if np.max(np.abs(probability - expit(intercept + test_input @ coefficient))) > 1e-12:
        raise ValueError("evidence groups do not reconstruct test probability")
    report = {
        "manifest": {
            "dataset": "twibot20", "official_split": True,
            "source_archive_sha256": sha256(args.zip),
            "base_facts_sha256": sha256(args.base_facts),
            "base_source_audit_sha256": sha256(args.base_source_audit),
            "active_source_audit_sha256": sha256(args.active_audit),
            "active_source_cache_sha256": sha256(args.active_cache),
            "selection_lock_sha256": sha256(args.selection_lock),
            "model_artifact_sha256": sha256(args.artifact),
            "reference_version": reference.version,
            "reference_artifact_sha256": sha256(args.reference),
            "reference_used_bot_human_labels": False,
            "text_vocabulary_fitted_on": "official_train_only",
            "threshold": threshold,
            "threshold_selected_on": "official_dev_macro_f1",
            "active_lexical_terms": int(np.count_nonzero(coefficient[44:])),
            "vocabulary_size": len(vocabulary),
        },
        "dev": dev_metric,
        "test": metric(labels[test], probability, threshold),
        "reasoner": {"type": "sparse_additive_evidence_only",
                     "intercept": intercept,
                     "structured_coefficient_count": 44,
                     "active_lexical_terms": int(np.count_nonzero(coefficient[44:]))},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    predictions = [{
        "account_id": ids[int(i)], "label": int(labels[i]),
        "probability": float(probability[k]),
        "group_contributions": {
            "structured_evidence": float(structured_contribution[k]),
            "unigram_evidence": float(text_contribution[k]),
        }} for k, i in enumerate(test)]
    (args.output / "test_predictions.json").write_text(
        json.dumps({"manifest": report["manifest"],
                    "predictions": predictions}, indent=2) + "\n",
        encoding="utf-8")
    masked_logit = np.full(len(test), intercept)
    verification = {
        "test_accounts": len(test),
        "max_group_logit_reconstruction_difference": float(np.max(np.abs(
            logits - (intercept + structured_contribution + text_contribution)))),
        "max_all_evidence_masked_logit_difference": float(np.max(np.abs(
            masked_logit - intercept))),
        "active_lexical_terms": int(np.count_nonzero(coefficient[44:])),
        "max_independent_active_unigram_value_difference": active_difference,
        "status": "verified_source_backed_additive_evidence_only",
    }
    (args.output / "artifact_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dev": report["dev"], "test": report["test"],
                      "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
