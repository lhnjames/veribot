"""Frozen sparse-model correction test for a source-checkable profile error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress_twibot20_source_correction import FIELD, FRACTION, SALT, counts, select_indices, source_values
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.selected_dir / "report.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if (manifest["source_archive_sha256"] != sha256(args.zip)
            or manifest["base_facts_sha256"] != sha256(args.facts)
            or manifest["model_artifact_sha256"] != sha256(args.artifact)
            or manifest["reference_artifact_sha256"] != sha256(args.reference)):
        raise ValueError("frozen model or source differs")
    with np.load(args.facts, allow_pickle=False) as facts:
        ids = facts["ids"].astype(str)
        split = facts["split"].astype(str)
        positions = facts["positions"].astype(int)
        labels = facts["labels"].astype(int)
        values = facts["values"].astype(float)
        contexts = facts["contexts"].astype(str)
        names = facts["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    test = np.flatnonzero(split == "test")
    documents = []
    with zipfile.ZipFile(args.zip) as archive:
        for part in ("train", "dev", "test"):
            with archive.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if ids[index] != str(row["ID"]) or positions[index] != position or split[index] != part:
                        raise ValueError("official source order differs")
                    documents.append(" ".join(str(text or "") for text in row.get("tweet") or []))
    if len(documents) != len(ids):
        raise ValueError("official population differs")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        vocabulary = artifact["vocabulary"].astype(str)
        idf = artifact["idf"].astype(float)
        coefficient = artifact["coefficient"].astype(float)
        intercept = float(artifact["intercept"][0])
        threshold = float(artifact["threshold"][0])
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                                 ngram_range=(1, 1), min_df=10, max_features=20000,
                                 sublinear_tf=True, dtype=np.float64)
    vectorizer.fit([documents[i] for i in train])
    if not np.array_equal(vectorizer.get_feature_names_out(), vocabulary) or not np.array_equal(vectorizer.idf_, idf):
        raise ValueError("train-only text evidence differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.version != manifest["reference_version"] or reference.names != tuple(names):
        raise ValueError("reference differs")
    ids_test = ids[test]
    y_test = labels[test]
    source_facts = values[test].copy()
    context_test = contexts[test]
    selected = select_indices(ids_test, source_facts[:, names.index(FIELD)])
    raw = source_values(args.zip, {str(ids_test[i]) for i in selected})
    if len(raw) != len(selected) or any(source_facts[i, names.index(FIELD)] != raw[str(ids_test[i])][0]
                                        for i in selected):
        raise ValueError("source field differs from fact cache")
    text_logit = np.asarray(vectorizer.transform([documents[i] for i in test]) @ coefficient[44:]).ravel()

    def predict(fact_matrix: np.ndarray) -> np.ndarray:
        evidence = reference.transform(fact_matrix, contexts=context_test)
        structured = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                      np.log1p(evidence.lower) * evidence.reliability))
        return expit(intercept + structured @ coefficient[:44] + text_logit)

    clean = predict(source_facts)
    frozen = json.loads((args.selected_dir / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    if ([row["account_id"] for row in frozen] != ids_test.tolist()
            or np.max(np.abs(clean - np.asarray([row["probability"] for row in frozen]))) > 1e-12):
        raise ValueError("clean source does not replay frozen model")
    corrupted_facts = source_facts.copy()
    corrupted_facts[selected, names.index(FIELD)] = 1.0 - corrupted_facts[selected, names.index(FIELD)]
    corrupted = predict(corrupted_facts)
    corrected_facts = corrupted_facts.copy()
    for i in selected:
        corrected_facts[i, names.index(FIELD)] = raw[str(ids_test[i])][0]
    corrected = predict(corrected_facts)
    if (not np.array_equal(corrected_facts, source_facts, equal_nan=True)
            or np.max(np.abs(corrected - clean)) > 1e-12):
        raise ValueError("source correction did not restore frozen evidence")
    corrupted_error = (corrupted >= threshold) != y_test
    corrected_error = (corrected >= threshold) != y_test
    result = {"dataset": "twibot20", "model": "source_backed_unigram_l1_c1",
              "source_archive_sha256": sha256(args.zip),
              "artifact_sha256": sha256(args.artifact),
              "injection": {"field": FIELD, "fraction": FRACTION,
                            "selection_salt": SALT, "uses_class_labels": False},
              "selected_accounts": len(selected), "threshold": threshold,
              "clean": counts(y_test, clean, threshold, selected),
              "corrupted": counts(y_test, corrupted, threshold, selected),
              "source_corrected": counts(y_test, corrected, threshold, selected),
              "correction_removed_errors": int(np.sum(corrupted_error[selected]
                                                      & ~corrected_error[selected])),
              "correction_introduced_errors": int(np.sum(~corrupted_error[selected]
                                                         & corrected_error[selected])),
              "direct_account_label_same_account_count": {
                  "actions": len(selected),
                  "all_errors": int(corrupted_error.sum() - corrupted_error[selected].sum()),
                  "comparison_unit": "account actions, not measured human time"},
              "max_source_correction_probability_difference": float(np.max(np.abs(corrected - clean))),
              "rows": [{"account_id": str(ids_test[i]),
                        "source_locator": f"Twibot-20/test.json[{raw[str(ids_test[i])][1]}].profile.{FIELD}",
                        "source_value": raw[str(ids_test[i])][0],
                        "injected_value": float(corrupted_facts[i, names.index(FIELD)]),
                        "label_for_evaluation": int(y_test[i]),
                        "clean_probability": float(clean[i]),
                        "corrupted_probability": float(corrupted[i]),
                        "corrected_probability": float(corrected[i])}
                       for i in selected]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
