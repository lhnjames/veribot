"""Dev-only native-follow neighbor text evidence added to frozen own-account inputs."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    archive_path = Path(raw_manifest["archive_path"])
    base_path = root / "data/processed/twibot20_v3/facts.npz"
    neighbor_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_facts.npz"
    neighbor_manifest = json.loads(neighbor_path.with_suffix(".json")
                                   .read_text(encoding="utf-8"))
    source_audit_path = root / "research/certificates/twibot20_full_source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    selected_path = root / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz"
    prior_path = root / "output/v3_twibot20_sparse_text_evidence/validation_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (neighbor_manifest["fact_cache_sha256"] != sha256(neighbor_path)
            or neighbor_manifest["source_archive_sha256"] != raw_manifest["archive_sha256"]
            or neighbor_manifest["base_facts_sha256"] != sha256(base_path)
            or source_audit["fact_cache_sha256"] != sha256(base_path)
            or prior["sparse_l1_c1_artifact_sha256"] != sha256(selected_path)):
        raise ValueError("official source, neighbor facts, or fixed model differs")
    with np.load(base_path, allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        positions = base["positions"].astype(int)
        labels = base["labels"].astype(int)
        values = base["values"].astype(float)
        contexts = base["contexts"].astype(str)
        names = base["names"].astype(str).tolist()
    with np.load(neighbor_path, allow_pickle=False) as neighbor:
        if (not np.array_equal(neighbor["ids"].astype(str), ids)
                or not np.array_equal(neighbor["split"].astype(str), parts)
                or not np.array_equal(neighbor["positions"].astype(int), positions)):
            raise ValueError("neighbor facts and official owner order differ")
        neighbor_values = neighbor["values"].astype(float)
        neighbor_support = neighbor["supports"].astype(float)
        neighbor_names = neighbor["names"].astype(str).tolist()
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    documents = []
    with zipfile.ZipFile(archive_path) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if (ids[index] != str(row["ID"]) or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("official own-account source order differs")
                    documents.append(" ".join(str(tweet or "")
                                              for tweet in row.get("tweet") or []))
    reference = LabelBlindReference.load(
        root / "output/v3_twibot20_sparse_text_evidence/reference.json.gz")
    own = reference.transform(values[:len(documents)],
                              contexts=contexts[:len(documents)])
    structured = np.column_stack((np.log1p(own.upper) * own.reliability,
                                  np.log1p(own.lower) * own.reliability))
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True,
        dtype=np.float64)
    text_train = vectorizer.fit_transform([documents[index] for index in train])
    text_dev = vectorizer.transform([documents[index] for index in dev])
    with np.load(selected_path, allow_pickle=False) as selected:
        if (not np.array_equal(vectorizer.get_feature_names_out(),
                               selected["vocabulary"].astype(str))
                or not np.array_equal(vectorizer.idf_, selected["idf"].astype(float))):
            raise ValueError("frozen own-account vocabulary or IDF differs")
        coefficient = selected["coefficient"].astype(float)
        intercept = float(selected["intercept"][0])
        old_threshold = float(selected["threshold"][0])
    old_train = hstack((csr_matrix(structured[train]), text_train), format="csr")
    old_dev = hstack((csr_matrix(structured[dev]), text_dev), format="csr")
    old_probability = expit(old_dev @ coefficient + intercept)
    old_metric = metric(labels[dev], old_probability, old_threshold)
    prior_run = next(row for row in prior["runs"]
                     if row["penalty"] == "l1" and row["C"] == 1.0)
    if (reference.version != prior["reference_version"]
            or abs(old_threshold - prior_run["threshold"]) > 1e-12
            or any(abs(old_metric[key] - prior_run["validation"][key]) > 1e-10
                   for key in ("acc", "macro_f1", "auroc"))):
        raise ValueError("selected own-account dev control did not replay")
    neighbor_reference = LabelBlindReference(min_context_n=30).fit(
        neighbor_values[train], neighbor_names)
    neighbor_evidence = neighbor_reference.transform(neighbor_values[:len(documents)])
    reliability = neighbor_evidence.reliability.copy()
    for direction in range(2):
        support = neighbor_support[:len(documents), direction]
        reliability[:, direction * 109:(direction + 1) * 109] *= (
            support[:, None] / (support[:, None] + 5.0))
    neighbor_matrix = np.column_stack((
        np.log1p(neighbor_evidence.upper) * reliability,
        np.log1p(neighbor_evidence.lower) * reliability))
    if neighbor_matrix.shape[1] != 436 or not np.isfinite(neighbor_matrix).all():
        raise ValueError("native-neighbor lexical evidence differs")
    grid = []
    for neighbor_scale in (0.5, 1.0, 2.0):
        fit_x = hstack((old_train, csr_matrix(neighbor_matrix[train] * neighbor_scale)),
                       format="csr")
        eval_x = hstack((old_dev, csr_matrix(neighbor_matrix[dev] * neighbor_scale)),
                        format="csr")
        for regularization_c in (0.3, 1.0, 3.0):
            candidate = LogisticRegression(C=regularization_c, penalty="l1",
                                           solver="liblinear", max_iter=1000,
                                           random_state=0)
            candidate.fit(fit_x, labels[train])
            probability = candidate.predict_proba(eval_x)[:, 1]
            threshold = threshold_on_dev(probability, labels[dev])
            new_metric = metric(labels[dev], probability, threshold)
            neighbor_coefficient = candidate.coef_.ravel()[old_dev.shape[1]:]
            masked_logit = (candidate.intercept_[0] + old_dev @
                            candidate.coef_.ravel()[:old_dev.shape[1]])
            masked_probability = expit(masked_logit)
            full_prediction = probability >= threshold
            masked_prediction = masked_probability >= threshold
            grid.append({
                "neighbor_scale": neighbor_scale,
                "regularization_c": regularization_c,
                "validation": new_metric,
                "threshold": threshold,
                "nonzero_neighbor_coefficients":
                    int(np.count_nonzero(neighbor_coefficient)),
                "fixed_candidate_neighbor_group_mask": {
                    "changed_class_accounts":
                        int(np.count_nonzero(full_prediction != masked_prediction)),
                    "full_correct": int(np.count_nonzero(full_prediction == labels[dev])),
                    "masked_correct":
                        int(np.count_nonzero(masked_prediction == labels[dev])),
                },
            })
    selected = max(grid, key=lambda row: (row["validation"]["macro_f1"],
                                          row["validation"]["acc"]))
    result = {
        "dataset": "twibot20", "scope": "official_train_dev_only_native_neighbor_text_evidence",
        "test_labels_used_for_fit_or_selection": False,
        "test_neighbor_text_used_without_labels": True,
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "base_facts_sha256": sha256(base_path),
        "neighbor_fact_cache_sha256": sha256(neighbor_path),
        "neighbor_fact_manifest_sha256": sha256(neighbor_path.with_suffix(".json")),
        "selected_own_model_sha256": sha256(selected_path),
        "own_reference_version": reference.version,
        "neighbor_reference_version": neighbor_reference.version,
        "own_active_unigrams": int(np.count_nonzero(coefficient[44:])),
        "neighbor_fact_count": len(neighbor_names),
        "neighbor_evidence_input_count": neighbor_matrix.shape[1],
        "baseline": {"validation": old_metric, "threshold": old_threshold},
        "grid": grid,
        "dev_selected_by": "maximum_macro_f1_then_acc",
        "dev_selected": selected,
    }
    destination = root / "output/v3_twibot20_neighbor_unigram/dev_only.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": old_metric, "dev_selected": selected,
                      "grid": [{"neighbor_scale": row["neighbor_scale"],
                                "regularization_c": row["regularization_c"],
                                "validation": row["validation"]} for row in grid]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
