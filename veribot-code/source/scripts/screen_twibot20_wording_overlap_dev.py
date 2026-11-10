"""Train/dev-only capacity test for literal first-40-post overlap evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def tails(reference: LabelBlindReference, facts: np.ndarray,
          contexts: np.ndarray) -> np.ndarray:
    scores = reference.transform(facts, contexts=contexts)
    return np.column_stack((np.log1p(scores.upper) * scores.reliability,
                            np.log1p(scores.lower) * scores.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    base_path = root / "data/processed/twibot20_v3/facts.npz"
    words_path = root / "data/processed/twibot20_v3/sparse_active_unigram_audit.npz"
    overlap_path = root / "data/processed/twibot20_v3/wording_overlap_first40.npz"
    overlap_manifest = json.loads(overlap_path.with_suffix(".json")
                                  .read_text(encoding="utf-8"))
    source_audit = root / "research/certificates/twibot20_wording_overlap_full_source_audit.json"
    source_check = json.loads(source_audit.read_text(encoding="utf-8"))
    baseline_dir = root / "output/v3_twibot20_sparse_text_evidence"
    selected_path = baseline_dir / "sparse_l1_c1_model.npz"
    lock = json.loads((baseline_dir / "selection_lock.json")
                      .read_text(encoding="utf-8"))
    if (source_check["feature_sha256"] != sha256(overlap_path) or
            overlap_manifest["class_labels_used"] is not False or
            source_check["accounts"] != 11826 or
            lock["model_artifact_sha256"] != sha256(selected_path)):
        raise ValueError("source or selected baseline binding differs")
    with np.load(base_path, allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        positions = base["positions"].astype(int)
        contexts = base["contexts"].astype(str)
        facts = base["values"].astype(float)
        labels = base["labels"].astype(int)
        names = base["names"].astype(str).tolist()
    with np.load(words_path, allow_pickle=False) as words:
        if not np.array_equal(ids, words["ids"].astype(str)):
            raise ValueError("active source word account order differs")
        active = words["active_indices"].astype(int)
        word_matrix = words["values"].astype(float)
    with np.load(overlap_path, allow_pickle=False) as overlap:
        if (not np.array_equal(ids, overlap["ids"].astype(str)) or
                not np.array_equal(parts, overlap["split"].astype(str)) or
                not np.array_equal(positions, overlap["positions"].astype(int))):
            raise ValueError("overlap account order differs")
        new_facts = overlap["values"].astype(float)
        new_names = overlap["names"].astype(str).tolist()
    with np.load(selected_path, allow_pickle=False) as saved:
        coefficient = saved["coefficient"].astype(float)
        intercept = float(saved["intercept"][0])
        frozen_threshold = float(saved["threshold"][0])
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if ((len(train), len(dev)) != (8278, 2365) or facts.shape[1] != 22
            or word_matrix.shape[1] != 109 or new_facts.shape[1] != 2):
        raise ValueError("source evidence shape differs")
    reference = LabelBlindReference.load(baseline_dir / "reference.json.gz")
    if reference.names != tuple(names):
        raise ValueError("base reference fact names differ")
    base_matrix = np.column_stack((tails(reference, facts, contexts), word_matrix))
    frozen_coefficients = np.r_[coefficient[:44], coefficient[44:][active]]
    frozen_probability = expit(intercept + base_matrix[dev] @ frozen_coefficients)
    frozen = metric(labels[dev], frozen_probability, frozen_threshold)
    if abs(frozen["macro_f1"] - lock["official_dev_macro_f1"]) > 0.001:
        raise ValueError("frozen baseline does not replay")
    overlap_reference = LabelBlindReference(min_context_n=30).fit(
        new_facts[train], new_names, contexts=contexts[train])
    added = tails(overlap_reference, new_facts, contexts)
    full_matrix = np.column_stack((base_matrix, added))
    baseline = LogisticRegression(C=1, penalty="l1", solver="liblinear",
                                  max_iter=1000, random_state=0)
    baseline.fit(base_matrix[train], labels[train])
    baseline_probability = baseline.predict_proba(base_matrix[dev])[:, 1]
    baseline_threshold = threshold_on_dev(baseline_probability, labels[dev])
    baseline_metric = metric(labels[dev], baseline_probability, baseline_threshold)
    candidates = []
    for strength in (0.3, 1.0, 3.0):
        model = LogisticRegression(C=strength, penalty="l1", solver="liblinear",
                                   max_iter=1000, random_state=0)
        model.fit(full_matrix[train], labels[train])
        probability = model.predict_proba(full_matrix[dev])[:, 1]
        threshold = threshold_on_dev(probability, labels[dev])
        masked = full_matrix[dev].copy()
        masked[:, -4:] = 0.0
        without = model.predict_proba(masked)[:, 1]
        candidates.append({"l1_c": strength,
            "validation": metric(labels[dev], probability, threshold),
            "threshold": threshold,
            "overlap_tail_coefficients": model.coef_[0, -4:].tolist(),
            "overlap_tails_nonzero": int(np.count_nonzero(model.coef_[0, -4:])),
            "accounts_with_probability_effect": int(np.count_nonzero(
                np.abs(probability - without) > 1e-10)),
            "accounts_with_class_effect": int(np.count_nonzero(
                (probability >= threshold) != (without >= threshold)))})
    result = {"scope": "twibot20_official_train_dev_only_literal_wording_overlap",
        "test_predictions_or_metrics_computed": False,
        "source_audit_sha256": sha256(source_audit),
        "source_fact_cache_sha256": sha256(overlap_path),
        "baseline_model_sha256": sha256(selected_path),
        "frozen_baseline_dev": frozen,
        "same_input_refit_baseline_dev": baseline_metric,
        "new_reference_version": overlap_reference.version,
        "new_fact_names": new_names,
        "candidates": candidates}
    out = root / "output/v3_twibot20_wording_overlap/dev_only.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": frozen, "candidates": candidates}), flush=True)


if __name__ == "__main__":
    main()
