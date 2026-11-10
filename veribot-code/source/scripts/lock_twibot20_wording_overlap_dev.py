"""Freeze the dev-selected literal-wording evidence model before test readback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_twibot20_wording_overlap_dev import tails
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "output/v3_twibot20_wording_overlap"
    screen_path = out / "dev_only.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if (screen["test_predictions_or_metrics_computed"] is not False or
            max(screen["candidates"], key=lambda row: row["validation"]["macro_f1"])[
                "l1_c"] != 1.0):
        raise ValueError("development choice or test isolation differs")
    base_path = root / "data/processed/twibot20_v3/facts.npz"
    words_path = root / "data/processed/twibot20_v3/sparse_active_unigram_audit.npz"
    overlap_path = root / "data/processed/twibot20_v3/wording_overlap_first40.npz"
    baseline_dir = root / "output/v3_twibot20_sparse_text_evidence"
    with np.load(base_path, allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        contexts = base["contexts"].astype(str)
        facts = base["values"].astype(float)
        labels = base["labels"].astype(int)
        names = base["names"].astype(str).tolist()
    with np.load(words_path, allow_pickle=False) as words:
        if not np.array_equal(ids, words["ids"].astype(str)):
            raise ValueError("source word IDs differ")
        word_values = words["values"].astype(float)
    with np.load(overlap_path, allow_pickle=False) as overlap:
        if not np.array_equal(ids, overlap["ids"].astype(str)):
            raise ValueError("overlap IDs differ")
        new_values = overlap["values"].astype(float)
        new_names = overlap["names"].astype(str).tolist()
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    reference = LabelBlindReference.load(baseline_dir / "reference.json.gz")
    if reference.names != tuple(names):
        raise ValueError("base reference changed")
    new_reference = LabelBlindReference(min_context_n=30).fit(
        new_values[train], new_names, contexts=contexts[train])
    if new_reference.version != screen["new_reference_version"]:
        raise ValueError("new reference changed since dev screening")
    matrix = np.column_stack((tails(reference, facts, contexts), word_values,
                              tails(new_reference, new_values, contexts)))
    model = LogisticRegression(C=1, penalty="l1", solver="liblinear",
                               max_iter=1000, random_state=0)
    model.fit(matrix[train], labels[train])
    probability = model.predict_proba(matrix[dev])[:, 1]
    threshold = threshold_on_dev(probability, labels[dev])
    observed = metric(labels[dev], probability, threshold)
    expected = next(row for row in screen["candidates"] if row["l1_c"] == 1.0)
    if (abs(observed["macro_f1"] - expected["validation"]["macro_f1"]) > 1e-12
            or abs(threshold - expected["threshold"]) > 1e-12 or
            not np.allclose(model.coef_[0, -4:],
                            expected["overlap_tail_coefficients"], atol=1e-12,
                            rtol=0)):
        raise ValueError("frozen development model differs from screen")
    model_path = out / "literal_overlap_l1_c1_model.npz"
    reference_path = out / "literal_overlap_reference.json.gz"
    np.savez_compressed(model_path, coefficient=model.coef_[0],
                        intercept=model.intercept_, threshold=np.asarray([threshold]),
                        fact_names=np.asarray(new_names),
                        source_fact_sha256=np.asarray([sha256(overlap_path)]))
    new_reference.save(reference_path)
    lock = {"scope": "dev_selected_source_evidence_classifier_lock",
        "test_predictions_or_metrics_computed": False,
        "selected_l1_c": 1.0, "dev_metric": observed,
        "source_screen_sha256": sha256(screen_path),
        "source_audit_sha256": screen["source_audit_sha256"],
        "source_fact_cache_sha256": sha256(overlap_path),
        "base_reference_version": reference.version,
        "new_reference_version": new_reference.version,
        "model_sha256": sha256(model_path),
        "new_reference_sha256": sha256(reference_path),
        "threshold": threshold,
        "train_accounts": len(train), "dev_accounts": len(dev)}
    (out / "selection_lock.json").write_text(json.dumps(lock, indent=2)
        + "\n", encoding="utf-8")
    print(json.dumps(lock), flush=True)


if __name__ == "__main__":
    main()
