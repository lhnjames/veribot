"""One frozen test readback for a source-verified wording-overlap candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_twibot20_wording_overlap_dev import tails
from train_v3_twibot20 import metric, sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "output/v3_twibot20_wording_overlap"
    lock_path = out / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model_path = out / "literal_overlap_l1_c1_model.npz"
    reference_path = out / "literal_overlap_reference.json.gz"
    overlap_path = root / "data/processed/twibot20_v3/wording_overlap_first40.npz"
    source_audit = root / "research/certificates/twibot20_wording_overlap_full_source_audit.json"
    baseline_dir = root / "output/v3_twibot20_sparse_text_evidence"
    baseline_path = baseline_dir / "sparse_l1_c1_model.npz"
    if (lock["test_predictions_or_metrics_computed"] is not False or
            sha256(model_path) != lock["model_sha256"] or
            sha256(reference_path) != lock["new_reference_sha256"] or
            sha256(overlap_path) != lock["source_fact_cache_sha256"] or
            sha256(source_audit) != lock["source_audit_sha256"]):
        raise ValueError("locked artifacts or source audit changed")
    with np.load(root / "data/processed/twibot20_v3/facts.npz",
                 allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        contexts = base["contexts"].astype(str)
        base_values = base["values"].astype(float)
        names = base["names"].astype(str).tolist()
        labels = base["labels"].astype(int)
    with np.load(root / "data/processed/twibot20_v3/sparse_active_unigram_audit.npz",
                 allow_pickle=False) as words:
        if not np.array_equal(ids, words["ids"].astype(str)):
            raise ValueError("source word account order differs")
        word_values = words["values"].astype(float)
        active = words["active_indices"].astype(int)
    with np.load(overlap_path, allow_pickle=False) as new:
        if not np.array_equal(ids, new["ids"].astype(str)):
            raise ValueError("overlap account order differs")
        new_values = new["values"].astype(float)
    base_reference = LabelBlindReference.load(baseline_dir / "reference.json.gz")
    new_reference = LabelBlindReference.load(reference_path)
    if (base_reference.names != tuple(names) or
            base_reference.version != lock["base_reference_version"] or
            new_reference.version != lock["new_reference_version"]):
        raise ValueError("frozen reference changed")
    base_matrix = np.column_stack((tails(base_reference, base_values, contexts),
                                   word_values))
    new_matrix = tails(new_reference, new_values, contexts)
    matrix = np.column_stack((base_matrix, new_matrix))
    with np.load(model_path, allow_pickle=False) as model:
        coef = model["coefficient"].astype(float)
        intercept = float(model["intercept"][0])
        threshold = float(model["threshold"][0])
    with np.load(baseline_path, allow_pickle=False) as model:
        base_coef = np.r_[model["coefficient"][:44],
                          model["coefficient"][44:][active]]
        base_intercept = float(model["intercept"][0])
        base_threshold = float(model["threshold"][0])
    test = np.flatnonzero(parts == "test")
    if len(test) != 1183 or matrix.shape[1] != len(coef):
        raise ValueError("official test or evidence dimension differs")
    probability = expit(matrix[test] @ coef + intercept)
    baseline_probability = expit(base_matrix[test] @ base_coef + base_intercept)
    masked_probability = expit(base_matrix[test] @ coef[:-4] + intercept)
    outcome = metric(labels[test], probability, threshold)
    baseline = metric(labels[test], baseline_probability, base_threshold)
    prediction = probability >= threshold
    baseline_prediction = baseline_probability >= base_threshold
    masked_prediction = masked_probability >= threshold
    result = {"scope": "official_fixed_test_exploratory_same_population_as_prior_screens",
        "selection_lock_sha256": sha256(lock_path),
        "model_sha256": sha256(model_path),
        "source_audit_sha256": sha256(source_audit),
        "test_accounts": len(test), "candidate": outcome, "baseline": baseline,
        "candidate_only_correct": int(np.count_nonzero(
            (prediction == labels[test]) & (baseline_prediction != labels[test]))),
        "baseline_only_correct": int(np.count_nonzero(
            (prediction != labels[test]) & (baseline_prediction == labels[test]))),
        "new_evidence_mask_probability_changed": int(np.count_nonzero(
            np.abs(probability - masked_probability) > 1e-10)),
        "new_evidence_mask_class_changed": int(np.count_nonzero(
            prediction != masked_prediction)),
        "source_overlap_positive_test_accounts": int(np.count_nonzero(
            np.isfinite(new_values[test, 0]) & (new_values[test, 0] > 0))),
        "candidate_model_consumes_only_source_evidence": True,
        "external_semantic_gold_evaluated": False}
    predictions = out / "fixed_test_predictions.npz"
    np.savez_compressed(predictions, ids=ids[test], probability=probability,
                        baseline_probability=baseline_probability,
                        masked_probability=masked_probability,
                        candidate_prediction=prediction.astype(np.int8),
                        baseline_prediction=baseline_prediction.astype(np.int8))
    result["prediction_artifact_sha256"] = sha256(predictions)
    (out / "fixed_test_report.json").write_text(json.dumps(result, indent=2)
        + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
