"""Official train/dev test of native-neighbor text sharing as classifier input."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
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
    lexical_path = root / "data/processed/twibot20_v3/sparse_active_unigram_audit.npz"
    neighbor_path = root / "data/processed/twibot20_v3/native_neighbor_exact_text.npz"
    neighbor_manifest_path = neighbor_path.with_suffix(".json")
    selected_dir = root / "output/v3_twibot20_sparse_text_evidence"
    selected_path = selected_dir / "sparse_l1_c1_model.npz"
    lock = json.loads((selected_dir / "selection_lock.json").read_text(encoding="utf-8"))
    neighbor_manifest = json.loads(neighbor_manifest_path.read_text(encoding="utf-8"))
    if (neighbor_manifest["classification_labels_used"] is not False
            or neighbor_manifest["fact_cache_sha256"] != sha256(neighbor_path)
            or lock["model_artifact_sha256"] != sha256(selected_path)):
        raise ValueError("source feature or selected baseline binding differs")
    with np.load(base_path, allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        contexts = base["contexts"].astype(str)
        base_facts = base["values"].astype(float)
        labels = base["labels"].astype(int)
        names = base["names"].astype(str).tolist()
        positions = base["positions"].astype(int)
    with np.load(lexical_path, allow_pickle=False) as lexical:
        if not np.array_equal(ids, lexical["ids"].astype(str)):
            raise ValueError("active unigram source account order differs")
        active_indices = lexical["active_indices"].astype(int)
        lexical_values = lexical["values"].astype(float)
    with np.load(neighbor_path, allow_pickle=False) as cached:
        if (not np.array_equal(ids, cached["ids"].astype(str))
                or not np.array_equal(parts, cached["split"].astype(str))
                or not np.array_equal(positions, cached["positions"].astype(int))):
            raise ValueError("native-neighbor source account order differs")
        neighbor_values = cached["values"].astype(float)
        neighbor_names = cached["names"].astype(str).tolist()
    with np.load(selected_path, allow_pickle=False) as saved:
        coefficient = saved["coefficient"].astype(float)
        intercept = float(saved["intercept"][0])
        selected_threshold = float(saved["threshold"][0])
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    if ((len(train), len(dev)) != (8278, 2365)
            or base_facts.shape[1] != 22 or lexical_values.shape[1] != 109
            or neighbor_values.shape[1] != 3
            or len(active_indices) != 109):
        raise ValueError("official train/dev evidence population differs")
    reference = LabelBlindReference.load(selected_dir / "reference.json.gz")
    if reference.names != tuple(names):
        raise ValueError("selected source reference vocabulary differs")
    base_matrix = np.column_stack((tails(reference, base_facts, contexts),
                                   lexical_values))
    frozen_coefficients = np.r_[coefficient[:44], coefficient[44:][active_indices]]
    frozen_probability = expit(intercept + base_matrix[dev] @ frozen_coefficients)
    frozen = metric(labels[dev], frozen_probability, selected_threshold)
    boundary_ties = int(np.count_nonzero(
        np.abs(frozen_probability - selected_threshold) < 1e-10))
    if (abs(frozen["macro_f1"] - lock["official_dev_macro_f1"]) > 0.001
            or boundary_ties != 1):
        raise ValueError("active source words do not replay selected baseline")
    neighbor_reference = LabelBlindReference(min_context_n=30).fit(
        neighbor_values[train], neighbor_names, contexts=contexts[train])
    neighbor_matrix = tails(neighbor_reference, neighbor_values, contexts)
    full_matrix = np.column_stack((base_matrix, neighbor_matrix))
    baseline_model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                        max_iter=1000, random_state=0)
    baseline_model.fit(base_matrix[train], labels[train])
    baseline_probability = baseline_model.predict_proba(base_matrix[dev])[:, 1]
    baseline_threshold = threshold_on_dev(baseline_probability, labels[dev])
    refit_baseline = metric(labels[dev], baseline_probability, baseline_threshold)
    candidates = []
    for strength in (0.1, 0.3, 1.0, 3.0):
        model = LogisticRegression(C=strength, penalty="l1", solver="liblinear",
                                   max_iter=1000, random_state=0)
        model.fit(full_matrix[train], labels[train])
        dev_probability = model.predict_proba(full_matrix[dev])[:, 1]
        threshold = threshold_on_dev(dev_probability, labels[dev])
        candidate = metric(labels[dev], dev_probability, threshold)
        masked = full_matrix[dev].copy()
        masked[:, -6:] = 0.0
        masked_probability = model.predict_proba(masked)[:, 1]
        effect = {
            "nonzero_new_tail_coefficients": int(np.count_nonzero(model.coef_[0, -6:])),
            "new_tail_coefficients": model.coef_[0, -6:].tolist(),
            "dev_probability_changed_accounts": int(np.count_nonzero(
                np.abs(masked_probability - dev_probability) > 1e-10)),
            "dev_class_changed_accounts": int(np.count_nonzero(
                (masked_probability >= threshold) != (dev_probability >= threshold))),
        }
        candidates.append({"regularization_c": strength,
                           "candidate_validation": candidate,
                           "candidate_threshold": threshold,
                           "new_evidence_effect": effect})
    frozen_logit = intercept + base_matrix @ frozen_coefficients
    shared_any = (neighbor_values[:, 0] > 0).astype(float)
    residual_candidates = []
    for regularization in (1.0, 10.0, 100.0):
        def objective(beta: float) -> float:
            z = frozen_logit[train] + beta * shared_any[train]
            return float(np.logaddexp(0, z).sum() -
                         np.dot(labels[train], z) + regularization * beta * beta)
        fitted = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if not fitted.success:
            raise ValueError("one-feature residual fit did not converge")
        probability = expit(frozen_logit[dev] + fitted.x * shared_any[dev])
        selected_threshold = threshold_on_dev(probability, labels[dev])
        residual_candidates.append({
            "l2_penalty": regularization, "single_new_coefficient": float(fitted.x),
            "validation": metric(labels[dev], probability, selected_threshold),
            "threshold": selected_threshold,
        })
    result = {
        "scope": "twibot20_train_dev_only_native_neighbor_exact_text_evidence",
        "test_predictions_or_metrics_computed": False,
        "source_archive_sha256": neighbor_manifest["source_archive_sha256"],
        "base_facts_sha256": sha256(base_path),
        "active_unigram_cache_sha256": sha256(lexical_path),
        "neighbor_fact_cache_sha256": sha256(neighbor_path),
        "baseline_model_sha256": sha256(selected_path),
        "baseline_reference_version": reference.version,
        "neighbor_reference_version": neighbor_reference.version,
        "frozen_baseline_dev": frozen,
        "frozen_baseline_threshold_numeric_tie_count": boundary_ties,
        "same_active_input_refit_baseline_dev": refit_baseline,
        "candidates": candidates,
        "single_feature_frozen_score_residual": residual_candidates,
    }
    path = root / "output/v3_twibot20_native_neighbor_exact_text/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": frozen, "refit_baseline": refit_baseline,
                      "candidates": candidates,
                      "single_feature_residual": residual_candidates}), flush=True)


if __name__ == "__main__":
    main()
