"""Measure reliance on a saved evidence group without retraining the model."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score


def evaluate_group_reliance(
    labels: np.ndarray,
    saved_probabilities: np.ndarray,
    contributions: dict[str, np.ndarray],
    intercept: float,
    threshold: float,
) -> dict:
    y = np.asarray(labels, dtype=int)
    saved = np.asarray(saved_probabilities, dtype=float)
    if y.ndim != 1 or saved.shape != y.shape or not contributions:
        raise ValueError("labels, probabilities and contribution groups must align")
    if not set(np.unique(y)) <= {0, 1} or not 0 < threshold < 1:
        raise ValueError("binary labels and a probability threshold are required")
    values = {name: np.asarray(column, dtype=float) for name, column in contributions.items()}
    if any(column.shape != y.shape or not np.isfinite(column).all()
           for column in values.values()):
        raise ValueError("contribution values must be finite and aligned")
    logit = float(intercept) + sum(values.values())
    reconstructed = expit(logit)
    maximum_difference = float(np.max(np.abs(reconstructed - saved)))
    if maximum_difference > 1e-12:
        raise ValueError("saved probabilities disagree with saved group contributions")
    prediction = reconstructed >= threshold
    original_error = prediction != y
    total_absolute = sum(np.abs(column) for column in values.values())
    def ablate(column: np.ndarray) -> dict:
        without = expit(logit - column)
        alternative = without >= threshold
        alternative_error = alternative != y
        return {
            "changed_decisions": int(np.sum(alternative != prediction)),
            "new_errors": int(np.sum(~original_error & alternative_error)),
            "corrected_errors": int(np.sum(original_error & ~alternative_error)),
            "acc_without_group": float(accuracy_score(y, alternative)),
            "macro_f1_without_group": float(f1_score(y, alternative, average="macro",
                                                     zero_division=0)),
            "mean_absolute_logit_contribution": float(np.mean(np.abs(column))),
            "mean_share_of_absolute_contributions": float(np.mean(
                np.abs(column) / np.maximum(total_absolute, 1e-12))),
        }
    groups = {name: ablate(column) for name, column in values.items()}
    graph_names = [name for name in values
                   if "graph" in name or "relation" in name
                   or "follow_neighbor" in name or "native_interaction" in name]
    combined_graph = (ablate(sum(values[name] for name in graph_names))
                      if graph_names else None)
    return {
        "n": len(y), "threshold": threshold,
        "max_saved_probability_difference": maximum_difference,
        "original_acc": float(accuracy_score(y, prediction)),
        "original_macro_f1": float(f1_score(y, prediction, average="macro",
                                            zero_division=0)),
        "groups": groups, "combined_graph_group_names": graph_names,
        "combined_graph_removal": combined_graph,
        "interpretation": "fixed-model group removal; not a causal intervention on source data",
    }
