"""Exact additive-model decision dependency, with zero-masking semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _prediction(logit: float, boundary: float) -> int:
    return int(logit >= boundary)


def additive_dependency(names: Sequence[str], contributions: Sequence[float],
                        intercept: float, threshold: float,
                        reported_logit: float, reported_prediction: int) -> dict:
    """Describe the frozen classifier's arithmetic dependence on its inputs.

    A masked term contributes zero in the model's evidence space. The returned
    minimal-count set is sufficient under that operation and says nothing about
    interventions on the original social account or independent semantic truth.
    """
    if len(names) != len(contributions) or not 0 < threshold < 1:
        raise ValueError("invalid evidence terms or decision threshold")
    if not all(math.isfinite(float(c)) for c in contributions):
        raise ValueError("nonfinite evidence contribution")
    boundary = math.log(threshold / (1.0 - threshold))
    full = float(intercept) + math.fsum(float(c) for c in contributions)
    if abs(full - reported_logit) > 1e-9 or _prediction(full, boundary) != reported_prediction:
        raise ValueError("evidence does not reproduce reported decision")
    baseline = _prediction(float(intercept), boundary)
    pivotal = [
        {"index": i, "name": str(names[i]), "contribution": float(value)}
        for i, value in enumerate(contributions)
        if _prediction(full - float(value), boundary) != reported_prediction
    ]
    selected: list[int] = []
    partial = float(intercept)
    if baseline != reported_prediction:
        direction = 1 if reported_prediction == 1 else -1
        candidates = sorted(
            (i for i, value in enumerate(contributions)
             if float(value) * direction > 0),
            key=lambda i: (-float(contributions[i]) * direction, i))
        for i in candidates:
            selected.append(i)
            partial += float(contributions[i])
            if _prediction(partial, boundary) == reported_prediction:
                break
        if _prediction(partial, boundary) != reported_prediction:
            raise ValueError("supporting evidence cannot reconstruct decision")
        if any(_prediction(partial - float(contributions[i]), boundary) ==
               reported_prediction for i in selected):
            raise ValueError("support set contains a removable term")
    return {
        "mask_operation": "set_selected_evidence_contributions_to_zero",
        "threshold_logit": boundary,
        "full_logit_reconstruction_difference": full - reported_logit,
        "baseline_prediction_after_all_evidence_masked": baseline,
        "full_evidence_contribution": math.fsum(float(c) for c in contributions),
        "signed_logit_margin": (full - boundary) * (1 if reported_prediction else -1),
        "individually_pivotal_terms": pivotal,
        "minimum_count_support_under_other_terms_masked": [
            {"index": i, "name": str(names[i]),
             "contribution": float(contributions[i])} for i in selected],
        "minimum_count_support_logit": partial,
    }
