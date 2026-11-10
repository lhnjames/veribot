"""Exact one-concept withholding for a frozen additive evidence classifier."""

from __future__ import annotations

import math

from scipy.special import expit


def concept_effects(decision: dict) -> dict:
    """Calculate how each encoded concept affects this fixed model decision."""
    logit = float(decision["logit"])
    threshold = float(decision["threshold"])
    predicted = int(decision["prediction"])
    terms = decision["terms"]
    if (not 0 < threshold < 1 or
            abs(float(expit(logit)) - float(decision["probability"])) > 1e-12 or
            int(float(expit(logit)) >= threshold) != predicted or
            abs(float(decision["intercept"]) + math.fsum(
                float(term["logit_contribution"]) for term in terms) - logit)
            > 1e-9):
        raise ValueError("concept contributions do not reconstruct decision")
    result = []
    for term in terms:
        difference = float(term["logit_contribution"])
        probability = float(expit(logit - difference))
        class_changes = int(probability >= threshold) != predicted
        result.append({
            "concept": term["name"],
            "source_fact": term["source_fact_name"],
            "recompute_query": term.get("recompute_query"),
            "probability_without_concept": probability,
            "logit_difference_when_withheld": difference,
            "class_changes_when_withheld": class_changes,
            "role": ("class_critical" if class_changes else
                     "score_relevant" if abs(difference) > 1e-10 else
                     "context_only"),
        })
    return {
        "account_id": str(decision["account_id"]),
        "seed": int(decision["seed"]),
        "probability": float(decision["probability"]),
        "prediction": predicted,
        "threshold": threshold,
        "intervention": "zero one encoded concept in the frozen additive classifier; source records and graph stay fixed",
        "concept_effects": result,
    }
