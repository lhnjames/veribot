"""Render and validate a source-bound report for the positive-Human MGTAB model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scipy.special import expit


def render_positive_human(result: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Keep statistical source facts separate from behavior interpretations."""
    if seed not in (0, 1, 2):
        raise ValueError("unsupported MGTAB seed")
    probability = float(result["probability"])
    threshold = float(result["threshold"])
    logit = float(result["logit"])
    intercept = float(result["intercept_logit"])
    zero_logit = float(result["zero_evidence_logit"])
    zero_probability = float(result["zero_evidence_probability"])
    if (not all(math.isfinite(v) for v in (
            probability, threshold, logit, intercept, zero_logit,
            zero_probability)) or not 0 < probability < 1 or
            not 0 < threshold < 1 or
            abs(float(expit(logit))-probability) > 1e-10 or
            abs(float(expit(zero_logit))-zero_probability) > 1e-10 or
            int(probability >= threshold) != int(result["prediction"])):
        raise ValueError("MGTAB score, probability or class differs")
    evidence = result["evidence"]
    human = result["positive_human_evidence"]
    effects = result["source_group_effects"]
    if len(evidence) != 20 or len(human) != 10 or len(effects) != 10:
        raise ValueError("MGTAB source group count differs")
    if (abs(intercept + math.fsum(float(row["model_logit_attribution"])
                                  for row in evidence + human) - logit) > 1e-9):
        raise ValueError("source evidence does not account for model logit")
    groups = []
    for k, row in enumerate(human):
        group = row["source_group"]
        source_rows = evidence[2*k:2*k+2]
        if (group not in effects or
                any(not fact["concept"].startswith(group + ":")
                    for fact in source_rows)):
            raise ValueError("Human score and source group differ")
        cited = [fact for fact in source_rows if fact["source"] is not None]
        deviation = max(math.log1p(float(fact["value"]))
                        for fact in source_rows)
        expected = math.exp(-deviation) if cited else 0.0
        if (bool(cited) != bool(row["source_present"]) or
                row["source_facts"] != cited or
                abs(float(row["deviation_score"])-deviation) > 1e-12 or
                abs(float(row["human_normality_value"])-expected) > 1e-12):
            raise ValueError("positive Human score lacks its source facts")
        effect = effects[group]
        without = float(effect["probability_without_group"])
        flip = int(without >= threshold) != int(result["prediction"])
        if flip != bool(effect["class_changes_without_group"]):
            raise ValueError("source group class effect differs")
        groups.append({
            "group": group,
            "source_facts": source_rows,
            "source_present": bool(cited),
            "small_deviation_score": expected,
            "score_definition": "source_present * exp(-max(log1p(high_tail), log1p(low_tail)))",
            "probability_without_group": without,
            "changes_class_without_group": flip,
            "exact_zero_mask_logit_difference": float(
                effect["exact_zero_mask_logit_difference"]),
        })
    if abs(logit - zero_logit - math.fsum(
            row["exact_zero_mask_logit_difference"]
            for row in groups)) > 1e-9:
        raise ValueError("group interventions do not account for model logit")
    without_human = float(result[
        "without_positive_human_evidence_probability"])
    human_flip = int(without_human >= threshold) != int(result["prediction"])
    if human_flip != bool(result[
            "without_positive_human_evidence_changes_class"]):
        raise ValueError("Human evidence intervention differs")
    return {
        "dataset": "mgtab", "model": "positive_human_evidence",
        "seed": seed, "account_id": str(result["account_id"]),
        "model_prediction": int(result["prediction"]),
        "bot_probability": probability, "threshold": threshold,
        "zero_evidence_prediction": int(zero_probability >= threshold),
        "zero_evidence_probability": zero_probability,
        "class_depends_on_evidence_vs_zero_input":
            int(zero_probability >= threshold) != int(result["prediction"]),
        "without_positive_human_evidence_probability": without_human,
        "without_positive_human_evidence_changes_class": human_flip,
        "source_groups": groups,
        "class_critical_source_groups": [row["group"] for row in groups
                                         if row["changes_class_without_group"]],
        "content_provenance_level": result["content_provenance_level"],
        "display_scope": (
            "published source values, defined small-deviation scores, "
            "and exact frozen-model interventions; no behavioral-semantic claim"),
    }
