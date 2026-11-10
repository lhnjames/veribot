"""Convert label-free compact inference to the audited readable report schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scipy.special import expit

from .current_compact_verified_report import render_card


def _card_from_live(result: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    probability = result["probability"]
    threshold = result["threshold"]
    common = {
        "account_id": result["account_id"],
        "prediction": result["prediction"],
        "probability": probability,
        "threshold": threshold,
        "logit": result["logit"],
        "intercept_logit": result["intercept_logit"],
    }
    if dataset == "fox8":
        common.update({
            "evidence": [{
                "concept": row["concept"], "value": row["value"],
                "winning_source_fact": row["winning_fact"],
                "winning_source_value": row["winning_fact_value"],
                "source_locator": row["source_locator"],
                "logit_attribution_from_model": row[
                    "logit_attribution_from_model"],
            } for row in result["evidence"]],
            "source_group_logit_attributions": {
                name: item["model_logit_attribution"]
                for name, item in result["source_group_effects"].items()},
            "source_group_zero_mask": {
                name: {"probability": item["probability_without_group"],
                       "changes_class": item["class_changes_without_group"]}
                for name, item in result["source_group_effects"].items()},
            "all_evidence_mask_probability": result[
                "all_evidence_zero_probability"],
            "all_evidence_mask_changes_class": bool(
                (result["all_evidence_zero_probability"] >= threshold) !=
                result["prediction"]),
        })
    elif dataset == "mgtab":
        common.update({
            "evidence": [{
                "concept": row["concept"], "value": row["value"],
                "winning_fact": row["winning_source_fact"],
                "winning_fact_value": row["winning_source_value"],
                "source_locator": row["source"],
                "model_logit_attribution": row["model_logit_attribution"],
            } for row in result["evidence"]],
            "zero_evidence_logit": result["zero_evidence_logit"],
            "zero_evidence_probability": float(expit(result[
                "zero_evidence_logit"])),
            "group_effects": {
                name: {
                    "model_logit_attribution": item["model_logit_attribution"],
                    "exact_zero_mask_logit_difference": item[
                        "exact_zero_mask_logit_difference"],
                    "probability_without_group": item[
                        "probability_without_group"],
                    "changes_class_without_group": item[
                        "class_changes_without_group"],
                } for name, item in result["source_group_effects"].items()},
            "content_provenance_level": result["content_provenance_level"],
        })
    else:
        raise ValueError("unsupported compact model")
    return common


def render_live(result: Mapping[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    """Render a new account inference without fixed-test cards or labels."""
    return render_card(_card_from_live(result, dataset), dataset, seed)
