"""Literal source and exact mask reports for the selected compact tree models.

Tree SHAP is an additive accounting identity, but a missing zero-valued input
can have nonzero SHAP attribution. Human-facing claims therefore use frozen
group-zero interventions and never attribute an effect to an absent fact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scipy.special import expit


def _number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fact_text(dataset: str, fact: str, value: float,
               locator: Mapping[str, Any]) -> str:
    if dataset == "fox8":
        return f"{fact} computes to {value:.12g} from the cited raw record or event."
    if fact.startswith("published_profile_coordinate_"):
        alias = locator.get("paper_table_a5_alias")
        if not isinstance(alias, str) or not alias:
            raise ValueError("MGTAB published profile field has no paper alias")
        return (f"Published normalized {alias.replace('_', ' ')} "
                f"(Table A5 dimension {int(locator['column'])+1}) is {value:.12g}.")
    if fact.startswith("published_tweet_projection_"):
        return (f"Published average-tweet tensor projection {fact} "
                f"is {value:.12g}.")
    return f"Published typed-edge/profile-derived {fact} computes to {value:.12g}."


def render_card(card: Mapping[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    """Check one audited card and render only source literals and group masks."""
    if dataset not in {"fox8", "mgtab"} or seed not in (0, 1, 2):
        raise ValueError("unsupported selected compact model")
    account_id = str(card["account_id"])
    if not account_id:
        raise ValueError("missing account ID")
    probability = _number(card["probability"], "probability")
    threshold = _number(card["threshold"], "threshold")
    logit = _number(card["logit"], "logit")
    intercept = _number(card["intercept_logit"], "intercept")
    if (not 0 < probability < 1 or not 0 < threshold < 1
            or abs(float(expit(logit)) - probability) > 1e-10
            or int(probability >= threshold) != card["prediction"]):
        raise ValueError("saved probability, logit, threshold or class differ")
    evidence = card["evidence"]
    if len(evidence) != 20:
        raise ValueError("selected evidence concept count differs")
    by_group: dict[str, list[dict[str, Any]]] = {}
    unavailable_nonzero_shap = 0
    for term in evidence:
        concept = str(term["concept"])
        group, sep, tail = concept.partition(":")
        if sep != ":" or tail not in {"high_tail_max", "low_tail_max"}:
            raise ValueError("unknown concept name")
        _number(term["value"], "concept score")
        fact = term["winning_source_fact"] if dataset == "fox8" else term["winning_fact"]
        value = (term["winning_source_value"] if dataset == "fox8"
                 else term["winning_fact_value"])
        locator = term["source_locator"]
        shap = _number((term["logit_attribution_from_model"] if dataset == "fox8"
                        else term["model_logit_attribution"]), "SHAP attribution")
        if fact is None:
            if value is not None or locator is not None:
                raise ValueError("absent fact has a source value or locator")
            unavailable_nonzero_shap += abs(shap) > 1e-12
            claim = f"No source fact is available for {concept}."
            source_value = None
            source_tier = "unavailable"
        else:
            if not isinstance(fact, str) or not fact or not isinstance(locator, dict) or not locator:
                raise ValueError("present fact has no source locator")
            source_value = _number(value, "source value")
            if dataset == "fox8":
                if locator.get("account_id") not in {None, account_id}:
                    raise ValueError("Fox8 source account differs")
                if "raw_profile_gzip_sha256" in locator:
                    source_tier = "raw_neighbor_profiles_and_native_events"
                elif "native_event_index_sha256" in locator:
                    source_tier = "native_event_index"
                elif "raw_tweet_archive_sha256" in locator:
                    source_tier = "raw_sampled_tweet"
                elif "raw_gzip_line" in locator:
                    source_tier = "raw_account_profile"
                else:
                    raise ValueError("Fox8 source tier is unknown")
            else:
                row = locator.get("row", locator.get("account_row"))
                if row != int(account_id):
                    raise ValueError("MGTAB published source row differs")
                source_tier = locator.get("level")
                if source_tier not in {
                    "published_normalized_profile_tensor",
                    "published_average_tweet_tensor_projection",
                    "published_typed_edge_and_profile_tensors",
                }:
                    raise ValueError("MGTAB published source tier is unknown")
            claim = _fact_text(dataset, fact, source_value, locator)
        by_group.setdefault(group, []).append({
            "concept": concept,
            "claim": claim,
            "winning_fact": fact,
            "source_value": source_value,
            "source_tier": source_tier,
            "source_locator": locator,
        })
    if len(by_group) != 10 or any(len(rows) != 2 for rows in by_group.values()):
        raise ValueError("selected source group vocabulary differs")
    if dataset == "fox8":
        effects = card["source_group_zero_mask"]
        attribution = card["source_group_logit_attributions"]
    else:
        effects = card["group_effects"]
        attribution = {name: row["model_logit_attribution"]
                       for name, row in effects.items()}
    if set(effects) != set(by_group) or set(attribution) != set(by_group):
        raise ValueError("group effect names differ from evidence")
    if abs(intercept + math.fsum(_number(v, "group attribution")
                                  for v in attribution.values()) - logit) > 1e-9:
        raise ValueError("model accounting does not reconstruct the logit")
    groups = []
    for name, facts in by_group.items():
        effect = effects[name]
        masked = _number((effect["probability"] if dataset == "fox8"
                          else effect["probability_without_group"]),
                         "masked probability")
        if not 0 <= masked <= 1:
            raise ValueError("masked probability is outside [0, 1]")
        flips = (int(masked >= threshold) != card["prediction"])
        recorded_flips = (effect["changes_class"] if dataset == "fox8"
                          else effect["changes_class_without_group"])
        if flips != recorded_flips:
            raise ValueError("group mask class flag differs")
        if all(row["winning_fact"] is None for row in facts) and abs(masked-probability) > 1e-10:
            raise ValueError("all-absent group changes the frozen prediction")
        statement = (f"With source group {name} set to zero, the frozen model's "
                     f"Bot probability changes from {probability:.6f} to {masked:.6f}." )
        groups.append({"group": name, "facts": facts,
                       "probability_without_group": masked,
                       "probability_difference": probability - masked,
                       "changes_class_without_group": flips,
                       "mask_claim": statement})
    ranked = sorted((row for row in groups
                     if abs(row["probability_difference"]) > 1e-12),
                    key=lambda row: (-abs(row["probability_difference"]),
                                     row["group"]))
    zero_probability = (card["all_evidence_mask_probability"] if dataset == "fox8"
                        else card["zero_evidence_probability"])
    zero_probability = _number(zero_probability, "all-zero probability")
    if dataset == "mgtab" and abs(float(expit(_number(card["zero_evidence_logit"],
                                                      "all-zero logit"))) -
                                       zero_probability) > 1e-10:
        raise ValueError("all-zero MGTAB score differs")
    all_zero_changes_class = int(zero_probability >= threshold) != card["prediction"]
    if (dataset == "fox8" and all_zero_changes_class !=
            card["all_evidence_mask_changes_class"]):
        raise ValueError("all-zero Fox8 class flag differs")
    decision_text = (f"The frozen classifier assigns account {account_id} "
                     f"the {'Bot' if card['prediction'] else 'Human'} class "
                     f"with Bot probability {probability:.6f}.")
    mask_text = " ".join(row["mask_claim"] for row in ranked[:3])
    return {
        "dataset": dataset, "seed": seed, "account_id": account_id,
        "model_prediction": card["prediction"], "bot_probability": probability,
        "threshold": threshold,
        "decision_statement": decision_text,
        "group_sensitivity_summary": mask_text,
        "source_groups": groups,
        "largest_three_exact_group_changes": [row["group"] for row in ranked[:3]],
        "all_evidence_zero_probability": zero_probability,
        "all_evidence_zero_changes_class": all_zero_changes_class,
        "unavailable_concepts_with_nonzero_shap_accounting": unavailable_nonzero_shap,
        "display_scope": "literal source values and exact frozen group-zero sensitivity",
    }
