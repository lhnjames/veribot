"""Immutable factual display of independently replayed compact evidence cards.

The display makes only numeric claims present in a source-to-decision card.
It never assigns a behavioral meaning to a tensor projection or infers the
account's true label from the source record.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from scipy.special import expit

from .mgtab_profile_dictionary import published_profile_description


SOURCE_TIERS = {
    "botsim": "raw_user_post_comment_and_native_reply_graph",
    "fox8": "raw_profile_sampled_tweets_and_native_targets",
    "mgtab": "published_profile_edge_and_mean_tweet_tensors",
}


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _fact_name(dataset: str, term: Mapping[str, Any]) -> str:
    name = term["winning_atomic_fact"] if dataset != "mgtab" else term["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("evidence fact name is missing")
    return name


def _source_tier(dataset: str, term: Mapping[str, Any]) -> str:
    locator = term["source_locator"]
    if not isinstance(locator, dict) or not locator:
        raise ValueError("evidence source locator is missing")
    if dataset == "mgtab":
        level = locator.get("source_level")
        if level not in {"published_profile_tensor_coordinate",
                         "published_typed_edge_tensor",
                         "published_averaged_tweet_tensor_projection"}:
            raise ValueError("MGTAB source tier is not recognized")
        return level
    if locator.get("operation") not in {"account_features", "native_reply_graph_fact",
                                        "profile_features_or_tweet_features",
                                        "sampled_native_interaction_fact"}:
        raise ValueError("raw-record operation is not recognized")
    return "raw_record_or_native_event"


def _decision(logit: float, threshold: float) -> int:
    return int(float(expit(logit)) >= threshold)


def _decision_dependence(items: list[dict[str, Any]], intercept: float,
                         logit: float, threshold: float) -> dict[str, Any]:
    """Exact dependence inside the fixed additive classifier."""
    prediction = _decision(logit, threshold)
    pivotal = [row["item_id"] for row in items
               if _decision(logit - row["logit_contribution"], threshold) != prediction]
    same_direction = sorted(
        (row for row in items if (row["logit_contribution"] > 0 if prediction
                                  else row["logit_contribution"] < 0)),
        key=lambda row: abs(row["logit_contribution"]), reverse=True)
    remaining = logit
    deletion: list[str] | None = None
    deleted: list[str] = []
    for row in same_direction:
        remaining -= row["logit_contribution"]
        deleted.append(row["item_id"])
        if _decision(remaining, threshold) != prediction:
            deletion = deleted.copy()
            break
    score = intercept
    sufficient: list[str] = []
    if _decision(score, threshold) != prediction:
        for row in same_direction:
            score += row["logit_contribution"]
            sufficient.append(row["item_id"])
            if _decision(score, threshold) == prediction:
                break
        if _decision(score, threshold) != prediction:
            raise ValueError("same-direction evidence cannot reconstruct decision")
    return {
        "intercept_only_prediction": _decision(intercept, threshold),
        "intercept_only_probability": float(expit(intercept)),
        "evidence_changes_intercept_class": prediction != _decision(intercept, threshold),
        "individually_pivotal_item_ids": pivotal,
        "minimum_support_deletion_to_flip_item_ids": deletion,
        "minimum_same_direction_sufficient_item_ids": sufficient,
        "meaning": "exact fixed-model input dependence, not a causal claim about source behavior",
    }


def render_verified_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Verify arithmetic and render literal fact/value/contribution statements.

    The calling exporter first checks the complete card's file hash and its
    dataset-specific independent source replay. This function then checks the
    individual card and returns text only from those checked fields.
    """
    dataset = card["dataset"]
    if dataset not in SOURCE_TIERS or card.get("source_level") != SOURCE_TIERS[dataset]:
        raise ValueError("card dataset or source level differs")
    key = "terms" if dataset == "mgtab" else "concepts"
    terms = card[key]
    if not isinstance(terms, list) or not terms:
        raise ValueError("card has no evidence terms")
    intercept = _finite(card["intercept"], "intercept")
    threshold = _finite(card["threshold"], "threshold")
    if not 0 < threshold < 1:
        raise ValueError("decision threshold must lie in (0, 1)")
    account_id = str(card["account_id"])
    if not account_id:
        raise ValueError("account ID is empty")
    result_terms = []
    names = set()
    for index, term in enumerate(terms):
        name = term["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("evidence term name is missing or repeated")
        names.add(name)
        fact_name = _fact_name(dataset, term)
        tier = _source_tier(dataset, term)
        available = term.get("available")
        if not isinstance(available, bool):
            raise ValueError("evidence availability is not Boolean")
        contribution = _finite(term["logit_contribution"], "logit contribution")
        raw = term.get("source_value") if dataset == "mgtab" else term.get("raw_source_value")
        profile_semantics = (published_profile_description(fact_name)
                             if dataset == "mgtab" else None)
        if profile_semantics is not None and (
                tier != "published_profile_tensor_coordinate"
                or term["source_locator"].get("column") !=
                profile_semantics["published_zero_based_column"]
                or term["source_locator"].get("row") != int(account_id)):
            raise ValueError("MGTAB profile statement differs from published tensor location")
        if available:
            value = _finite(raw, "source fact value")
            if profile_semantics is not None:
                published_name = profile_semantics["paper_feature_name"].replace("_", " ")
                if profile_semantics["paper_feature_name"] == "followers_friends_ratio":
                    published_name = "followers-to-friends ratio"
                statement = (f"Published normalized {published_name} "
                             f"(MGTAB Table A5, dimension "
                             f"{profile_semantics['paper_table_a5_one_based_dimension']}) "
                             f"is {value:.12g} in features.pt.")
            else:
                statement = f"{fact_name} computes to {value:.12g} from the cited source."
        else:
            value = None
            if abs(contribution) > 1e-12:
                raise ValueError("unavailable evidence has a nonzero contribution")
            statement = f"{fact_name} is unavailable under the recorded source operation."
        effect = f"Its contribution to the bot log-odds is {contribution:+.12g}."
        item = {
            "item_id": f"{dataset}:{card['seed']}:{account_id}:{index}",
            "evidence_name": name,
            "fact_name": fact_name,
            "source_tier": tier,
            "source_locator": term["source_locator"],
            "source_value": value,
            "available": available,
            "logit_contribution": contribution,
            "claim": f"{statement} {effect}",
        }
        if profile_semantics is not None:
            item["published_profile_semantics"] = profile_semantics
        result_terms.append(item)
    logit = intercept + math.fsum(row["logit_contribution"] for row in result_terms)
    probability = float(expit(logit))
    prediction = int(probability >= threshold)
    if (abs(logit - _finite(card["logit"], "saved logit")) > 1e-10
            or abs(probability - _finite(card["probability"], "saved probability")) > 1e-10
            or prediction != card["prediction"]):
        raise ValueError("evidence terms do not reconstruct the saved decision")
    if "group_contributions" in card and abs(
            math.fsum(_finite(value, "group contribution")
                      for value in card["group_contributions"].values()) -
            math.fsum(row["logit_contribution"] for row in result_terms)) > 1e-10:
        raise ValueError("group contributions differ from evidence terms")
    positive = sorted((row for row in result_terms if row["logit_contribution"] > 0),
                      key=lambda row: row["logit_contribution"], reverse=True)
    negative = sorted((row for row in result_terms if row["logit_contribution"] < 0),
                      key=lambda row: row["logit_contribution"])
    dependence = _decision_dependence(result_terms, intercept, logit, threshold)
    pivotal_ids = set(dependence["individually_pivotal_item_ids"])
    sufficient_ids = set(dependence["minimum_same_direction_sufficient_item_ids"])
    deletion_ids = set(dependence["minimum_support_deletion_to_flip_item_ids"] or ())
    for row in result_terms:
        row["individually_pivotal_within_fixed_model"] = row["item_id"] in pivotal_ids
        row["in_one_minimum_same_direction_sufficient_set"] = row["item_id"] in sufficient_ids
        row["in_minimum_support_deletion_to_flip"] = row["item_id"] in deletion_ids
    return {
        "dataset": dataset, "seed": card["seed"], "account_id": account_id,
        "source_level": card["source_level"],
        "prediction": prediction, "probability": probability,
        "threshold": threshold, "intercept": intercept, "logit": logit,
        "all_items": result_terms,
        "largest_bot_support_item_ids": [row["item_id"] for row in positive[:3]],
        "largest_human_support_item_ids": [row["item_id"] for row in negative[:3]],
        "decision_dependence": dependence,
        "display_rule": "literal source value and exact additive model contribution only",
    }
