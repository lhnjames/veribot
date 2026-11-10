"""Render fixed source-and-model claims from independently checked reason cards."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .evidence_narrator import EvidenceItem, organize_evidence


def _number(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {field}")
    return result


def _source_claim(dataset: str, account_id: str, item: Mapping[str, Any]) -> str:
    name = str(item["source_fact"])
    value = _number(item["source_value"], "source value")
    level = item["source_level"]
    query = item["source_query"]
    if dataset == "botsim":
        if level == "published_user_summary":
            observed = int(query["activity_object_count"])
            status = query["cross_source_status"]
            if status == "object_count_matches_declared_summary":
                if observed != value:
                    raise ValueError("matching activity count differs")
                ending = f"The released activity file contains {observed} corresponding objects."
            elif status == "object_count_differs_from_declared_summary":
                if observed == value:
                    raise ValueError("discordant activity count matches")
                ending = (f"The released activity file contains {observed} "
                          "corresponding objects; the two published quantities differ.")
            else:
                raise ValueError("unknown published-count comparison")
            return (f"For account {account_id}, the published user table declares "
                    f"{name} = {value:.17g}. {ending}")
        if level == "published_user_profile":
            return (f"For account {account_id}, the published user profile "
                    f"yields {name} = {value:.17g}.")
        if level == "published_activity_objects":
            return (f"For account {account_id}, the released post/comment "
                    f"objects yield {name} = {value:.17g}.")
        if level == "published_native_reply_objects":
            return (f"For account {account_id}, the released reply and parent "
                    f"objects yield {name} = {value:.17g}.")
        raise ValueError("unknown BotSim source level")
    if dataset == "fox8":
        if level != "raw_profile_tweets_or_native_interaction":
            raise ValueError("unknown Fox8 source level")
        return (f"For account {account_id}, the cited raw account, sampled "
                f"tweet, or native interaction records yield {name} = {value:.17g}.")
    if dataset == "mgtab":
        if level == "published_averaged_tweet_tensor_train_only_projection":
            return (f"For account {account_id}, the published average-tweet "
                    f"tensor and frozen train-only projection yield {name} "
                    f"= {value:.17g}.")
        if level == "published_normalized_profile_tensor":
            return (f"For account {account_id}, the published normalized "
                    f"profile tensor gives {name} = {value:.17g}.")
        if level == "published_typed_relation_tensors_and_derived_fact":
            return (f"For account {account_id}, the published typed relation "
                    f"tensors yield {name} = {value:.17g}.")
        raise ValueError("unknown MGTAB source level")
    raise ValueError("unknown dataset")


def evidence_items(card: Mapping[str, Any]) -> list[EvidenceItem]:
    dataset = str(card["dataset"])
    uid = str(card["account_id"])
    probability = _number(card["probability"], "probability")
    threshold = _number(card["threshold"], "threshold")
    if (dataset not in {"botsim", "fox8", "mgtab"} or
            not uid or not 0 <= probability <= 1 or not 0 < threshold < 1 or
            int(probability >= threshold) != int(card["prediction"])):
        raise ValueError("fixed account decision differs")
    result = []
    for index, item in enumerate(card["displayed_items"]):
        query = item["source_query"]
        if not isinstance(query, dict) or not query.get("source_ids"):
            raise ValueError("displayed source lacks a locator")
        without = _number(item["probability_without_fact"], "masked probability")
        if not 0 <= without <= 1:
            raise ValueError("masked probability is outside [0,1]")
        changed = int(without >= threshold) != int(card["prediction"])
        role = item["role"]
        expected_role = "class_critical" if changed else "score_relevant"
        contrary = role == "contrary_score_relevant"
        effect = _number(item["logit_difference_when_withheld"],
                         "withheld-input logit effect")
        predicted_sign = 1 if int(card["prediction"]) else -1
        if (changed != item["class_changes_when_withheld"] or
                (role != expected_role and not contrary) or
                (contrary and (changed or effect * predicted_sign >= -1e-10))):
            raise ValueError("exact effect role differs from saved decision")
        source = _source_claim(dataset, uid, item)
        effect_text = (f"With this frozen model input withheld, Bot probability "
                       f"changes from {probability:.17g} to {without:.17g}; "
                       f"the predicted class {'changes' if changed else 'stays the same'}.")
        if contrary:
            effect_text += " This input effect opposes the predicted class."
        result.append(EvidenceItem.from_mapping({
            "item_id": f"reason_{index + 1}",
            "claim": source + " " + effect_text,
            "operation": str(query["operation"]),
            "source_ids": query["source_ids"],
        }))
    if len(result) > 6:
        raise ValueError("displayed source-item budget differs")
    return result


def render_report(card: Mapping[str, Any], *, use_api: bool = False,
                  client: Any | None = None, model: str | None = None,
                  group_effects: list[Mapping[str, Any]] | None = None,
                  group_effect_matrix_sha256: str | None = None,
                  source_counterfactual: Mapping[str, Any] | None = None
                  ) -> dict[str, Any]:
    """An optional LLM can order fixed claims; it cannot write their content."""
    items = evidence_items(card)
    source_index = {source: True for item in items for source in item.source_ids}
    prediction = "bot" if int(card["prediction"]) == 1 else "human"
    organized = organize_evidence(prediction, items, source_index,
                                  use_api=use_api, client=client, model=model)
    by_id = {item.item_id: item for item in items}
    ordered_claims = " ".join(
        f"{by_id[item_id].claim} [{', '.join(by_id[item_id].source_ids)}]"
        for item_id in organized["ordered_item_ids"])
    uid = str(card["account_id"])
    probability = _number(card["probability"], "probability")
    threshold = _number(card["threshold"], "threshold")
    empty_probability = _number(card["all_evidence_zero_probability"],
                                "all-zero encoded evidence probability")
    if not 0 <= empty_probability <= 1:
        raise ValueError("all-zero probability is outside [0,1]")
    decision = (f"The frozen classifier assigns account {uid} the "
                f"{'Bot' if prediction == 'bot' else 'Human'} class "
                f"(Bot probability {probability:.17g}; threshold {threshold:.17g}).")
    empty_baseline = (
        "With all encoded evidence inputs set to zero, the same frozen "
        f"classifier assigns {'Bot' if empty_probability >= threshold else 'Human'} "
        f"(Bot probability {empty_probability:.17g}). This is a mathematical "
        "input baseline, not a reconstructed account.")
    category = card["reason_category"]
    if category == "single_fact_class_critical":
        meaning = "At least one displayed source input is individually class-critical."
    elif category == "distributed_class_dependence":
        meaning = ("No single displayed input changes the class when withheld, "
                   "but setting all evidence inputs to zero does.")
    elif category == "no_class_change_from_all_evidence":
        meaning = ("Setting all evidence inputs to zero leaves the class "
                   "unchanged; displayed items affect the frozen score.")
    else:
        raise ValueError("unknown exact decision-dependence category")
    if category == "single_fact_class_critical" and not any(
            item["class_changes_when_withheld"] for item in card["displayed_items"]):
        raise ValueError("critical category lacks a displayed class-critical fact")
    fixed_groups = []
    seen_groups = set()
    for group in group_effects or []:
        name = str(group["name"])
        without = _number(group["probability_without_group"],
                          "source group masked probability")
        count = int(group["fact_count"])
        if (not name or name in seen_groups or count < 1 or
                not 0 <= without <= 1):
            raise ValueError("invalid frozen source group effect")
        seen_groups.add(name)
        changed = int(without >= threshold) != int(card["prediction"])
        fixed_groups.append({"name": name, "fact_count": count,
            "probability_without_group": without,
            "class_changes_when_withheld": changed})
    if fixed_groups and not group_effect_matrix_sha256:
        raise ValueError("source group effects lack their matrix binding")
    changing = [row for row in fixed_groups if row["class_changes_when_withheld"]]
    if changing:
        group_summary = " ".join(
            f"Withholding source group {row['name']} from the frozen classifier "
            f"changes the class (Bot probability {row['probability_without_group']:.17g})."
            for row in changing)
    elif fixed_groups:
        largest = max(fixed_groups, key=lambda row: abs(
            row["probability_without_group"] - probability))
        group_summary = (
            "No single source group changes the class when withheld. "
            f"Group {largest['name']} has the largest absolute score effect: "
            f"Bot probability changes from {probability:.17g} to "
            f"{largest['probability_without_group']:.17g}.")
    else:
        group_summary = ""
    source_statement = ""
    fixed_counterfactual = None
    if card["dataset"] == "mgtab":
        if source_counterfactual is None:
            raise ValueError("MGTAB report lacks published content-row response")
        operation = "replace_published_average_tweet_row_with_train_centroid"
        source_p = _number(source_counterfactual["source_probability"],
                           "published content-row replacement probability")
        exemplar_p = _number(source_counterfactual["exemplar_probability"],
                             "observed train-row replacement probability")
        local_p = _number(source_counterfactual["local_donor_probability"],
                          "nearest local train-row replacement probability")
        encoded_p = _number(source_counterfactual[
            "encoded_group_zero_probability"], "content mask probability")
        locators = source_counterfactual["source_ids"]
        certificate = str(source_counterfactual["certificate_sha256"])
        stability_certificate = str(source_counterfactual[
            "stability_certificate_sha256"])
        local_certificate = str(source_counterfactual[
            "local_donor_certificate_sha256"])
        k5_certificate = str(source_counterfactual[
            "local_k5_certificate_sha256"])
        exemplar_row = int(source_counterfactual["nearest_train_row"])
        exemplar_locator = str(source_counterfactual[
            "nearest_train_source_id"])
        local_row = int(source_counterfactual["local_donor_row"])
        local_locator = str(source_counterfactual["local_donor_source_id"])
        local_distance = _number(source_counterfactual[
            "local_donor_distance_in_64d_projection"], "local donor distance")
        centroid_distance = _number(source_counterfactual[
            "centroid_distance_in_64d_projection"], "centroid distance")
        k5_rows = [int(v) for v in source_counterfactual[
            "local_k5_donor_rows"]]
        k5_probabilities = [_number(v, "local k5 donor probability") for v in
                            source_counterfactual["local_k5_donor_probabilities"]]
        k5_distances = [_number(v, "local k5 donor distance") for v in
                        source_counterfactual["local_k5_donor_distances"]]
        k5_locators = [str(v) for v in source_counterfactual[
            "local_k5_source_ids"]]
        content_group = next((row for row in fixed_groups if row["name"] ==
                              "published_averaged_tweet_projection"), None)
        if (source_counterfactual["operation"] != operation or
                not 0 <= source_p <= 1 or not 0 <= exemplar_p <= 1 or
                not 0 <= local_p <= 1 or
                not 0 <= encoded_p <= 1 or
                not isinstance(locators, list) or len(locators) != 1 or
                not str(locators[0]) or len(certificate) != 64 or
                len(stability_certificate) != 64 or
                len(local_certificate) != 64 or exemplar_row < 0 or
                len(k5_certificate) != 64 or
                not exemplar_locator or local_row < 0 or not local_locator or
                not 0 <= local_distance <= centroid_distance or
                len(k5_rows) != 5 or len(k5_probabilities) != 5 or
                len(k5_distances) != 5 or len(k5_locators) != 5 or
                len(set(k5_rows)) != 5 or any(v < 0 for v in k5_rows) or
                any(not 0 <= v <= 1 for v in k5_probabilities) or
                any(v < 0 for v in k5_distances) or
                any(k5_distances[i] > k5_distances[i+1]
                    for i in range(4)) or
                any(not v for v in k5_locators) or
                k5_rows[0] != local_row or
                abs(k5_probabilities[0]-local_p) > 1e-10 or
                abs(k5_distances[0]-local_distance) > 1e-10 or
                k5_locators[0] != local_locator or
                content_group is None or
                abs(encoded_p-content_group["probability_without_group"])
                > 1e-10):
            raise ValueError("published source and encoded mask are not paired")
        source_changes = int(source_p >= threshold) != int(card["prediction"])
        exemplar_changes = int(exemplar_p >= threshold) != int(
            card["prediction"])
        local_changes = int(local_p >= threshold) != int(card["prediction"])
        encoded_changes = int(encoded_p >= threshold) != int(card["prediction"])
        replacement_conclusion_stable = source_changes == exemplar_changes
        local_conclusion_stable = source_changes == local_changes
        k5_flip_count = sum((p >= threshold) != bool(card["prediction"])
                            for p in k5_probabilities)
        fixed_counterfactual = {
            "operation": operation,
            "source_level": "published_average_tweet_tensor_row",
            "source_ids": [str(locators[0])],
            "source_probability": source_p,
            "exemplar_probability": exemplar_p,
            "local_donor_probability": local_p,
            "nearest_train_row": exemplar_row,
            "nearest_train_source_id": exemplar_locator,
            "local_donor_row": local_row,
            "local_donor_source_id": local_locator,
            "local_donor_distance_in_64d_projection": local_distance,
            "centroid_distance_in_64d_projection": centroid_distance,
            "local_k5_donor_rows": k5_rows,
            "local_k5_donor_probabilities": k5_probabilities,
            "local_k5_donor_distances": k5_distances,
            "local_k5_source_ids": k5_locators,
            "local_k5_flip_count": k5_flip_count,
            "encoded_group_zero_probability": encoded_p,
            "source_class_changes": source_changes,
            "exemplar_class_changes": exemplar_changes,
            "local_donor_class_changes": local_changes,
            "encoded_class_changes": encoded_changes,
            "replacement_class_conclusion_stable": replacement_conclusion_stable,
            "local_vs_centroid_class_conclusion_stable": local_conclusion_stable,
            "certificate_sha256": certificate,
            "stability_certificate_sha256": stability_certificate,
            "local_donor_certificate_sha256": local_certificate,
            "local_k5_certificate_sha256": k5_certificate,
        }
        replacement_statement = (
            "The centroid and global-donor replacements agree on whether the class changes. "
            if replacement_conclusion_stable else
            "The class-change conclusion differs between the centroid and global-donor replacements. ")
        local_statement = (
            "The local-donor and centroid replacements agree on whether the class changes. "
            if local_conclusion_stable else
            "The class-change conclusion differs between the local-donor and centroid replacements. ")
        k5_statement = (
            f"Among this account's five nearest published training content rows "
            f"{k5_rows}, {k5_flip_count} of five replacements change the class; "
            f"their projection distances range from {k5_distances[0]:.17g} "
            f"to {k5_distances[-1]:.17g}. "
            + ("The class response varies across these five donors. "
               if 0 < k5_flip_count < 5 else
               "The class-change outcome agrees across these five donors. "))
        source_statement = (
            "Replacing this account's published average-tweet tensor row "
            "with the training-content centroid and rerunning the frozen "
            f"projection, reference, and classifier yields Bot probability "
            f"{source_p:.17g}; the class {'changes' if source_changes else 'stays the same'}. "
            "Replacing that row with the published training content row "
            f"{exemplar_row}, chosen by label-blind nearest-centroid "
            f"distance, yields Bot probability {exemplar_p:.17g}; the class "
            f"{'changes' if exemplar_changes else 'stays the same'}. "
            "Replacing that row with this account's nearest published training content row "
            f"{local_row} in the same frozen projection yields Bot probability "
            f"{local_p:.17g}; the class {'changes' if local_changes else 'stays the same'}. "
            f"The local-donor projection distance is {local_distance:.17g}, "
            f"versus {centroid_distance:.17g} for the centroid. "
            "Zeroing the encoded content group instead yields Bot probability "
            f"{encoded_p:.17g}; the class {'changes' if encoded_changes else 'stays the same'}. "
            + replacement_statement
            + local_statement
            + k5_statement
            + "The tensor substitution is a defined counterfactual, not a "
            f"correction of a known error. [{locators[0]}; "
            f"global replacement source {exemplar_locator}; "
            f"local replacement sources {', '.join(k5_locators)}]")
    elif source_counterfactual is not None:
        raise ValueError("source counterfactual belongs to MGTAB only")
    return {
        "dataset": card["dataset"], "seed": card["seed"],
        "account_id": uid, "prediction": int(card["prediction"]),
        "probability": probability, "threshold": threshold,
        "all_evidence_zero_probability": empty_probability,
        "source_card_sha256": card["source_card_sha256"],
        "effect_matrix_sha256": card["effect_matrix_sha256"],
        "decision_dependency_category": category,
        "group_effect_matrix_sha256": group_effect_matrix_sha256,
        "source_group_effects": fixed_groups,
        "source_counterfactual": fixed_counterfactual,
        "fixed_claims": [{"item_id": item.item_id, "claim": item.claim,
                          "operation": item.operation,
                          "source_ids": list(item.source_ids)} for item in items],
        "language_api_used": use_api,
        "report": " ".join(part for part in
                            (decision, empty_baseline, meaning, group_summary,
                             source_statement, ordered_claims)
                            if part),
        "ordered_item_ids": organized["ordered_item_ids"],
    }
