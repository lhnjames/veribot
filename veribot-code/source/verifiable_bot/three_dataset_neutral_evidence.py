"""One label-free prediction interface for source-bound neutral evidence models."""

from __future__ import annotations

from pathlib import Path

from .botsim_neutral_threshold_runtime import BotSimNeutralThresholdRuntime
from .fox8_twenty_neutral_anchor_runtime import Fox8TwentyNeutralAnchorRuntime
from .mgtab_neutral_anchor_runtime import MGTABNeutralAnchorRuntime


class ThreeDatasetNeutralEvidence:
    def __init__(self, root: Path, dataset: str, seed: int):
        if dataset not in {"botsim", "fox8", "mgtab"}:
            raise ValueError("dataset must be botsim, fox8 or mgtab")
        self.dataset = dataset
        self.seed = seed
        cls = {"botsim": BotSimNeutralThresholdRuntime,
               "fox8": Fox8TwentyNeutralAnchorRuntime,
               "mgtab": MGTABNeutralAnchorRuntime}[dataset]
        self.runtime = cls(root, seed)

    def predict_many(self, account_ids: list[str]) -> list[dict]:
        ids = [str(uid) for uid in account_ids]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Provide a nonempty list of distinct account IDs")
        if self.dataset == "mgtab":
            cards = [self.runtime.predict(uid) for uid in ids]
        else:
            cards = self.runtime.predict_many(ids)
        results = []
        for uid, card in zip(ids, cards, strict=True):
            if card["account_id"] != uid:
                raise ValueError("Source response account order differs")
            if self.dataset == "mgtab":
                shift = card["source_group_decision_certificate"][
                    "evidence_logit_shift_from_zero"]
                threshold = self.runtime.threshold
                model_sha256 = card["neutral_anchor_model_sha256"]
                source_scope = "published_profile_relation_and_average_tweet_tensors"
                items = card["evidence"]
                group_effects = {
                    name: float(value["group_logit_difference"])
                    for name, value in card["group_dependencies"].items()}
                group_source_level = {
                    name: ("published_averaged_tweet_tensor_projection"
                           if name == "published_tweet_projections" else
                           "published_normalized_profile_tensor"
                           if name == "published_profile_coordinates" else
                           "published_typed_relation_tensors")
                    for name in group_effects}
                group_verification_level = {
                    name: "released_tensor_recomputable"
                    for name in group_effects}
                raw_record_provenance = "not_available_in_released_package"
            elif self.dataset == "fox8":
                shift = card["evidence_logit_shift_from_neutral"]
                threshold = card["threshold"]
                model_sha256 = card["model_binding"]["model_sha256"]
                source_scope = "raw_account_tweet_and_native_interaction_records"
                items = card["concepts"]
                group_effects = {
                    group["group"]: float(group["logit_effect"])
                    for group in card["source_group_effects"]}
                group_source_level = {
                    name: "raw_profile_tweet_or_native_interaction_records"
                    for name in group_effects}
                group_verification_level = {
                    name: "raw_record_recomputable"
                    for name in group_effects}
                raw_record_provenance = "available_with_record_locator"
            else:
                shift = card["evidence_logit_shift_from_neutral"]
                threshold = card["threshold"]
                model_sha256 = self.runtime.source_binding["reasoner_sha256"]
                source_scope = "released_user_post_comment_and_reply_records"
                items = card["terms"]
                group_effects = {name: 0.0
                                 for name in self.runtime.reasoner.groups}
                for term in card["terms"]:
                    group_effects[term["name"].split(":", 1)[0]] += float(
                        term["logit_contribution"])
                group_source_level = {
                    name: "released_user_post_comment_or_native_reply_record"
                    for name in group_effects}
                group_verification_level = {
                    name: "released_record_or_declared_summary_readable"
                    for name in group_effects}
                raw_record_provenance = "mixed_records_and_declared_summaries"
            if (abs(card["probability"] - threshold) <= 1e-10 or
                    bool(card["prediction"]) != bool(shift > 0)):
                raise ValueError("Prediction lacks a signed evidence direction")
            reconstruction_error = abs(sum(group_effects.values()) - shift)
            if reconstruction_error > 1e-9:
                raise ValueError("Source-group ledger does not reconstruct score")
            predicted_bot = bool(card["prediction"])
            ledger = []
            for name, effect in group_effects.items():
                if abs(effect) <= 1e-10:
                    role = "neutral_at_1e-10"
                elif (effect > 0) == predicted_bot:
                    role = "supports_predicted_class"
                else:
                    role = "counterevidence"
                ledger.append({
                    "source_group": name,
                    "source_level": group_source_level[name],
                    "verification_level": group_verification_level[name],
                    "logit_effect_relative_to_neutral": effect,
                    "role": role})
            if not any(x["role"] == "supports_predicted_class"
                       for x in ledger):
                raise ValueError("No source group supports the predicted class")
            results.append({
                "dataset": self.dataset, "seed": self.seed,
                "account_id": uid,
                "prediction": card["prediction"],
                "bot_probability": card["probability"],
                "neutral_probability": threshold,
                "evidence_logit_shift_from_neutral": shift,
                "evidence_strengthens_predicted_class": True,
                "source_scope": source_scope,
                "raw_record_provenance": raw_record_provenance,
                "evidence_item_count": len(items),
                "source_group_ledger": ledger,
                "supporting_source_groups": [
                    x["source_group"] for x in ledger
                    if x["role"] == "supports_predicted_class"],
                "counterevidence_source_groups": [
                    x["source_group"] for x in ledger
                    if x["role"] == "counterevidence"],
                "source_group_logit_reconstruction_error":
                    reconstruction_error,
                "model_sha256": model_sha256,
                "source_bound_card": card,
            })
        return results
