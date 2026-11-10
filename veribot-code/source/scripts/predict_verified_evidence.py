"""One source-bound evidence-to-decision contract for four graph datasets.

The only caller-provided model input is an account ID. Dataset-specific
adapters recompute facts from independently audited published records before
this module normalizes the resulting evidence and exact decision certificate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.special import expit

from predict_botsim_observed_means_from_sources import predict as predict_botsim_observed
from predict_compact_from_verified_sources import predict as predict_compact
from predict_twibot20_from_verified_zip import predict as predict_twibot20
from verifiable_bot.decision_dependency import additive_dependency


DATASETS = ("botsim", "fox8", "mgtab", "twibot20")


def _compact_items(decision: dict, dataset: str) -> list[dict]:
    items = []
    for term in decision["terms"]:
        query = term.get("recompute_query")
        available = bool(term["available"])
        if available and (query is None or not query.get("source_ids")):
            raise ValueError("available compact evidence lacks a source query")
        source_tier = ("published_mean_tweet_projection" if dataset == "mgtab" and
                       str(term["source_fact_name"]).startswith(
                           "published_tweet_projection_") else
                       "published_numeric_profile_or_typed_edge" if dataset == "mgtab"
                       else "raw_record_or_native_event")
        items.append({
            "name": term["name"],
            "source_fact_name": term["source_fact_name"],
            "source_fact_value": term["source_fact_value"],
            "evidence_value": term["model_input_magnitude"],
            "evidence_upper": term["evidence_upper"],
            "evidence_lower": term["evidence_lower"],
            "evidence_reliability": term["evidence_reliability"],
            "selected_tail": term["selected_tail"],
            "available": available,
            "logit_contribution": float(term["logit_contribution"]),
            "source_query": query,
            "source_tier": source_tier,
            "published_profile_semantics": term.get("published_profile_semantics"),
        })
    return items


def _twibot_items(decision: dict) -> list[dict]:
    queries = {row["fact_name"]: row for row in
               decision["structured_recompute_queries"]}
    raw = decision["structured_raw_facts"]
    names = decision["structured_names"]
    values = decision["structured_values"]
    contributions = decision["structured_contributions"]
    if (len(names) != len(values) or len(names) != len(contributions)
            or len(queries) != len(raw)):
        raise ValueError("TwiBot structured evidence dimensions differ")
    items = []
    for name, value, contribution in zip(names, values, contributions, strict=True):
        fact = name.rsplit(":", 1)[0]
        if fact not in queries or fact not in raw or not queries[fact]["source_ids"]:
            raise ValueError("TwiBot structured evidence lacks a raw fact query")
        if raw[fact] is None and abs(float(contribution)) > 1e-12:
            raise ValueError("unavailable TwiBot fact contributes to classification")
        items.append({
            "name": name, "source_fact_name": fact,
            "source_fact_value": raw[fact], "evidence_value": float(value),
            "available": raw[fact] is not None,
            "logit_contribution": float(contribution),
            "source_query": queries[fact],
            "source_tier": "official_raw_profile_tweet_or_follow_declaration",
        })
    tokens = decision["active_unigram_terms"]
    token_values = decision["active_unigram_values"]
    token_contributions = decision["active_unigram_contributions"]
    locations = decision["active_unigram_source_locations"]
    if len(tokens) != len(token_values) or len(tokens) != len(token_contributions):
        raise ValueError("TwiBot token evidence dimensions differ")
    for token, value, contribution in zip(tokens, token_values,
                                          token_contributions, strict=True):
        source_ids = locations.get(token, [])
        if float(value) > 0 and not source_ids:
            raise ValueError("active TwiBot token lacks a raw tweet position")
        items.append({
            "name": token, "source_fact_name": token,
            "source_fact_value": None, "evidence_value": float(value),
            "available": bool(source_ids),
            "logit_contribution": float(contribution),
            "source_query": {"fact_name": token,
                             "source_ids": source_ids,
                             "operation": "frozen_train_vocabulary_tfidf_from_raw_tweets"},
            "source_tier": "official_raw_tweet_token",
        })
    return items


def normalize(raw: dict, dataset: str, account_id: str, seed: int | None) -> dict:
    if raw["dataset"] != dataset or len(raw["decisions"]) != 1:
        raise ValueError("dataset-specific runtime returned an unexpected population")
    decision = raw["decisions"][0]
    if str(decision["account_id"]) != account_id:
        raise ValueError("source-bound runtime returned a different account")
    if (raw.get("classification_labels_accessed",
                raw.get("classification_label_field_used")) is not False or
            raw.get("caller_supplied_fact_matrix_accepted") is not False):
        raise ValueError("source-bound inference contract differs")
    items = (_twibot_items(decision) if dataset == "twibot20" else
             _compact_items(decision, dataset))
    if len(items) != len({item["name"] for item in items}):
        raise ValueError("evidence names are not unique")
    full = float(decision["intercept"]) + math.fsum(
        item["logit_contribution"] for item in items)
    if abs(full - float(decision["logit"])) > 1e-9:
        raise ValueError("source-bound evidence does not reconstruct the logit")
    certificate = decision["decision_dependency"]
    if abs(certificate["full_logit_reconstruction_difference"]) > 1e-9:
        raise ValueError("decision certificate does not reconstruct the logit")
    pivotal = [items[term["index"]] for term in
               certificate["individually_pivotal_terms"]]
    sufficient = [items[term["index"]] for term in
                  certificate["minimum_count_support_under_other_terms_masked"]]
    baseline = certificate["baseline_prediction_after_all_evidence_masked"]
    prediction = int(decision["prediction"])
    content_mask = None
    if dataset == "mgtab":
        content = [item for item in items if item["source_tier"] ==
                   "published_mean_tweet_projection"]
        named = [item for item in items if item["source_tier"] ==
                 "published_numeric_profile_or_typed_edge"]
        if len(content) != 16 or len(named) != 4:
            raise ValueError("MGTAB source tiers differ from frozen evidence model")
        named_logit = float(decision["intercept"]) + math.fsum(
            item["logit_contribution"] for item in named)
        named_probability = float(expit(named_logit))
        named_prediction = int(named_probability >= float(decision["threshold"]))
        content_mask = {
            "scope": "same_frozen_model_and_threshold_with_16_content_terms_zeroed",
            "published_numeric_profile_or_typed_edge_terms": len(named),
            "published_mean_tweet_projection_terms": len(content),
            "content_masked_probability": named_probability,
            "content_masked_prediction": named_prediction,
            "content_projection_changes_class": named_prediction != prediction,
            "content_projection_logit_contribution": math.fsum(
                item["logit_contribution"] for item in content),
        }
    return {
        "contract": "audited_source_to_evidence_to_frozen_decision_v1",
        "dataset": dataset, "seed": seed, "account_id": account_id,
        "prediction": prediction, "probability": decision["probability"],
        "threshold": decision["threshold"],
        "intercept": decision["intercept"],
        "logit": decision["logit"],
        "source_level": raw["source_level"],
        "source_binding": raw["source_binding"],
        "model_binding": raw["model_binding"],
        "caller_supplied_fact_matrix_accepted": False,
        "classification_label_field_used": False,
        "decision_basis_status": (
            "evidence_changes_class_from_intercept" if prediction != baseline
            else "evidence_modifies_score_without_changing_intercept_class"),
        "decision_dependency": certificate,
        "individually_pivotal_items": pivotal,
        "one_minimum_sufficient_item_set": sufficient,
        "evidence_items": items,
        "published_content_projection_mask_audit": content_mask,
        "evidence_scope": (
            "published_tensor_numeric_only_for_content_projection"
            if dataset == "mgtab" else
            "source_recomputable_observations_not_behavior_intent_truth"),
    }


def predict(root: Path, dataset: str, account_id: str, seed: int | None,
            archive: Path | None = None) -> dict:
    if dataset not in DATASETS or not account_id:
        raise ValueError("unsupported dataset or empty account ID")
    if dataset == "twibot20":
        if seed is not None:
            raise ValueError("TwiBot-20 uses the official split and no seed")
        raw = predict_twibot20(root, [account_id], archive=archive)
    elif dataset == "botsim":
        if seed not in (0, 1, 2) or archive is not None:
            raise ValueError("BotSim requires a frozen seed and no ZIP override")
        raw = predict_botsim_observed(root, dataset, seed, [account_id])
        decision = raw["decisions"][0]
        for term in decision["terms"]:
            name = term["source_fact_name"]
            if name is None:
                source_ids = []
            elif name.startswith("native_reply_"):
                source_ids = [
                    f"{root / 'data/raw/botsim_24/Users.csv'}#all_user_names",
                    f"{root / 'data/raw/botsim_24/user_post_comment.json'}#all_comments_and_parent_objects,select_account={account_id}",
                    f"{root / 'data/processed/botsim_v3/reply_sources.jsonl.gz'}#incident_account={account_id}",
                ]
            else:
                source_ids = [
                    f"{root / 'data/raw/botsim_24/Users.csv'}#user_id={account_id}",
                    f"{root / 'data/raw/botsim_24/user_post_comment.json'}#account_id={account_id}",
                ]
            term["recompute_query"] = None if name is None else {
                "fact_name": name, "account_id": account_id,
                "source_ids": source_ids,
                "operation": "recompute_literal_zero_raw_fact_then_frozen_reference_and_concept",
            }
        decision["decision_dependency"] = additive_dependency(
            [term["name"] for term in decision["terms"]],
            [term["logit_contribution"] for term in decision["terms"]],
            decision["intercept"], decision["threshold"],
            decision["logit"], decision["prediction"])
        raw["model_binding"] = {
            "dev_lock_sha256": raw["source_binding"]["dev_lock_sha256"],
            "reference_sha256": raw["source_binding"]["reference_sha256"],
            "reasoner_sha256": raw["source_binding"]["reasoner_sha256"],
        }
    else:
        if seed not in (0, 1, 2) or archive is not None:
            raise ValueError("compact datasets require a frozen seed and no ZIP override")
        raw = predict_compact(root, dataset, seed, [account_id])
    return normalize(raw, dataset, account_id, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(args.project_root.resolve(), args.dataset,
                     args.account_id, args.seed, args.zip)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
