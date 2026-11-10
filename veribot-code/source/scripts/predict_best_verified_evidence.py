"""One source-to-evidence decision card for the three selected models."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_best_evidence_models import predict as predict_selected


DATASETS = ("botsim", "fox8", "mgtab")


@lru_cache(maxsize=16)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _fox8_first_lines(raw: Path) -> dict[str, int]:
    """Match the first-duplicate policy in the Fox8 source adapter."""
    pattern = re.compile(rb'^\{"user_id"\s*:\s*"?(\d+)"?')
    first: dict[str, int] = {}
    with gzip.open(raw, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.match(line)
            if match is None:
                raise ValueError("Fox8 raw row lacks a leading account ID")
            first.setdefault(match.group(1).decode("ascii"), line_number)
    return first


def _fox8_query(root: Path, account_id: str, fact: str,
                locator: dict) -> dict:
    raw = root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    event = root / "data/processed/fox8_v3/interaction_event_index.json"
    first = _fox8_first_lines(raw)
    def record(uid: str) -> str:
        uid = str(uid)
        if uid not in first:
            raise ValueError("Fox8 cited account is absent from raw source")
        return f"{raw}#line={first[uid]},user_id={uid}"
    if fact.startswith("native_interaction_neighbor_mean_"):
        neighbors = locator["neighbor_account_ids"]
        sources = [record(uid) for uid in neighbors]
        sources.append(f"{event}#incident_account={account_id}")
    elif fact.startswith(("native_reply_", "native_retweet_")):
        sources = [f"{raw}#all_interaction_events,select_account={account_id}",
                   f"{event}#incident_account={account_id}"]
    else:
        sources = [record(account_id)]
    return {"fact": fact, "source_ids": sources,
            "operation": "recompute_from_raw_account_or_native_events"}


def _mgtab_query(fact: str, source: dict) -> dict:
    if source["level"] == "published_averaged_tweet_tensor_train_only_projection":
        sources = [f"{source['file']}#row={source['row']},columns={source['input_columns']}",
                   f"{source['projection_artifact']}#train_only_projection"]
    elif source["level"] == "published_normalized_profile_tensor":
        sources = [f"{source['file']}#row={source['row']},column={source['column']}"]
    else:
        sources = [f"{source['edge_index']}#incident_account={source['account_row']}",
                   f"{source['edge_type']}#matching_edge_rows"]
        if "_neighbor_" in fact:
            feature_tensor = Path(source["edge_index"]).with_name("features.pt")
            neighbor_columns = {
                "default_profile_mean": 1,
                "verified_mean": 2,
                "description_length_mean": 13,
                "followers_friends_ratio_mean": 14,
            }
            column = next((value for suffix, value in neighbor_columns.items()
                           if fact.endswith("_neighbor_" + suffix)), None)
            if column is None:
                raise ValueError("unknown MGTAB neighbor profile source column")
            sources.append(f"{feature_tensor}#neighbor_rows,column={column}")
        if source.get("fact_cache"):
            sources.append(f"{source['fact_cache']}#row={source['account_row']}")
    return {"fact": fact, "source_ids": sources,
            "operation": "recompute_from_published_tensors_and_training_projection"}


@lru_cache(maxsize=2)
def _botsim_activity_object_counts(root: Path) -> dict[str, dict[str, int]]:
    with (root / "data/raw/botsim_24/Users.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        ids = {row["user_id"] for row in csv.DictReader(stream)}
    activity = json.loads((root /
        "data/raw/botsim_24/user_post_comment.json").read_text(encoding="utf-8"))
    if ids != set(activity) or len(ids) != 2907:
        raise ValueError("BotSim user and activity sources differ")
    return {uid: {
        "submission_num": len(record["posts"]),
        "comment_num": len(record["comment_1"]) + len(record["comment_2"]),
        "comment_num_1": len(record["comment_1"]),
        "comment_num_2": len(record["comment_2"]),
    } for uid, record in activity.items()}


def _botsim_query(root: Path, account_id: str, fact: str,
                  locator: str, value: float | None) -> dict:
    users = root / "data/raw/botsim_24/Users.csv"
    activity = root / "data/raw/botsim_24/user_post_comment.json"
    if fact.startswith("native_reply_"):
        sources = [f"{users}#all_user_names",
                   f"{activity}#all_comments_and_parent_objects,select_account={account_id}",
                   locator]
        operation = "recompute_native_reply_relation_from_comment_parent_objects"
        level = "published_native_reply_objects"
    elif fact in {"submission_num", "comment_num", "comment_num_1",
                  "comment_num_2"}:
        sources = [f"{users}#user_id={account_id}"]
        operation = "read_declared_user_summary_count"
        level = "published_user_summary"
    elif fact in {"has_description", "description_len", "n_subreddits"}:
        sources = [f"{users}#user_id={account_id}"]
        operation = "recompute_from_published_user_profile"
        level = "published_user_profile"
    else:
        sources = [f"{activity}#account_id={account_id}"]
        operation = "recompute_from_published_activity_objects"
        level = "published_activity_objects"
    query = {"fact": fact, "source_ids": sources,
             "operation": operation, "verification_scope": level}
    if level == "published_user_summary":
        observed = _botsim_activity_object_counts(root)[account_id][fact]
        if value is None:
            raise ValueError("published summary lacks a numeric value")
        query["activity_object_count"] = observed
        query["cross_source_status"] = (
            "object_count_matches_declared_summary" if float(value) == observed
            else "object_count_differs_from_declared_summary")
    return query


def render_selected_decision(root: Path, dataset: str, seed: int,
                             account_id: str, raw: dict,
                             *, withholding_effects: bool = False) -> dict:
    """Build and check a source card from an already evaluated frozen decision."""
    if dataset not in DATASETS or seed not in (0, 1, 2) or not str(account_id):
        raise ValueError("provide one supported dataset, seed and account ID")
    root = Path(root).resolve()
    if (raw["dataset"] != dataset or raw["seed"] != seed or
            str(raw["account_id"]) != str(account_id)):
        raise ValueError("selected decision identity differs from requested source card")
    decision = raw["decision"]
    terms = []
    for term in (decision["terms"] if dataset == "botsim" else decision["evidence"]):
        if dataset == "botsim":
            fact = term["source_fact_name"]
            query = None if fact is None else _botsim_query(
                root, str(account_id), fact, term["source_locator"],
                term["source_fact_value"])
            level = query["verification_scope"] if query is not None else None
            upper, lower = term["evidence_upper"], term["evidence_lower"]
            reliability = term["evidence_reliability"]
            value = term["source_fact_value"]
            name = term["name"]
        elif dataset == "fox8":
            fact = name = term["fact"]
            query = _fox8_query(root, str(account_id), fact,
                                term["source_locator"])
            level = "raw_profile_tweets_or_native_interaction"
            upper, lower = term["upper_tail"], term["lower_tail"]
            reliability = term["reliability"]
            value = term["raw_value"]
        else:
            fact = name = term["fact"]
            query = _mgtab_query(fact, term["source"])
            level = term["source"]["level"]
            upper, lower = term["upper_tail"], term["lower_tail"]
            reliability = term["reliability"]
            value = term["raw_value"]
        if query is not None and not query["source_ids"]:
            raise ValueError("model evidence lacks a source query")
        terms.append({
            "name": name, "source_fact": fact,
            "source_value": value, "source_level": level,
            "source_fact_observed": value is not None,
            "concept_available": (
                bool(term["available"]) if dataset == "botsim" else
                value is not None),
            "source_query": query,
            "upper_tail": float(upper), "lower_tail": float(lower),
            "reliability": float(reliability),
            "upper_model_input": float(np.log1p(upper) * reliability),
            "lower_model_input": float(np.log1p(lower) * reliability),
            "logit_attribution": float(term["logit_contribution"]),
            "attribution_method": (
                "exact_additive_concept" if dataset == "botsim" else
                "tree_shap_logit_allocation"),
        })
    intercept = float(decision["intercept"] if dataset == "botsim"
                      else decision["intercept_logit"])
    logit = float(decision["logit"])
    probability = float(decision["probability"])
    threshold = float(decision["threshold"])
    reconstructed = intercept + math.fsum(
        term["logit_attribution"] for term in terms)
    if (len(terms) != raw["evidence_term_count"] or
            abs(reconstructed - logit) > 1e-9 or
            abs(float(expit(logit)) - probability) > 1e-12 or
            int(probability >= threshold) != int(decision["prediction"])):
        raise ValueError("source evidence fails to reconstruct selected decision")
    if dataset == "botsim":
        zero_probability = float(expit(intercept))
        exact_dependency = {
            "all_evidence_zero_probability": zero_probability,
            "all_evidence_changes_class": bool(
                (zero_probability >= threshold) != (probability >= threshold)),
        }
    elif dataset == "fox8":
        exact_dependency = decision["evidence_dependency"]
    else:
        zero_probability = float(decision["all_evidence_zero_probability"])
        exact_dependency = {
            "all_evidence_zero_probability": zero_probability,
            "all_evidence_changes_class": bool(decision["all_evidence_changes_class"]),
            "source_group_decision_certificate":
                decision["source_group_decision_certificate"],
            "source_group_masks": decision["group_dependencies"],
        }
    source_binding = dict(raw["source_binding"])
    if dataset in {"fox8", "mgtab"}:
        certificate = root / "research/certificates" / (
            "fox8_neighbor_volume_full_raw_source.json" if dataset == "fox8" else
            "mgtab_full_source_audit.json")
        manifest = Path(source_binding["inference_manifest"])
        source_binding.update({
            "independent_source_audit": str(certificate),
            "independent_source_audit_sha256": _sha256(certificate),
            "inference_manifest_sha256": _sha256(manifest),
        })
    result = {
        "contract": "selected_source_evidence_decision_v1",
        "dataset": dataset, "seed": seed, "model_id": raw["model_id"],
        "account_id": str(account_id),
        "prediction": int(decision["prediction"]),
        "probability": probability, "threshold": threshold,
        "logit": logit, "intercept_logit": intercept,
        "reconstruction_difference": reconstructed - logit,
        "source_binding": source_binding,
        "source_scope": raw["source_scope"],
        "reference_version": decision["reference_version"],
        "evidence_terms": terms,
        "exact_frozen_model_dependency": exact_dependency,
        "caller_supplied_fact_matrix_accepted": False,
        "classification_labels_accessed": False,
    }
    if withholding_effects:
        result["exact_single_input_withholding_effects"] = raw["evidence_effects"]
    return result


def predict(root: Path, dataset: str, seed: int, account_id: str,
            *, withholding_effects: bool = False) -> dict:
    raw = predict_selected(root, dataset, seed, str(account_id),
                           withholding_effects=withholding_effects)
    return render_selected_decision(root, dataset, seed, str(account_id), raw,
                                    withholding_effects=withholding_effects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--withholding-effects", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = predict(Path(__file__).resolve().parents[2], args.dataset,
                     args.seed, args.account_id,
                     withholding_effects=args.withholding_effects)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(args.out)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
