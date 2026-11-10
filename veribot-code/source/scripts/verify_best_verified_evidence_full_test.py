"""Independently read back and validate every exported source decision card."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path

from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = ROOT / "research/certificates/best_verified_evidence_full_test.json"
OUT = ROOT / "research/certificates/best_verified_evidence_full_test_readback.json"
SAVED = {
    "botsim": "output/v3_botsim_observed_means_12",
    "fox8": "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": "output/v3_mgtab_ab_group_additive_tree",
}
COUNTS = {"botsim": 12, "fox8": 74, "mgtab": 150}


def verify_botsim_scope(item: dict, account_id: str,
                        users: dict[str, dict], activity: dict[str, dict]) -> None:
    fact = item["source_fact"]
    if fact is None:
        return
    query = item["source_query"]
    sources = query["source_ids"]
    if fact.startswith("native_reply_"):
        level = "published_native_reply_objects"
        operation = "recompute_native_reply_relation_from_comment_parent_objects"
        expected_files = ("Users.csv", "user_post_comment.json",
                          "reply_sources.jsonl.gz")
    elif fact in {"submission_num", "comment_num", "comment_num_1",
                  "comment_num_2"}:
        level = "published_user_summary"
        operation = "read_declared_user_summary_count"
        expected_files = ("Users.csv",)
    elif fact in {"has_description", "description_len", "n_subreddits"}:
        level = "published_user_profile"
        operation = "recompute_from_published_user_profile"
        expected_files = ("Users.csv",)
    else:
        level = "published_activity_objects"
        operation = "recompute_from_published_activity_objects"
        expected_files = ("user_post_comment.json",)
    if (item["source_level"] != level or
            query["verification_scope"] != level or
            query["operation"] != operation or
            tuple(Path(source.partition("#")[0]).name for source in sources)
            != expected_files):
        raise ValueError("BotSim evidence scope or source file differs")
    if level in {"published_user_summary", "published_user_profile",
                 "published_activity_objects"}:
        if len(sources) != 1 or not sources[0].endswith(
                ("#user_id=" if level != "published_activity_objects" else
                 "#account_id=") + account_id):
            raise ValueError("BotSim direct evidence source account differs")
    if level == "published_user_summary":
        declared = float(users[account_id][fact])
        record = activity[account_id]
        observed = {
            "submission_num": len(record["posts"]),
            "comment_num": len(record["comment_1"]) + len(record["comment_2"]),
            "comment_num_1": len(record["comment_1"]),
            "comment_num_2": len(record["comment_2"]),
        }[fact]
        expected_status = ("object_count_matches_declared_summary"
                           if declared == observed else
                           "object_count_differs_from_declared_summary")
        if (float(item["source_value"]) != declared or
                query["activity_object_count"] != observed or
                query["cross_source_status"] != expected_status):
            raise ValueError("BotSim declared count or object recount differs")


def main() -> None:
    export = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    if len(export["runs"]) != 9 or export["total_accounts"] != 5400:
        raise ValueError("export certificate lacks the nine fixed runs")
    with (ROOT / "data/raw/botsim_24/Users.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        botsim_users = {row["user_id"]: row for row in csv.DictReader(stream)}
    botsim_activity = json.loads((ROOT /
        "data/raw/botsim_24/user_post_comment.json").read_text(encoding="utf-8"))
    if set(botsim_users) != set(botsim_activity) or len(botsim_users) != 2907:
        raise ValueError("BotSim source population differs")
    runs = []
    for run in export["runs"]:
        dataset, seed = run["dataset"], int(run["seed"])
        saved_path = ROOT / SAVED[dataset] / f"seed{seed}/test_predictions.json"
        if sha256_file(saved_path) != run["saved_predictions_sha256"]:
            raise ValueError("saved prediction file changed")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        path = Path(run["cards_file"])
        if sha256_file(path) != run["cards_sha256"]:
            raise ValueError("compressed card file changed")
        cards = 0
        terms = 0
        projection_terms = 0
        changed = 0
        max_logit_difference = 0.0
        max_probability_difference = 0.0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if cards >= len(saved):
                    raise ValueError("more cards than frozen accounts")
                card = json.loads(line)
                expected = saved[cards]
                if (card["dataset"] != dataset or card["seed"] != seed or
                        card["account_id"] != str(expected["account_id"]) or
                        card["classification_labels_accessed"] is not False or
                        card["caller_supplied_fact_matrix_accepted"] is not False or
                        any(key in card for key in ("label", "true_label",
                                                "ground_truth"))):
                    raise ValueError("card identity, order or label-free contract differs")
                items = card["evidence_terms"]
                if len(items) != COUNTS[dataset] or len({
                        item["name"] for item in items}) != len(items):
                    raise ValueError("card evidence terms differ")
                logit = float(card["intercept_logit"]) + math.fsum(
                    float(item["logit_attribution"]) for item in items)
                max_logit_difference = max(max_logit_difference,
                    abs(logit - float(card["logit"])))
                probability = float(expit(logit))
                max_probability_difference = max(max_probability_difference,
                    abs(probability - float(expected["probability"])))
                if (abs(logit - float(card["logit"])) > 1e-9 or
                        abs(probability - float(card["probability"])) > 1e-12 or
                        abs(probability - float(expected["probability"])) > 1e-12 or
                        int(probability >= float(card["threshold"])) !=
                        int(card["prediction"])):
                    raise ValueError("source terms do not reconstruct frozen decision")
                for item in items:
                    if dataset == "botsim":
                        verify_botsim_scope(item, card["account_id"],
                                            botsim_users, botsim_activity)
                    if (abs(float(item["upper_model_input"]) -
                            math.log1p(float(item["upper_tail"])) *
                            float(item["reliability"])) > 1e-12 or
                            abs(float(item["lower_model_input"]) -
                            math.log1p(float(item["lower_tail"])) *
                            float(item["reliability"])) > 1e-12):
                        raise ValueError("recorded model input differs from evidence")
                    if item["source_fact_observed"] and not item["source_query"]:
                        raise ValueError("observed source fact lacks query")
                    if dataset == "mgtab" and "_neighbor_" in item["name"]:
                        source_ids = item["source_query"]["source_ids"]
                        columns = {
                            "default_profile_mean": 1,
                            "verified_mean": 2,
                            "description_length_mean": 13,
                            "followers_friends_ratio_mean": 14,
                        }
                        column = next((value for suffix, value in columns.items()
                                       if item["name"].endswith(
                                           "_neighbor_" + suffix)), None)
                        if (column is None or not any(
                                "features.pt#neighbor_rows,column=" + str(column)
                                in source for source in source_ids)):
                            raise ValueError("neighbor mean lacks profile tensor column")
                    projection_terms += int(item["source_level"] ==
                        "published_averaged_tweet_tensor_train_only_projection")
                changed += int(card["exact_frozen_model_dependency"][
                    "all_evidence_changes_class"])
                cards += 1
                terms += len(items)
        if (cards != run["accounts"] or terms != run["evidence_terms"] or
                projection_terms != run[
                    "published_averaged_tweet_projection_terms"] or
                changed != run["all_evidence_changes_class"]):
            raise ValueError("full card readback counts differ")
        runs.append({"dataset": dataset, "seed": seed,
                     "accounts": cards, "terms": terms,
                     "max_logit_difference": max_logit_difference,
                     "max_probability_difference": max_probability_difference})
        print(runs[-1], flush=True)
    OUT.write_text(json.dumps({
        "status": "all_full_test_cards_read_back_and_reconstructed",
        "source_export_certificate_sha256": sha256_file(CERTIFICATE),
        "total_accounts": sum(run["accounts"] for run in runs),
        "total_terms": sum(run["terms"] for run in runs),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
