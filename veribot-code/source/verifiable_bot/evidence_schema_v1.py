"""Label-free, source-addressable candidate evidence vocabulary.

This module fixes *what* an observation means before any class-direction or
classifier fitting.  It does not claim that an empirical tail is semantically
valid or that these concepts are sufficient for classification.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _entries(rows: tuple[tuple[str, str, str], ...]) -> list[dict]:
    return [{"fact": name, "meaning": meaning, "group": group,
             "evidence_type": "computed_empirical_tail",
             "class_direction": "unset_until_disjoint_training_A"}
            for name, meaning, group in rows]


SCHEMA = {
    "mgtab": {
        "source_depth": "published_normalized_profile_and_typed_edge_tensors",
        "source_files": ["features.pt", "edge_index.pt", "edge_type.pt"],
        "reference_context": {
            "fact": "published_profile_coordinate_06",
            "meaning": "published normalized status volume",
            "rule": "unlabeled_training_tertiles_with_global_fallback",
            "minimum_observed_accounts_per_stratum": 30,
        },
        "missing_rule": "nonfinite tensor value is missing; absent typed edge is an observed zero",
        "reliability": "training reference size and missing rate; no raw-record validity claim",
        "permitted_correction": "independently verified correction to a released tensor entry, followed by recomputation of all dependent facts",
        "excluded_from_named_evidence": ["published_tweet_projection_*", "account_id", "Bot/Human label"],
        "concepts": _entries((
            ("published_profile_coordinate_02", "published verified flag", "profile_status"),
            ("published_profile_coordinate_03", "published normalized follower count", "profile_audience"),
            ("published_profile_coordinate_05", "published normalized listed count", "profile_audience"),
            ("published_profile_coordinate_06", "published normalized status count", "profile_activity"),
            ("published_profile_coordinate_07", "published normalized friend count", "profile_audience"),
            ("published_profile_coordinate_09", "published normalized favorite count", "profile_activity"),
            ("published_profile_coordinate_13", "published normalized description length", "profile_presentation"),
            ("published_profile_coordinate_14", "published normalized follower/friend ratio", "profile_audience"),
            ("published_followers_nonself_out_unique_log1p", "unique outgoing followers-type neighbors in released graph", "follow_relations"),
            ("published_followers_nonself_in_unique_log1p", "unique incoming followers-type neighbors in released graph", "follow_relations"),
            ("published_friends_nonself_out_unique_log1p", "unique outgoing friends-type neighbors in released graph", "follow_relations"),
            ("published_friends_nonself_in_unique_log1p", "unique incoming friends-type neighbors in released graph", "follow_relations"),
            ("published_mention_nonself_out_unique_log1p", "unique outgoing mention-type neighbors in released graph", "communication_relations"),
            ("published_reply_nonself_out_unique_log1p", "unique outgoing reply-type neighbors in released graph", "communication_relations"),
            ("published_reply_reciprocal_unique_log1p", "reciprocal reply-type neighbors in released graph", "communication_relations"),
            ("published_quoted_nonself_out_unique_log1p", "unique outgoing quote-type neighbors in released graph", "sharing_relations"),
            ("published_url_nonself_out_unique_log1p", "unique outgoing URL-type neighbors in released graph", "sharing_relations"),
            ("published_hashtag_nonself_out_unique_log1p", "unique outgoing hashtag-type neighbors in released graph", "sharing_relations"),
        )),
    },
    "fox8": {
        "source_depth": "raw_profile_tweets_and_observed_reply_retweet_targets",
        "source_files": ["fox8_23_dataset.ndjson.gz"],
        "reference_context": {
            "fact": "n_tweets", "meaning": "released tweet sample size",
            "rule": "unlabeled_training_tertiles_with_global_fallback",
            "minimum_observed_accounts_per_stratum": 30,
        },
        "missing_rule": "empty tweet sample makes tweet ratios/means missing; absent target links yield an observed zero",
        "reliability": "training reference size, missing rate and tweet/neighbor support where applicable",
        "permitted_correction": "independently verified profile or tweet-source edit; recompute all linked aggregates and native target facts",
        "excluded_from_named_evidence": ["feature-derived kNN edges", "account_id", "Bot/Human label"],
        "concepts": _entries((
            ("followers_count", "profile follower count", "profile_audience"),
            ("friends_count", "profile friend count", "profile_audience"),
            ("listed_count", "profile listed count", "profile_audience"),
            ("statuses_count", "profile total status count", "profile_activity"),
            ("description_len", "profile description character count", "profile_presentation"),
            ("ff_ratio", "profile follower count divided by friend count plus one", "profile_audience"),
            ("n_tweets", "released tweet sample size", "sample_activity"),
            ("retweet_ratio", "fraction of sampled tweet text beginning with 'RT '", "sample_activity_mix"),
            ("reply_ratio", "fraction of sampled tweets with a non-null in_reply_to_status_id", "sample_activity_mix"),
            ("avg_tweet_len", "mean sampled tweet character count", "sample_text"),
            ("unique_lang_ratio", "fraction of distinct language codes in sampled tweets", "sample_text"),
            ("avg_favorites", "mean favorite count on sampled tweets", "sample_response"),
            ("native_reply_out_unique_targets", "distinct sampled reply recipients present in released accounts", "native_reply"),
            ("native_reply_in_unique_sources", "distinct released accounts replying to this account", "native_reply"),
            ("native_retweet_out_unique_targets", "distinct sampled retweet targets present in released accounts", "native_retweet"),
            ("native_retweet_in_unique_sources", "distinct released accounts retweeting this account", "native_retweet"),
        )),
    },
    "botsim": {
        "source_depth": "raw_user_rows_posts_comments_and_observed_reply_objects",
        "source_files": ["Users.csv", "user_post_comment.json"],
        "reference_context": {
            "fact": "n_posts", "meaning": "observed post count",
            "rule": "unlabeled_training_tertiles_with_global_fallback",
            "minimum_observed_accounts_per_stratum": 30,
        },
        "missing_rule": "an absent mean denominator is missing, not a literal zero; observed event counts may be zero",
        "reliability": "training reference size, missing rate and post/comment denominator or graph support",
        "permitted_correction": "independently verified user, post, or comment-source edit; recompute dependent aggregates and reply graph",
        "excluded_from_named_evidence": ["avg_upvote_ratio", "avg_post_score", "avg_comment_score", "feature-derived kNN edges", "account_id", "Bot/Human label"],
        "concepts": _entries((
            ("submission_num", "submission count declared in the user row", "declared_activity"),
            ("comment_num", "comment count declared in the user row", "declared_activity"),
            ("has_description", "whether the user row has a description", "presentation"),
            ("description_len", "description character count", "presentation"),
            ("n_subreddits", "number of subreddit entries declared in the user row", "community"),
            ("n_posts", "number of observed posts", "observed_activity"),
            ("n_comments_received", "sum of comments received by observed posts", "observed_activity"),
            ("n_comments_made", "number of observed comment records", "observed_activity"),
            ("avg_post_len", "mean character count of observed posts", "text_length"),
            ("avg_comment_len", "mean character count of observed comment_1 bodies", "text_length"),
            ("n_unique_subs", "distinct subreddits among observed posts", "community"),
            ("native_reply_out_unique_log1p", "log unique observed reply recipients plus one", "native_reply"),
            ("native_reply_in_unique_log1p", "log unique observed incoming reply authors plus one", "native_reply"),
            ("native_reply_mutual_unique_log1p", "log reciprocal reply partners plus one", "native_reply"),
            ("native_reply_out_comments_log1p", "log observed outgoing reply comments plus one", "native_reply"),
            ("native_reply_in_comments_log1p", "log observed incoming reply comments plus one", "native_reply"),
        )),
    },
}


def _source_contract(dataset: str, fact: str) -> tuple[str, str]:
    if dataset == "mgtab":
        if fact.startswith("published_profile_coordinate_"):
            index = int(fact.rsplit("_", 1)[1])
            return (f"features.pt[row,{index}]",
                    "mgtab_profile_schema.PROFILE_SCHEMA + tensor coordinate lookup")
        return ("edge_index.pt + edge_type.pt, typed incident edge rows",
                "mgtab_structural_facts.extract: log1p unique nonself/reciprocal count")
    if dataset == "fox8":
        if fact.startswith("native_"):
            return ("fox8_23_dataset.ndjson.gz, user_tweets target identifiers",
                    "fox8_native_interaction_facts.build")
        if fact in _source_tuple("fox8_adapter", "PROFILE_FIELDS"):
            return ("fox8_23_dataset.ndjson.gz, first profile user object",
                    "fox8_adapter.profile_features")
        return ("fox8_23_dataset.ndjson.gz, user_tweets array",
                "fox8_adapter.tweet_features")
    if fact.startswith("native_"):
        return ("user_post_comment.json, comment and parent object identifiers",
                "botsim_native_facts.extract")
    if fact in {"submission_num", "comment_num", "has_description",
                "description_len", "n_subreddits"}:
        return ("Users.csv, matching user_id row",
                "botsim_literal_zero_runtime._account_facts")
    return ("user_post_comment.json, matching user_id activity object",
            "botsim_literal_zero_runtime._account_facts(observed_means_only=True)")


def _source_tuple(module: str, variable: str) -> tuple:
    """Read a literal extractor vocabulary without importing model dependencies."""
    path = Path(__file__).with_name(f"{module}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                target.id == variable for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"{module}.{variable} has no static vocabulary")


def validate_schema() -> dict[str, int]:
    profile = _source_tuple("mgtab_profile_schema", "PROFILE_SCHEMA")
    relations = _source_tuple("mgtab_source_certificate", "RELATIONS")
    fox8_fields = (_source_tuple("fox8_adapter", "PROFILE_FIELDS") +
                   _source_tuple("fox8_adapter", "TWEET_FIELDS"))
    fox8_native = _source_tuple("fox8_native_interaction_facts", "FACT_NAMES")
    botsim_fields = _source_tuple("botsim_adapter", "FEATURE_FIELDS")
    botsim_native = _source_tuple("botsim_native_facts", "FACT_NAMES")
    mgtab_allowed = {f"published_profile_coordinate_{i:02d}"
                      for i in range(len(profile))}
    mgtab_allowed |= {f"published_{r}_{suffix}" for r in relations for suffix in (
        "nonself_out_unique_log1p", "nonself_in_unique_log1p",
        "reciprocal_unique_log1p", "self_edge_rows_log1p")}
    allowed = {
        "mgtab": mgtab_allowed,
        "fox8": set(fox8_fields) | set(fox8_native),
        "botsim": set(botsim_fields) | set(botsim_native),
    }
    counts = {}
    for dataset, spec in SCHEMA.items():
        concepts = spec["concepts"]
        names = [row["fact"] for row in concepts]
        if not 10 <= len(names) <= 20 or len(set(names)) != len(names):
            raise ValueError(f"{dataset}: concept count or uniqueness differs")
        if set(names) - allowed[dataset] or spec["reference_context"]["fact"] not in names:
            raise ValueError(f"{dataset}: concept name or reference context differs from extractor")
        if any(not row["meaning"] or not row["group"] or
               row["class_direction"] != "unset_until_disjoint_training_A"
               for row in concepts):
            raise ValueError(f"{dataset}: incomplete or label-directed evidence definition")
        if set(names) & set(spec["excluded_from_named_evidence"]):
            raise ValueError(f"{dataset}: excluded field entered named evidence")
        counts[dataset] = len(names)
    return counts


def export(path: Path) -> dict[str, int]:
    counts = validate_schema()
    for dataset, spec in SCHEMA.items():
        for concept in spec["concepts"]:
            selector, operation = _source_contract(dataset, concept["fact"])
            concept["source_selector"] = selector
            concept["calculation_entry"] = operation
            concept["missing_rule"] = (
                "undefined when denominator is zero; emit missing marker"
                if dataset == "botsim" and concept["fact"] in
                {"avg_post_len", "avg_comment_len"} else
                "undefined when n_tweets is zero; emit missing marker"
                if dataset == "fox8" and concept["fact"] in
                {"retweet_ratio", "reply_ratio", "avg_tweet_len", "avg_favorites"} else
                spec["missing_rule"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "candidate-v1", "status":
        "predeclared_vocabulary_not_validated_semantic_truth_or_gate1",
        "schema": SCHEMA}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return counts
