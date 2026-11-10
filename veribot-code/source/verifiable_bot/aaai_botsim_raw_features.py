"""Rebuild BotSim's 17 named user facts from Users.csv and activity JSON."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np


FEATURE_FIELDS = (
    "submission_num", "comment_num", "comment_num_1", "comment_num_2",
    "has_description", "description_len", "n_subreddits", "n_posts",
    "n_comments_received", "n_comments_made", "avg_post_score",
    "avg_upvote_ratio", "avg_post_len", "avg_comment_score",
    "avg_comment_len", "n_unique_subs", "is_active",
)


def subreddits(row: dict) -> list[str]:
    value = row.get("subreddit") or "[]"
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("invalid raw subreddit list")
    return parsed


def account_features(user: dict, activity: dict) -> dict[str, float]:
    posts = activity.get("posts", [])
    comments_1 = activity.get("comment_1", [])
    comments_2 = activity.get("comment_2", [])
    scores = [post.get("score", 0) or 0 for post in posts]
    post_comments = [post.get("num_comments", 0) or 0 for post in posts]
    upvotes = [post.get("upvote_ratio", 0.5) or 0.5 for post in posts]
    post_lengths = [len(post.get("posts", "") or "") for post in posts]
    comment_scores = [comment.get("comment_score", 0) or 0
                      for comment in comments_1]
    comment_lengths = [len(comment.get("comment_body", "") or "")
                       for comment in comments_1]
    description = user.get("description") or ""
    result = {
        "submission_num": float(user.get("submission_num") or 0),
        "comment_num": float(user.get("comment_num") or 0),
        "comment_num_1": float(user.get("comment_num_1") or 0),
        "comment_num_2": float(user.get("comment_num_2") or 0),
        "has_description": int(bool(description)),
        "description_len": len(description),
        "n_subreddits": len(subreddits(user)),
        "n_posts": len(posts),
        "n_comments_received": int(np.sum(post_comments)) if post_comments else 0,
        "n_comments_made": len(comments_1) + len(comments_2),
        "avg_post_score": float(np.mean(scores)) if scores else 0.0,
        "avg_upvote_ratio": float(np.mean(upvotes)) if upvotes else 0.5,
        "avg_post_len": float(np.mean(post_lengths)) if post_lengths else 0.0,
        "avg_comment_score": float(np.mean(comment_scores)) if comment_scores else 0.0,
        "avg_comment_len": float(np.mean(comment_lengths)) if comment_lengths else 0.0,
        "n_unique_subs": len(set(post.get("subreddit", "") for post in posts)),
        "is_active": int(bool(posts or comments_1 or comments_2)),
    }
    if tuple(result) != FEATURE_FIELDS:
        raise ValueError("raw feature formulas differ from model schema")
    return result


def rebuild_feature_matrix(users_path: Path, activity_path: Path,
                           account_ids: list[str]) -> np.ndarray:
    with users_path.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    ids = [str(row["user_id"]) for row in users]
    if ids != account_ids or len(set(ids)) != len(ids):
        raise ValueError("raw and model account inventories differ")
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    if set(activity) != set(ids):
        raise ValueError("raw activity and user records differ")
    matrix = np.asarray([
        [float(account_features(user, activity[uid])[field])
         for field in FEATURE_FIELDS]
        for user, uid in zip(users, ids)
    ], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("raw-derived BotSim feature is non-finite")
    return matrix
