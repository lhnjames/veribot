"""Rebuild the VeriBot Fox8 account input columns from original records."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS, TWEET_FIELDS

RAW_PROFILE_FIELDS = (
    "followers_count", "friends_count", "statuses_count", "favourites_count",
    "listed_count", "verified", "default_profile", "default_profile_image",
    "protected", "geo_enabled", "profile_use_background_image",
    "has_extended_profile", "contributors_enabled", "is_translator",
    "is_translation_enabled", "profile_background_tile", "description",
    "name", "screen_name", "location", "url",
)


def profile_features(user: dict) -> dict[str, float]:
    numeric = ("followers_count", "friends_count", "statuses_count",
               "favourites_count", "listed_count")
    boolean = ("verified", "default_profile", "default_profile_image",
               "protected", "geo_enabled", "profile_use_background_image",
               "has_extended_profile", "contributors_enabled", "is_translator",
               "is_translation_enabled", "profile_background_tile")
    values = {key: user.get(key, 0) or 0 for key in numeric}
    values.update({key: int(user.get(key, False) or False) for key in boolean})
    values.update({
        "description_len": len(user.get("description", "") or ""),
        "name_len": len(user.get("name", "") or ""),
        "screen_name_len": len(user.get("screen_name", "") or ""),
        "has_location": int(bool(user.get("location", ""))),
        "has_url": int(bool(user.get("url", ""))),
        "has_description": int(bool(user.get("description", ""))),
        "ff_ratio": (user.get("followers_count", 0) or 0)
                    / max((user.get("friends_count", 0) or 0) + 1, 1),
    })
    if tuple(values) != PROFILE_FIELDS:
        raise ValueError("raw profile formula order differs from model schema")
    return values


def tweet_features(tweets: list[dict]) -> dict[str, float]:
    if not tweets:
        return {key: 0.0 for key in TWEET_FIELDS}
    count = len(tweets)
    texts = [tweet.get("text", "") or "" for tweet in tweets]
    favorites = [tweet.get("favorite_count", 0) or 0 for tweet in tweets]
    retweets = [tweet.get("retweet_count", 0) or 0 for tweet in tweets]
    languages = [tweet.get("lang", "und") for tweet in tweets]
    values = {
        "n_tweets": count,
        "retweet_ratio": sum(text.startswith("RT ") for text in texts) / count,
        "reply_ratio": sum(tweet.get("in_reply_to_status_id") is not None
                           for tweet in tweets) / count,
        "avg_tweet_len": float(np.mean([len(text) for text in texts])),
        "avg_favorites": float(np.mean(favorites)),
        "avg_retweets": float(np.mean(retweets)),
        "avg_score": float(np.mean([f + r for f, r in zip(favorites, retweets)])),
        "unique_lang_ratio": len(set(languages)) / count,
    }
    if tuple(values) != TWEET_FIELDS:
        raise ValueError("raw tweet formula order differs from model schema")
    return values


def account_features(row: dict) -> dict[str, float]:
    tweets = row.get("user_tweets")
    if not isinstance(tweets, list) or not tweets:
        raise ValueError("raw account has no tweet list")
    profile = tweets[0].get("user")
    if not isinstance(profile, dict):
        raise ValueError("raw account has no first-tweet profile")
    if str(profile.get("id_str", profile.get("id"))) != str(row["user_id"]):
        raise ValueError("raw account and profile IDs differ")
    return {**profile_features(profile), **tweet_features(tweets)}


def rebuild_feature_matrix(
    raw_path: Path, account_ids: list[str],
) -> tuple[np.ndarray, dict[str, int], int]:
    """Read the first released record per account; never read its label field."""
    wanted = set(account_ids)
    if len(wanted) != len(account_ids):
        raise ValueError("account inventory contains duplicate IDs")
    rows = {}
    locations = {}
    raw_rows = 0
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            raw_rows += 1
            uid = str(row["user_id"])
            if uid not in wanted or uid in rows:
                continue
            rows[uid] = account_features(row)
            locations[uid] = line_no
    if set(rows) != wanted:
        raise ValueError(f"raw source lacks {len(wanted - set(rows))} model accounts")
    matrix = np.asarray([[float(rows[uid][name]) for name in FEATURE_FIELDS]
                         for uid in account_ids], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("raw-derived model facts are non-finite")
    return matrix, locations, raw_rows


def read_first_profile(raw_path: Path, account_id: str) -> tuple[dict, int]:
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if str(row["user_id"]) == account_id:
                account_features(row)
                return dict(row["user_tweets"][0]["user"]), line_no
    raise KeyError(account_id)
