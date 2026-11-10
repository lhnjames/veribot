"""v4 numeric features: the single source of truth for the 33-dimensional account vector.

The ORDER of ``FEATURE_NAMES`` must stay byte-identical to ``scripts/rf_baseline.py``'s ``NAMES``.
Two things depend on it:

* every baseline number quoted in the design doc (RF 83.01% / TF-IDF 72.53% / logit-avg 85.04%)
  was produced with this exact ordering, so a silent reorder would invalidate the comparison;
* E1 concatenates this vector onto the LLM's pooled representation, and the frozen bucket
  thresholds are stored per-index.

``tests/test_render_v4.py::test_feature_order_matches_rf_baseline`` pins the names by parsing
``scripts/rf_baseline.py`` with ``ast``, and
``test_feature_values_match_rf_baseline`` re-derives every value with a verbatim copy of that
script's ``feats()`` and asserts elementwise equality on real accounts. Change either side without
the other and the suite fails.

Division guards: every denominator goes through ``max(x, 1)`` exactly as the baseline does, so an
account with zero tweets / zero followers / unparseable ``created_at`` yields zeros rather than a
ZeroDivisionError. Accounts with no tweets still get their eight tweet-statistic slots filled with
0.0 here (that is what the baseline did, and E1 must see the same vector); it is the *renderer*
that omits ACTIVITY_xx units for them, see render_v4.py.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

# Order-critical. Mirrors scripts/rf_baseline.py::NAMES exactly.
FEATURE_NAMES: tuple[str, ...] = (
    "log_followers",        # 0
    "log_friends",          # 1
    "log_statuses",         # 2
    "log_favourites",       # 3
    "log_listed",           # 4
    "friends/followers",    # 5
    "followers/friends",    # 6
    "statuses/age",         # 7
    "fav/age",              # 8
    "followers/age",        # 9
    "friends/age",          # 10
    "log_age",              # 11
    "len_name",             # 12
    "len_screenname",       # 13
    "digits_in_sn",         # 14
    "digit_ratio_sn",       # 15
    "len_desc",             # 16
    "has_desc",             # 17
    "verified",             # 18
    "default_profile",      # 19
    "default_prof_img",     # 20
    "geo_enabled",          # 21
    "protected",            # 22
    "has_url",              # 23
    "has_location",         # 24
    "n_tweets",             # 25
    "retweet_ratio",        # 26
    "mean_tweet_len",       # 27
    "std_tweet_len",        # 28
    "uniq_tweet_ratio",     # 29
    "url_per_tweet",        # 30
    "hashtag_per_tweet",    # 31
    "mention_per_tweet",    # 32
)

N_FEATURES = len(FEATURE_NAMES)

# Indices whose value is already a 0/1 flag: rendered verbatim, never bucketed.
BINARY_INDICES: frozenset[int] = frozenset(range(17, 25))
# Indices derived from the account's timeline: undefined when the account has no tweets, so their
# bucket thresholds are fitted only over train accounts that do have tweets (see thresholds_v4.py).
TWEET_STAT_INDICES: tuple[int, ...] = tuple(range(25, 33))
CONTINUOUS_INDICES: tuple[int, ...] = tuple(i for i in range(N_FEATURES) if i not in BINARY_INDICES)

# TwiBot-20's collection cutoff. Hard-coded (not "now") so the vector is reproducible forever, and
# identical to scripts/rf_baseline.py.
REFERENCE_DATE = datetime(2020, 9, 1)
_CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def clean_str(value: Any) -> Any:
    """TwiBot-20 pads almost every scalar with a trailing space; strip it before anything else."""
    return (value or "").strip() if isinstance(value, str) else value


def to_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def to_bool(value: Any) -> float:
    """TwiBot-20 stores booleans as the strings 'True '/'False '."""
    return 1.0 if str(value).strip() == "True" else 0.0


def account_age_days(profile: dict) -> float:
    """Days from account creation to REFERENCE_DATE, floored at 1 so it is always a safe divisor."""
    try:
        created = datetime.strptime(clean_str(profile.get("created_at")), _CREATED_AT_FORMAT)
        age = (REFERENCE_DATE - created.replace(tzinfo=None)).days
    except (TypeError, ValueError):
        age = 0
    return float(max(age, 1))


def tweet_list(record: dict) -> list:
    """Raw timeline entries. ``None`` (some accounts) degrades to an empty list, never a crash."""
    return record.get("tweet") or []


def neighbor_counts(record: dict) -> tuple[int, int] | None:
    """(following_count, follower_count) from the sampled ``neighbor`` field, or ``None``.

    These counts are NOT part of the 33-dim E1 vector (the RF baselines never used them); they feed
    only the RELATION_xx prompt unit. ``neighbor`` is ``None`` for ~9% of accounts and both lists
    are capped at the sampling limit (10), so this is a coarse, bounded signal — rendered as a tier
    against train-fitted cuts like everything else, never as a raw number. Returns ``None`` when the
    field is missing or both lists are empty, so the renderer emits zero RELATION units rather than
    fabricating one.
    """
    neighbor = record.get("neighbor")
    if not isinstance(neighbor, dict):
        return None
    following = len(neighbor.get("following") or [])
    follower = len(neighbor.get("follower") or [])
    if following + follower == 0:
        return None
    return following, follower


def extract_features(record: dict) -> list[float]:
    """The 33-dim vector for one raw TwiBot-20 record. Non-finite values are zeroed, matching the
    baseline's ``np.nan_to_num(X, posinf=0, neginf=0)``."""
    profile = record.get("profile") or {}

    followers = to_float(profile.get("followers_count"))
    friends = to_float(profile.get("friends_count"))
    statuses = to_float(profile.get("statuses_count"))
    favourites = to_float(profile.get("favourites_count"))
    listed = to_float(profile.get("listed_count"))
    age = account_age_days(profile)

    name = clean_str(profile.get("name")) or ""
    screen_name = clean_str(profile.get("screen_name")) or ""
    description = clean_str(profile.get("description")) or ""
    sn_digits = sum(c.isdigit() for c in screen_name)

    tweets = tweet_list(record)
    n_tweets = len(tweets)
    retweets = sum(1 for t in tweets if isinstance(t, str) and t.startswith("RT @"))
    lengths = [len(t) for t in tweets if isinstance(t, str)]
    distinct_ratio = len(set(tweets)) / max(n_tweets, 1)
    urls = sum(1 for t in tweets if isinstance(t, str) and "http" in t)
    hashtags = sum(t.count("#") for t in tweets if isinstance(t, str))
    mentions = sum(t.count("@") for t in tweets if isinstance(t, str))

    values = [
        math.log1p(followers),
        math.log1p(friends),
        math.log1p(statuses),
        math.log1p(favourites),
        math.log1p(listed),
        friends / max(followers, 1),
        followers / max(friends, 1),
        statuses / age,
        favourites / age,
        followers / age,
        friends / age,
        math.log1p(age),
        float(len(name)),
        float(len(screen_name)),
        float(sn_digits),
        sn_digits / max(len(screen_name), 1),
        float(len(description)),
        1.0 if description else 0.0,
        to_bool(profile.get("verified")),
        to_bool(profile.get("default_profile")),
        to_bool(profile.get("default_profile_image")),
        to_bool(profile.get("geo_enabled")),
        to_bool(profile.get("protected")),
        1.0 if clean_str(profile.get("url")) else 0.0,
        1.0 if clean_str(profile.get("location")) else 0.0,
        float(n_tweets),
        retweets / max(n_tweets, 1),
        float(np.mean(lengths)) if lengths else 0.0,
        float(np.std(lengths)) if lengths else 0.0,
        distinct_ratio,
        urls / max(n_tweets, 1),
        hashtags / max(n_tweets, 1),
        mentions / max(n_tweets, 1),
    ]
    assert len(values) == N_FEATURES, f"expected {N_FEATURES} features, built {len(values)}"
    return [0.0 if not math.isfinite(v) else float(v) for v in values]


def feature_matrix(records: list[dict]) -> np.ndarray:
    return np.asarray([extract_features(r) for r in records], dtype=np.float64)
