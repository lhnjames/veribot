"""Named, exhaustive dimensions for the Fox8 evidence-view inputs.

The groups describe what a raw fact measures; none asserts that the fact
proves either account class. A selected tweet can contribute to several facts.
"""

from __future__ import annotations

from .fox8_two_stage_inputs import ALL_FACTS


FACT_DIMENSIONS = {
    "network_scale": (
        "followers_count", "friends_count", "listed_count", "ff_ratio",
    ),
    "account_activity": ("statuses_count", "favourites_count"),
    "profile_configuration": (
        "verified", "default_profile", "default_profile_image", "protected",
        "geo_enabled", "profile_use_background_image", "has_extended_profile",
        "contributors_enabled", "is_translator", "is_translation_enabled",
        "profile_background_tile",
    ),
    "profile_presentation": (
        "description_len", "name_len", "screen_name_len", "has_location",
        "has_url", "has_description",
    ),
    "posting_and_interaction": (
        "n_tweets", "retweet_ratio", "reply_ratio", "quote_ratio",
    ),
    "engagement": ("avg_favorites", "avg_retweets", "avg_score"),
    "temporal_pattern": (
        "observed_span_hours", "median_intertweet_gap_hours", "gap_cv",
    ),
    "text_surface": (
        "avg_tweet_len", "mention_rate", "hashtag_rate", "url_rate",
        "exact_text_duplicate_ratio",
    ),
    "language_metadata": ("unique_lang_ratio", "language_entropy"),
}

GRAPH_DIMENSIONS = ("reply", "retweet")


def validate_dimensions() -> None:
    listed = tuple(name for group in FACT_DIMENSIONS.values() for name in group)
    if len(listed) != len(set(listed)) or set(listed) != set(ALL_FACTS):
        missing = set(ALL_FACTS) - set(listed)
        extra = set(listed) - set(ALL_FACTS)
        raise ValueError(f"dimension partition differs: missing={missing}, extra={extra}")


validate_dimensions()
