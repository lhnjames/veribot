"""Fixed within-group source-fact conjunctions for TwiBot-20."""

from __future__ import annotations


PAIRS = (
    ("followers_count", "friends_count"),
    ("followers_count", "listed_count"),
    ("statuses_count", "favourites_count"),
    ("verified", "default_profile"),
    ("default_profile", "default_profile_image"),
    ("verified", "description_length"),
    ("default_profile", "description_length"),
    ("retweet_fraction", "url_fraction"),
    ("retweet_fraction", "mention_fraction"),
    ("hashtag_fraction", "mention_fraction"),
    ("tweet_sample_count", "unique_tweet_fraction"),
    ("url_fraction", "hashtag_fraction"),
    ("sampled_following_count", "graph_out_degree"),
    ("sampled_follower_count", "graph_in_degree"),
    ("graph_out_degree", "graph_in_degree"),
    ("graph_out_degree", "graph_reciprocal_degree"),
    ("graph_in_degree", "graph_reciprocal_degree"),
)


def fixed_pairs(names: list[str]) -> tuple[tuple[str, str], ...]:
    if any(left not in names or right not in names for left, right in PAIRS):
        raise ValueError("TwiBot-20 pair feature missing from fixed vocabulary")
    return PAIRS
