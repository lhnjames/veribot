"""Disjoint, source-addressable groups for budgeted Fox8 coalition search."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .aaai_fox8_raw_features import RAW_PROFILE_FIELDS


COUNT_FIELDS = frozenset(("followers_count", "friends_count", "statuses_count",
                          "favourites_count", "listed_count"))
TEXT_FIELDS = frozenset(("description", "name", "screen_name", "location",
                         "url"))
FLAG_FIELDS = frozenset(RAW_PROFILE_FIELDS) - COUNT_FIELDS - TEXT_FIELDS
if COUNT_FIELDS & TEXT_FIELDS or (COUNT_FIELDS | TEXT_FIELDS | FLAG_FIELDS !=
                                  frozenset(RAW_PROFILE_FIELDS)):
    raise ValueError("Fox8 profile field groups do not partition the schema")


def fine_grained_target_groups(*, account, target_profile: set[str],
                               target_tweets: set[str],
                               incident_relations: Iterable,
                               selected_relations: set[str],
                               neighbor_profile: set[str],
                               neighbor_tweets: set[str],
                               hgt_remainder: frozenset[str]) -> dict[str, frozenset[str]]:
    """Partition selected target sources by raw modality and relation type."""
    groups: dict[str, set[str]] = defaultdict(set)
    for field in account.profile_field_ids:
        source_id = account.source_id("profile", field)
        if source_id not in target_profile:
            continue
        category = ("counts" if field in COUNT_FIELDS else
                    "text" if field in TEXT_FIELDS else "flags")
        groups[f"target_profile_{category}"].add(source_id)
    for tweet in account.tweets:
        source_id = account.source_id("tweet", tweet.ordinal)
        if source_id not in target_tweets:
            continue
        category = ("retweet" if tweet.is_retweet else
                    "reply" if tweet.is_reply else
                    "quote" if tweet.is_quote else "original")
        groups[f"target_tweets_{category}"].add(source_id)
    for relation in incident_relations:
        if relation.source_id in selected_relations:
            groups[f"incident_relation_{relation.relation}"].add(
                relation.source_id)
    groups["direct_neighbor_profile"].update(neighbor_profile)
    groups["direct_neighbor_tweets"].update(neighbor_tweets)
    groups["hgt_context_remainder"].update(hgt_remainder)

    expected = (target_profile | target_tweets | selected_relations |
                neighbor_profile | neighbor_tweets | set(hgt_remainder))
    result = {name: frozenset(sources) for name, sources in sorted(groups.items())
              if sources}
    listed = [source for sources in result.values() for source in sources]
    if len(listed) != len(set(listed)) or set(listed) != expected:
        raise ValueError("Fine-grained Fox8 groups do not partition sources")
    return result
