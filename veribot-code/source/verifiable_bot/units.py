"""Turns an AccountRecord into numbered, citable evidence units.

Every unit gets a unique id (PROFILE_xx, TWEET_xx, ACTIVITY_xx, RELATION_xx, FACT_xx, OBS_xx).
Only OBS_xx units are computed here from hidden raw values; they state qualitative facts and never use
judgmental words (abnormal / suspicious / bot-like / automated, ...). The model is only ever
allowed to cite ids that appear in this map, which is what makes the E -> claim references
independently checkable by a program instead of by another model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import AccountRecord

BANNED_WORDS = ("abnormal", "suspicious", "bot-like", "automated", "fake", "spam", "malicious")

MAX_TWEET_CHARS = 280
BURST_WINDOW_MINUTES = 15
BURST_MIN_COUNT = 5
SIMILARITY_THRESHOLD = 0.90
MAX_SIMILARITY_OBS = 6


@dataclass
class Unit:
    unit_id: str
    text: str
    hidden: dict = field(default_factory=dict)


def _digit_free(text: str) -> str:
    """Remove raw numeral strings from model-visible free text without discarding the text itself."""
    return re.sub(r"\d+(?:[.,:/-]\d+)*", "number", str(text))


def _profile_units(record: AccountRecord) -> list[Unit]:
    profile = record.profile
    order = ["description", "followers_count", "following_count", "tweet_count", "listed_count", "verified"]
    units = []
    labels = {
        "followers_count": "Follower count",
        "following_count": "Following count",
        "tweet_count": "Tweet count",
        "listed_count": "Listed count",
    }
    for key in order:
        if key not in profile:
            continue
        unit_id = f"PROFILE_{len(units) + 1:02d}"
        if key == "description":
            text = f"Description: {_digit_free(str(profile[key] or '')).strip()}"
            units.append(Unit(unit_id, text, {"kind": "text", "field": key, "raw_value": profile[key]}))
        elif key == "verified":
            value = bool(profile[key])
            text = "Verification status: verified." if value else "Verification status: not verified."
            units.append(Unit(unit_id, text, {"kind": "boolean", "field": key, "raw_value": value}))
        else:
            if key not in record.profile_buckets:
                raise ValueError(
                    f"record {record.source}/{record.account_id} has no fixed bucket for {key}; "
                    "refusing to expose a raw number"
                )
            bucket = record.profile_buckets[key]
            units.append(Unit(
                unit_id, f"{labels[key]}: {bucket}.",
                {"kind": "profile_bucket", "field": key, "raw_value": profile[key], "bucket": bucket,
                 "bucket_version": record.bucket_version},
            ))
    return units


def _tweet_units(posts) -> list[Unit]:
    units = []
    for i, post in enumerate(posts):
        raw_text = post.text.strip().replace("\n", " ")[:MAX_TWEET_CHARS]
        text = _digit_free(raw_text)
        units.append(Unit(f"TWEET_{i + 1:02d}", text, {"kind": "text", "raw_text": raw_text}))
    return units


def _activity_units(posts) -> list[Unit]:
    # Exact timestamps are intentionally hidden. Burst severity is emitted as an OBS unit below.
    return []


MAX_RELATION_EXAMPLES = 3


def _relation_units(edges) -> list[Unit]:
    """Cap at a handful of concrete example edges (was: one line per edge, up to 24 -- costly in
    tokens for very low information density) plus one aggregate directionality stat computed in
    _graph_ratio_obs. True reciprocity (does neighbor X appear as BOTH a follower and someone this
    account follows) is NOT computable here: edge.target values are per-account positional labels
    ("n0", "n1", ...) assigned independently for each relation direction in data.py, not a real
    neighbor id preserved across directions, so the same label in a "following" edge and a
    "follower" edge does not necessarily denote the same underlying account. Do not fabricate a
    reciprocity claim from this data -- it would require a data-pipeline change (data.py) to carry
    a real, direction-consistent neighbor identity, which this change does not make.
    """
    units = []
    seen = set()
    for edge in edges:
        if edge.relation in seen or len(units) >= MAX_RELATION_EXAMPLES:
            continue
        seen.add(edge.relation)
        if edge.relation == "following":
            text = "A sampled neighboring account is followed by this account."
        elif edge.relation == "follower":
            text = "A sampled neighboring account follows this account."
        else:
            text = "A sampled neighboring account has another recorded relation with this account."
        units.append(Unit(f"RELATION_{len(units) + 1:02d}", text, {"kind": "relation", "relation": edge.relation}))
    return units


def _graph_ratio_obs(edges) -> list[str]:
    """Local aggregate over the sampled graph_edges (no API call): what fraction of the observed
    connections are outgoing (following) vs incoming (follower). This is a coarser, but real,
    directional-imbalance signal -- distinct from _ratio_obs's PROFILE-level followers/following
    counts, since graph_edges is only a bounded sample (max_neighbors, see data.py), not the
    account's true global follower/following totals.
    """
    if not edges:
        return []
    following = sum(1 for e in edges if e.relation == "following")
    follower = sum(1 for e in edges if e.relation == "follower")
    other = len(edges) - following - follower
    if following + follower == 0:
        return []
    share = following / (following + follower)
    if share >= 0.75:
        descriptor = "mostly outgoing"
    elif share <= 0.25:
        descriptor = "mostly incoming"
    else:
        descriptor = "mixed-direction"
    return [(f"The sampled social connections are {descriptor}.", {
        "kind": "graph_direction", "descriptor": descriptor,
        "following_count": following, "follower_count": follower, "other_count": other,
    })]


def _fact_units(graph_facts: list[str]) -> list[Unit]:
    return [Unit(f"FACT_{i + 1:02d}", _digit_free(fact), {"kind": "graph_fact", "raw_text": fact})
            for i, fact in enumerate(graph_facts)]


def _graph_units(hgt_neighbor_facts: list[str]) -> list[Unit]:
    """GRAPH_xx units (design doc section 1): pre-aggregated, HGT-attention-derived neighbor facts.
    Empty input (the normal case for any account outside the clean HGT's sidecar) yields zero
    units, not an error. Digit-free is a hard build-time invariant here, same posture as
    BANNED_WORDS below for OBS_xx -- these units are supposed to arrive already stripped of any
    raw figure by the generation pipeline (hgt_import/build_graph_units.py); if one slips through,
    fail loudly at construction time rather than silently rendering it to the model.
    """
    units = [Unit(f"GRAPH_{i + 1:02d}", fact) for i, fact in enumerate(hgt_neighbor_facts)]
    for unit in units:
        if re.search(r"\d", unit.text):
            raise ValueError(f"unit {unit.unit_id} contains a digit, GRAPH_xx must stay digit-free: "
                              f"{unit.text!r}")
    return units


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _burst_obs(posts) -> list[str]:
    timed = [(i, _parse_time(p.created_at)) for i, p in enumerate(posts)]
    timed = [(i, t) for i, t in timed if t is not None]
    if len(timed) < BURST_MIN_COUNT:
        return []
    timed.sort(key=lambda pair: pair[1])
    times = [t for _, t in timed]
    best = 0
    left = 0
    for right in range(len(times)):
        while (times[right] - times[left]).total_seconds() > BURST_WINDOW_MINUTES * 60:
            left += 1
        best = max(best, right - left + 1)
    if best >= BURST_MIN_COUNT:
        if best >= 20:
            descriptor = "high-intensity burst"
        elif best >= 10:
            descriptor = "moderate burst"
        else:
            descriptor = "low-level burst"
        return [(f"The selected posts contain a {descriptor} within a short interval.", {
            "kind": "burst", "descriptor": descriptor, "raw_post_count": best,
            "raw_window_minutes": BURST_WINDOW_MINUTES,
        })]
    return []


def _ratio_obs(profile: dict) -> list[str]:
    followers = float(profile.get("followers_count", 0) or 0)
    following = float(profile.get("following_count", 0) or 0)
    obs = []
    if followers > 0 and following / followers >= 2.0:
        ratio = following / followers
        descriptor = "extreme imbalance" if ratio >= 20 else ("strong imbalance" if ratio >= 5 else "mild imbalance")
        obs.append((f"Following exceeds followers with an {descriptor}.", {
            "kind": "ratio", "direction": "following_over_followers", "descriptor": descriptor,
            "raw_ratio": ratio,
        }))
    elif following > 0 and followers / following >= 2.0:
        ratio = followers / following
        descriptor = "extreme imbalance" if ratio >= 20 else ("strong imbalance" if ratio >= 5 else "mild imbalance")
        obs.append((f"Followers exceed following with an {descriptor}.", {
            "kind": "ratio", "direction": "followers_over_following", "descriptor": descriptor,
            "raw_ratio": ratio,
        }))
    return obs


def _similarity_obs(tweet_units: list[Unit]) -> list[str]:
    texts = [u.text for u in tweet_units if len(u.text) >= 20]
    ids = [u.unit_id for u in tweet_units if len(u.text) >= 20]
    if len(texts) < 2:
        return []
    matrix = TfidfVectorizer(min_df=1).fit_transform(texts)
    sims = cosine_similarity(matrix)
    pairs = []
    for i, j in combinations(range(len(texts)), 2):
        if sims[i, j] >= SIMILARITY_THRESHOLD:
            pairs.append((sims[i, j], ids[i], ids[j]))
    pairs.sort(reverse=True)
    return [("Two selected tweets have high pairwise textual similarity.", {
                "kind": "similarity", "source_ids": [a, b], "raw_similarity": float(score),
             }) for score, a, b in pairs[:MAX_SIMILARITY_OBS]]


def _obs_units(record: AccountRecord, tweet_units: list[Unit]) -> list[Unit]:
    facts = (_ratio_obs(record.profile) + _burst_obs(record.posts) + _similarity_obs(tweet_units)
             + _graph_ratio_obs(record.graph_edges))
    return [Unit(f"OBS_{i + 1:02d}", text, hidden) for i, (text, hidden) in enumerate(facts)]


def build_units(record: AccountRecord) -> list[Unit]:
    tweet_units = _tweet_units(record.posts)
    obs_units = _obs_units(record, tweet_units)
    units = (
        _profile_units(record)
        + tweet_units
        + _activity_units(record.posts)
        + _relation_units(record.graph_edges)
        + _fact_units(record.graph_facts)
        + _graph_units(record.hgt_neighbor_facts)
        + obs_units
    )
    # The banned-word check only applies to units this program generates itself (OBS_xx): it
    # guards against the program injecting a judgmental framing into a "fact". It must NOT apply
    # to verbatim account content (tweets, profile fields, ...), which can legitimately contain
    # any of these words as part of what the account actually wrote (e.g. a tweet about "pump
    # fakes" in football, or a bio that says "verified" or "not spam").
    for unit in obs_units:
        lowered = unit.text.lower()
        if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in BANNED_WORDS):
            raise ValueError(f"unit {unit.unit_id} contains a judgmental word, must stay factual: {unit.text!r}")
    return units


def hidden_unit_metadata(units: list[Unit]) -> dict[str, dict]:
    """Program-only sidecar used by format/reward checks; never rendered into a prompt."""
    return {unit.unit_id: unit.hidden for unit in units if unit.hidden}


def render_units(units: list[Unit]) -> str:
    return "\n".join(f"[{u.unit_id}] {u.text}" for u in units)


def unit_map(units: list[Unit]) -> dict[str, str]:
    return {u.unit_id: u.text for u in units}


def render_masked(units: list[Unit], excluded_ids: set[str]) -> str:
    kept = [u for u in units if u.unit_id not in excluded_ids]
    return render_units(kept)
