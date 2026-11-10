"""v4 prompt rendering: every discriminative signal becomes a citable unit under a FIXED public
unit-ID contract that the evidence bank and the reward gate both rely on.

Why this module replaces the v3 renderer (verifiable_bot/units.py):

v3 exposed 7 profile fields and no timeline statistics at all. The RF ablation quantifies the cost:
those 7 fields reach 80.39% on TwiBot-20 test, the full 33 reach 83.94% (AUC 84.64% vs 90.13%).
Roughly a fifth of the available discriminative power was never rendered into the prompt, and the
tweet statistics v3 omitted entirely (retweet share, links/hashtags/mentions per post, post length
mean and variability, distinct-post ratio) account for most of the gap. A model cannot cite
evidence it was never shown.

Public unit-ID contract (do NOT renumber — the evidence bank and reward keyed to these ids)
-------------------------------------------------------------------------------------------
    PROFILE_01 verified          PROFILE_02 follower count     PROFILE_03 following count
    PROFILE_04 total posts       PROFILE_05 times listed       PROFILE_06 favourited count
    PROFILE_07 account age        PROFILE_08 description text    PROFILE_09 profile completeness
    OBS_01 friends/followers     OBS_02 followers/friends      OBS_03 posting rate (+following rate)
    OBS_04 follower-growth rate  OBS_05 favouriting rate       OBS_06 handle digit density
    OBS_07 description length    OBS_08 name & handle length
    ACTIVITY_01 retweet share    ACTIVITY_02 links/post        ACTIVITY_03 hashtags/post
    ACTIVITY_04 mentions/post    ACTIVITY_05 mean post length  ACTIVITY_06 post-length variability
    ACTIVITY_07 distinct-post ratio
    RELATION_01 neighbour following & follower count tiers
    TWEET_01.. verbatim sampled posts

``n_tweets`` (feature index 25) stays in the E1 numeric vector but is deliberately NOT rendered:
it is mostly the 200-post collection cap, not a real activity level, so a "timeline sample size:
high" unit would mislead. ``friends/age`` (index 10) has no OBS id of its own; it rides along as a
second clause of OBS_03. This keeps the public id count at OBS_01..08 / ACTIVITY_01..07.

Each unit carries a ``clauses`` sidecar: one entry per bucketed sub-statement, with the feature
index, tier and raw value, so a program (audit / reward) can verify every tier the model repeats
without another model in the loop. The contract test pins ``clauses[0].feature_index`` for every
bucketed id — in particular ACTIVITY_07 -> uniq_tweet_ratio(29), OBS_04 -> followers/age(9),
OBS_05 -> fav/age(8) — so a silent renumber fails the suite instead of corrupting the reward.

Design rules, all enforced by tests:

* **No Arabic digits reach the model** (bucketing is free: 83.94% bucketed vs 83.01% raw, and
  scrubbing digits from verbatim text costs nothing on the TF-IDF channel: 72.53% -> 72.70%).
* **Ratios are pre-computed, not left to the model** — LLMs are unreliable at arithmetic.
* **Exact facts get absolute wording** — a handle with no digits reads "contains no digits", a
  fully-distinct timeline reads "all sampled posts are distinct", rather than a positional tier,
  because those are true statements regardless of the distribution. The five-slot tier is still
  recorded in the clause sidecar; only the prose changes.
* **Missing data degrades, never crashes** — no timeline -> zero ACTIVITY units; ``neighbor`` None
  -> zero RELATION units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .features_v4 import FEATURE_NAMES, N_FEATURES, extract_features, neighbor_counts, tweet_list
from .thresholds_v4 import slot_of, tier_of

MAX_TWEET_UNITS = 20
MAX_TWEET_CHARS = 280

_DIGIT_RUN = re.compile(r"\d+(?:[.,:/\-]\d+)*")


def digit_free(text: str) -> str:
    """Replace every run of numeric glyphs with the word 'number'.

    ``re.\\d`` only matches Unicode *decimal* digits (category Nd: ASCII 0-9, Arabic-Indic, full
    width, Devanagari, …). It misses superscripts/subscripts (¹²⁶₂), circled/parenthesised digits
    (①②③❶➊) and similar — all of which ``str.isdigit()`` DOES flag and all of which are still
    numeric information the model must not see. So we first fold every ``ch.isdigit()`` character to
    ASCII ``0`` and then let ``_DIGIT_RUN`` collapse runs, which preserves the existing
    date/separator-merging semantics (e.g. "12/34" -> "number"). We deliberately use ``isdigit()``,
    NOT ``isnumeric()``: the latter would also strip Chinese numeral WORDS (一, 二, 十) which are
    legitimate verbatim content, not digits.
    """
    folded = "".join("0" if ch.isdigit() else ch for ch in str(text))
    return _DIGIT_RUN.sub("number", folded)


@dataclass
class Unit:
    unit_id: str
    text: str
    # feature indices this unit draws on (for prompt-coverage accounting); relation counts are None.
    feature_indices: list[int] = field(default_factory=list)
    # bucketed sub-statements: {feature_index|None, name, tier, raw}. Empty for text/flag units.
    clauses: list[dict[str, Any]] = field(default_factory=list)
    hidden: dict[str, Any] = field(default_factory=dict)


def _clause(index: int, values: list[float], thresholds: dict) -> dict[str, Any]:
    name = FEATURE_NAMES[index]
    spec = thresholds["features"][name]
    value = values[index]
    return {"feature_index": index, "name": name, "tier": tier_of(value, spec), "raw": value}


# --------------------------------------------------------------------------------------------
# PROFILE_01..09
# --------------------------------------------------------------------------------------------
def _profile_units(record: dict, values: list[float], thresholds: dict) -> list[Unit]:
    profile = record.get("profile") or {}
    units: list[Unit] = []

    verified = values[18] >= 0.5
    units.append(Unit(
        "PROFILE_01",
        digit_free(f"Verification status: {'verified' if verified else 'not verified'}."),
        feature_indices=[18],
    ))

    for index, unit_id, label in (
        (0, "PROFILE_02", "Follower count"),
        (1, "PROFILE_03", "Following count"),
        (2, "PROFILE_04", "Total posts published"),
        (4, "PROFILE_05", "Times listed by others"),
        (3, "PROFILE_06", "Posts this account has favourited"),
        (11, "PROFILE_07", "Account age"),
    ):
        clause = _clause(index, values, thresholds)
        units.append(Unit(
            unit_id,
            digit_free(f"{label}: {clause['tier']} relative to other accounts in this dataset."),
            feature_indices=[index], clauses=[clause],
        ))

    description = str(profile.get("description") or "").strip()
    units.append(Unit(
        "PROFILE_08",
        digit_free(f"Profile description: {description}" if description
                   else "Profile description: none provided."),
        feature_indices=[16, 17], hidden={"raw_description": description},
    ))

    flags = [
        ("a profile description", values[17] >= 0.5),
        ("a profile link", values[23] >= 0.5),
        ("a stated location", values[24] >= 0.5),
        ("geotagging enabled", values[21] >= 0.5),
    ]
    present = [label for label, ok in flags if ok]
    absent = [label for label, ok in flags if not ok]
    parts = [
        "The profile has " + ", ".join(present) + "." if present
        else "The profile has none of: a description, a link, a stated location, geotagging."
    ]
    if absent and present:
        parts.append("It lacks " + ", ".join(absent) + ".")
    if values[19] >= 0.5:
        parts.append("The profile theme is left at the platform default.")
    if values[20] >= 0.5:
        parts.append("The avatar is left at the platform default.")
    if values[22] >= 0.5:
        parts.append("The account is protected.")
    units.append(Unit(
        "PROFILE_09", digit_free(" ".join(parts)),
        feature_indices=[17, 19, 20, 21, 22, 23, 24],
    ))
    return units


# --------------------------------------------------------------------------------------------
# OBS_01..08  (derived ratios; model never does arithmetic)
# --------------------------------------------------------------------------------------------
def _obs_units(values: list[float], thresholds: dict) -> list[Unit]:
    units: list[Unit] = []

    def simple(unit_id: str, index: int, template: str) -> None:
        clause = _clause(index, values, thresholds)
        units.append(Unit(unit_id, digit_free(template.format(tier=clause["tier"])),
                          feature_indices=[index], clauses=[clause]))

    simple("OBS_01", 5, "Following-to-follower ratio: {tier} relative to other accounts.")
    simple("OBS_02", 6, "Follower-to-following ratio: {tier} relative to other accounts.")

    # OBS_03 carries statuses/age as its headline plus friends/age as a second clause (index 10 has
    # no id of its own; see module docstring).
    c_status = _clause(7, values, thresholds)
    c_friends_age = _clause(10, values, thresholds)
    units.append(Unit(
        "OBS_03",
        digit_free(f"Posting rate since account creation: {c_status['tier']} relative to other "
                   f"accounts; following-accumulation rate: {c_friends_age['tier']}."),
        feature_indices=[7, 10], clauses=[c_status, c_friends_age],
    ))

    simple("OBS_04", 9, "Follower-accumulation rate since creation: {tier} relative to other accounts.")
    simple("OBS_05", 8, "Favouriting rate since account creation: {tier} relative to other accounts.")

    # OBS_06 handle digit density: absolute wording when there is genuinely no digit, else the
    # positional tier plus the count as a second clause.
    c_ratio = _clause(15, values, thresholds)
    c_count = _clause(14, values, thresholds)
    if values[14] == 0:
        text = "The account handle contains no digits."
    else:
        text = (f"Share of digits in the account handle: {c_ratio['tier']} relative to other "
                f"accounts.")
    units.append(Unit("OBS_06", digit_free(text), feature_indices=[15, 14],
                      clauses=[c_ratio, c_count]))

    simple("OBS_07", 16, "Profile description length: {tier} relative to other accounts.")

    # OBS_08 renders BOTH the display-name length and the handle length as their own conclusions.
    c_name = _clause(12, values, thresholds)
    c_handle = _clause(13, values, thresholds)
    units.append(Unit(
        "OBS_08",
        digit_free(f"Display-name length: {c_name['tier']} relative to other accounts; "
                   f"handle length: {c_handle['tier']}."),
        feature_indices=[12, 13], clauses=[c_name, c_handle],
    ))
    return units


# --------------------------------------------------------------------------------------------
# ACTIVITY_01..07  (timeline stats; [] when no timeline)
# --------------------------------------------------------------------------------------------
def _activity_units(record: dict, values: list[float], thresholds: dict) -> list[Unit]:
    if not tweet_list(record):
        return []
    units: list[Unit] = []

    def simple(unit_id: str, index: int, template: str) -> None:
        clause = _clause(index, values, thresholds)
        units.append(Unit(unit_id, digit_free(template.format(tier=clause["tier"])),
                          feature_indices=[index], clauses=[clause]))

    # ACTIVITY_01 retweet share: absolute wording at the exact-zero fact.
    c_rt = _clause(26, values, thresholds)
    rt_text = ("None of the sampled posts are retweets." if values[26] == 0
               else f"Share of the timeline that is retweets: {c_rt['tier']} relative to other accounts.")
    units.append(Unit("ACTIVITY_01", digit_free(rt_text), feature_indices=[26], clauses=[c_rt]))

    simple("ACTIVITY_02", 30, "Links per post: {tier} relative to other accounts.")
    simple("ACTIVITY_03", 31, "Hashtags per post: {tier} relative to other accounts.")
    simple("ACTIVITY_04", 32, "Mentions per post: {tier} relative to other accounts.")
    simple("ACTIVITY_05", 27, "Average post length: {tier} relative to other accounts.")
    simple("ACTIVITY_06", 28, "Variability of post length: {tier} relative to other accounts.")

    # ACTIVITY_07 distinct-post ratio: absolute wording when the timeline is fully distinct.
    c_uniq = _clause(29, values, thresholds)
    uniq_text = ("All sampled posts are distinct from one another." if values[29] >= 1.0
                 else f"Share of posts distinct from one another: {c_uniq['tier']} relative to other accounts.")
    units.append(Unit("ACTIVITY_07", digit_free(uniq_text), feature_indices=[29], clauses=[c_uniq]))
    return units


# --------------------------------------------------------------------------------------------
# RELATION_01  (neighbour count tiers; [] when neighbor missing)
# --------------------------------------------------------------------------------------------
def _relation_units(record: dict, thresholds: dict) -> list[Unit]:
    counts = neighbor_counts(record)
    if counts is None:
        return []
    following, follower = counts
    rel = thresholds["relation"]
    f_tier = tier_of(float(following), rel["neighbor_following_count"])
    g_tier = tier_of(float(follower), rel["neighbor_follower_count"])
    text = (f"Among the sampled social neighbours, the number this account follows is {f_tier} "
            f"relative to other accounts, and the number following this account is {g_tier}.")
    return [Unit(
        "RELATION_01", digit_free(text), feature_indices=[],
        clauses=[
            {"feature_index": None, "name": "neighbor_following_count", "tier": f_tier, "raw": following},
            {"feature_index": None, "name": "neighbor_follower_count", "tier": g_tier, "raw": follower},
        ],
    )]


# --------------------------------------------------------------------------------------------
# TWEET_xx  (verbatim sampled posts)
# --------------------------------------------------------------------------------------------
def _select_tweets(tweets: list, limit: int) -> list[str]:
    """Evenly spaced, deterministic sample across the whole timeline (not the first N, which would
    show only one slice of the account's history). ``limit <= 0`` yields no tweets — the adaptive
    prompt fitter drives ``limit`` down to zero for pathologically long accounts."""
    texts = [re.sub(r"\s+", " ", str(t)).strip() for t in tweets if isinstance(t, str)]
    texts = [t for t in texts if t]
    if limit <= 0:
        return []
    if len(texts) <= limit:
        return texts
    step = len(texts) / limit
    return [texts[min(int(i * step), len(texts) - 1)] for i in range(limit)]


def _tweet_units(record: dict, limit: int) -> list[Unit]:
    units = []
    for i, text in enumerate(_select_tweets(tweet_list(record), limit)):
        raw = text[:MAX_TWEET_CHARS]
        units.append(Unit(f"TWEET_{i + 1:02d}", digit_free(raw), hidden={"raw_text": raw}))
    return units


def build_units_v4(record: dict, thresholds: dict, max_tweets: int = MAX_TWEET_UNITS) -> list[Unit]:
    values = extract_features(record)
    units = (
        _profile_units(record, values, thresholds)
        + _obs_units(values, thresholds)
        + _activity_units(record, values, thresholds)
        + _relation_units(record, thresholds)
        + _tweet_units(record, max_tweets)
    )
    for unit in units:
        # isdigit(), not just \d: also catches superscripts/circled digits digit_free folds away.
        bad = [ch for ch in unit.text if ch.isdigit()]
        if bad:
            raise ValueError(f"unit {unit.unit_id} still contains numeric glyphs {bad}: {unit.text!r}")
    return units


def render_units(units: list[Unit]) -> str:
    return "\n".join(f"[{u.unit_id}] {u.text}" for u in units)


def unit_map(units: list[Unit]) -> dict[str, str]:
    return {u.unit_id: u.text for u in units}


def hidden_unit_metadata(units: list[Unit]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for unit in units:
        meta = dict(unit.hidden)
        if unit.clauses:
            meta["clauses"] = unit.clauses
        if meta:
            out[unit.unit_id] = meta
    return out


def covered_feature_indices(units: list[Unit]) -> set[int]:
    """Feature indices reachable by the model through the prompt (relation counts excluded)."""
    covered: set[int] = set()
    for unit in units:
        covered.update(unit.feature_indices)
    return covered


def unit_metadata(units: list[Unit]) -> list[dict[str, Any]]:
    """Programmatically-verifiable sidecar, one entry per unit, aligned 1:1 with ``units``.

    Carries the bucketed clauses (feature index, name, tier, raw value) that the audit and the
    reward gate check against. It is NEVER part of the canonical model prompt (prompt_v4 builds the
    prompt from unit *text* only) and holds no ground-truth beyond the account's own derived values
    — no evidence-bank labels.
    """
    return [
        {
            "unit_id": u.unit_id,
            "feature_indices": list(u.feature_indices),
            "clauses": [dict(c) for c in u.clauses],
        }
        for u in units
    ]


def record_payload(
    record: dict, units: list[Unit], *, prompt_token_count: int,
    candidate_max_tweets: int, effective_tweet_cap: int,
) -> dict:
    """One rendered account for a jsonl shard. ``units`` are already the adaptively-fitted set from
    prompt_v4.fit_account_prompt; this function only serialises, it does not choose the tweet count.
    The canonical prompt string is intentionally NOT stored — E1/E2 rebuild it through prompt_v4 so
    there is exactly one wrapper — but its token count is, so overflow can be asserted downstream.

    Three distinct tweet fields, deliberately not conflated:
      * ``candidate_max_tweets``  — the fixed starting cap (always 30);
      * ``effective_tweet_cap``   — the cap that actually fit the budget (< candidate iff the prompt
                                    was shortened FOR THE BUDGET);
      * ``n_tweet_units``         — how many TWEET units were really rendered (<= effective cap; can
                                    be lower simply because the account has few tweets).
    """
    n_tweets = sum(1 for u in units if u.unit_id.startswith("TWEET_"))
    return {
        "account_id": str(record.get("ID", "")).strip(),
        "label": ("bot" if str(record.get("label")).strip() == "1" else "human")
        if record.get("label") is not None else None,
        "features": extract_features(record),
        "feature_names": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
        "units": [{"unit_id": u.unit_id, "text": u.text} for u in units],
        "unit_metadata": unit_metadata(units),
        "candidate_max_tweets": candidate_max_tweets,
        "effective_tweet_cap": effective_tweet_cap,
        "n_tweet_units": n_tweets,
        "prompt_token_count": prompt_token_count,
    }
