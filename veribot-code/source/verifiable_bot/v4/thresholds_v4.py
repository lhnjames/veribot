"""Versioned quantile bucket thresholds — fitted on the official TRAIN split and nothing else.

Why this file exists: the model prompt never shows a raw number (the RF ablation supports this —
5-tier bucketing scored 83.94% vs 83.01% for raw values, i.e. bucketing costs nothing), so every
continuous feature is turned into one of five ordinal tiers. Those boundaries are a *learned
statistic of the training distribution*, which makes them exactly as leak-prone as model weights.

Two guards enforce train-only fitting, and both are tested:

* :func:`fit_thresholds` refuses any source file not named ``train.json``;
* :func:`load_thresholds` refuses any manifest whose recorded ``source_file`` is not a
  ``train.json``.

So dev/test can only ever be *scored* against frozen train-fitted cuts; no code path inspects a
dev/test value while deciding a boundary.

Fixed five-slot scheme (pre-registered, do not "optimise")
----------------------------------------------------------
Every continuous feature always stores **exactly four cut points — p10, p30, p70, p90 — and always
maps onto exactly five slots**. An earlier draft de-duplicated equal cut points and collapsed the
scheme to fewer tiers; that was wrong on two counts: it silently changed the tier count per feature
(so the same label meant different things for different features), and it re-mapped the surviving
cuts onto the wrong quantiles, corrupting the percentile bracket reported in the audit. Ties are
expected; some slots simply become unreachable for some features, which is harmless. A silently
re-defined slot is not.

Interval convention, and why ties need a per-value rule
-------------------------------------------------------
The base rule is ``np.digitize(value, cuts, right=True)``: slot *i* is ``(cuts[i-1], cuts[i]]``,
slot 0 is ``(-inf, cuts[0]]``, slot 4 is ``(cuts[3], +inf)`` — i.e. a value sitting exactly on a
cut falls into the lower slot.

That single rule is not sufficient once a cut value is *duplicated*, and picking either direction
globally produces a rendered falsehood on real TwiBot-20 data:

* ``digits_in_sn`` has cuts ``[0, 0, 0, 2]`` (most handles contain no digit). A handle with no
  digits must not be described as sitting high in the distribution — ties must fall **low** here.
* ``uniq_tweet_ratio`` has cuts ``[0.99, 1, 1, 1]`` (most timelines are fully distinct). A timeline
  whose posts are *all distinct* (ratio exactly 1.0) must not be described as sitting low — ties
  must fall **high** here.

The direction is therefore a property of where the tied mass sits, not a global preference. At fit
time each duplicated cut value gets its true percentile span in the fitting population, and the
slot whose pre-registered percentile bracket contains the **midpoint** of that span is stored in
``tie_map``. This never changes the number of slots, never re-maps a quantile, and makes the tier
word a true statement about the account's position. Non-duplicated cuts are untouched and keep the
plain ``right=True`` behaviour (which the midpoint rule agrees with in the limit anyway).

Label vocabulary
----------------
Slots always number five, and their wording is customisable per feature via
:data:`LABEL_OVERRIDES` — but an override is **rejected at fit time for any feature with duplicated
cuts**. Absolute-sounding wording ("none", "all distinct") cannot be derived from a purely
positional slot once a slot spans a large tied mass; those features keep the positional vocabulary,
which the render templates pair with "compared with other accounts in this dataset" so the rendered
sentence stays true. The manifest always records the raw four quantile cut points regardless of
wording.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .features_v4 import (
    BINARY_INDICES,
    CONTINUOUS_INDICES,
    FEATURE_NAMES,
    N_FEATURES,
    TWEET_STAT_INDICES,
    extract_features,
    neighbor_counts,
    tweet_list,
)

THRESHOLDS_VERSION = "bucket_thresholds_v4"

# Pre-registered and fixed. Four cut points, five slots, for every continuous feature.
QUANTILES: tuple[float, ...] = (10.0, 30.0, 70.0, 90.0)
N_SLOTS = len(QUANTILES) + 1
# Slot i covers percentile bracket (QUANTILE_EDGES[i], QUANTILE_EDGES[i+1]].
QUANTILE_EDGES: tuple[float, ...] = (0.0,) + QUANTILES + (100.0,)

DEFAULT_LABELS: tuple[str, ...] = ("very low", "low", "moderate", "high", "very high")

# Per-feature wording, five entries each, digit-free, ordered low -> high. Only applied to features
# whose four cuts are all distinct (see module docstring); rejected at fit time otherwise.
LABEL_OVERRIDES: dict[str, tuple[str, ...]] = {
    "std_tweet_len": ("highly uniform", "fairly uniform", "moderate", "varied", "highly varied"),
    "retweet_ratio": ("almost none", "low", "moderate", "high", "almost all"),
}

ALL_TIER_LABELS: frozenset[str] = frozenset(DEFAULT_LABELS).union(
    label for labels in LABEL_OVERRIDES.values() for label in labels
)


def labels_for(feature_name: str, cuts: list[float] | None = None) -> tuple[str, ...]:
    """Label vocabulary for a feature. Overrides are dropped when the cuts contain duplicates."""
    labels = LABEL_OVERRIDES.get(feature_name, DEFAULT_LABELS)
    if cuts is not None and len(set(cuts)) < len(QUANTILES):
        labels = DEFAULT_LABELS
    if len(labels) != N_SLOTS:
        raise ValueError(f"label vocabulary for {feature_name!r} must have {N_SLOTS} entries")
    return labels


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cut_points(values: np.ndarray) -> list[float]:
    """The four raw quantile cut points, ties preserved (see 'Fixed five-slot scheme' above)."""
    return [float(v) for v in np.percentile(values, QUANTILES)]


def _slot_for_percentile(percentile: float) -> int:
    """Pre-registered bracket containing this percentile: slot i = (edges[i], edges[i+1]]."""
    for slot in range(N_SLOTS):
        if percentile <= QUANTILE_EDGES[slot + 1]:
            return slot
    return N_SLOTS - 1


def _tie_map(column: np.ndarray, cuts: list[float]) -> list[dict[str, float]]:
    """For each duplicated cut value: its true percentile span and the slot holding the midpoint."""
    entries: list[dict[str, float]] = []
    n = column.size
    for value in sorted({c for c in cuts if cuts.count(c) > 1}):
        below = 100.0 * float((column < value).sum()) / n
        at_or_below = 100.0 * float((column <= value).sum()) / n
        midpoint = 0.5 * (below + at_or_below)
        entries.append({
            "value": float(value),
            "slot": _slot_for_percentile(midpoint),
            "pct_below": below,
            "pct_at_or_below": at_or_below,
            "pct_midpoint": midpoint,
        })
    return entries


def _fit_column(column: np.ndarray, *, name: str, index: int | None, fitted_on: str,
                labels: tuple[str, ...]) -> dict[str, Any]:
    """One feature's frozen spec: four raw cuts, five slots, tie_map, diagnostics."""
    cuts = _cut_points(column)
    return {
        "index": index,
        "cuts": cuts,                       # always exactly four, ties preserved
        "quantiles": list(QUANTILES),
        "n_slots": N_SLOTS,
        "labels": list(labels),
        "n_distinct_cuts": len(set(cuts)),  # diagnostic only; never changes the slot count
        "tie_map": _tie_map(column, cuts),
        "fitted_on": fitted_on,
        "n_fitted": int(column.size),
    }


def fit_thresholds(train_path: str | Path) -> dict[str, Any]:
    """Fit the frozen five-slot cuts on the official train split. Raises on any other split."""
    path = Path(train_path)
    if path.name != "train.json":
        raise ValueError(
            f"thresholds may only be fitted on the official train split, got {path.name!r}. "
            "Fitting on dev/test would leak the evaluation distribution into the prompt encoding."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray([extract_features(r) for r in records], dtype=np.float64)
    with_tweets = np.asarray([bool(tweet_list(r)) for r in records])

    features: dict[str, Any] = {}
    for index in CONTINUOUS_INDICES:
        name = FEATURE_NAMES[index]
        if index in TWEET_STAT_INDICES:
            # Undefined for accounts with no timeline; fitting over their zeros would drag every
            # boundary toward 0 and mislabel genuinely low-activity accounts.
            column = matrix[with_tweets, index]
            fitted_on = "train accounts with at least one tweet"
        else:
            column = matrix[:, index]
            fitted_on = "all train accounts"
        features[name] = _fit_column(
            column, name=name, index=index, fitted_on=fitted_on, labels=labels_for(name, _cut_points(column))
        )

    # RELATION_xx counts are fitted only over train accounts that actually have neighbours (the
    # ~91% with a non-empty neighbor field), for the same reason tweet stats skip empty timelines:
    # an absent count is undefined, not zero. These are NOT in the 33-dim E1 vector.
    neighbours = [nc for nc in (neighbor_counts(r) for r in records) if nc is not None]
    following_col = np.asarray([f for f, _ in neighbours], dtype=np.float64)
    follower_col = np.asarray([g for _, g in neighbours], dtype=np.float64)
    relation = {
        "neighbor_following_count": _fit_column(
            following_col, name="neighbor_following_count", index=None,
            fitted_on="train accounts with a non-empty neighbor field", labels=DEFAULT_LABELS,
        ),
        "neighbor_follower_count": _fit_column(
            follower_col, name="neighbor_follower_count", index=None,
            fitted_on="train accounts with a non-empty neighbor field", labels=DEFAULT_LABELS,
        ),
    }

    return {
        "relation": relation,
        "version": THRESHOLDS_VERSION,
        "quantiles": list(QUANTILES),
        "n_slots": N_SLOTS,
        "interval_convention": (
            "np.digitize(right=True): slot i = (cuts[i-1], cuts[i]], ties fall low; "
            "duplicated cut values instead use tie_map (slot holding the midpoint of the tied "
            "percentile span)"
        ),
        "source_file": str(path),
        "source_sha256": sha256_of(path),
        "n_accounts": len(records),
        "n_accounts_with_tweets": int(with_tweets.sum()),
        "n_features": N_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "binary_indices": sorted(BINARY_INDICES),
        "tweet_stat_indices": list(TWEET_STAT_INDICES),
        "features": features,
    }


def save_thresholds(thresholds: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic: never leave a half-written thresholds file behind
    return path


def load_thresholds(path: str | Path) -> dict[str, Any]:
    """Read frozen thresholds, rejecting anything not fitted on the official train split."""
    thresholds = json.loads(Path(path).read_text(encoding="utf-8"))
    if thresholds.get("version") != THRESHOLDS_VERSION:
        raise ValueError(f"unexpected thresholds version: {thresholds.get('version')!r}")
    source = str(thresholds.get("source_file", ""))
    if not source.endswith("train.json"):
        raise ValueError(
            f"thresholds were fitted on {source!r}, not on an official train.json; refusing to use "
            "them because dev/test statistics must never reach the prompt encoding."
        )
    if list(thresholds.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError("thresholds feature_names disagree with features_v4.FEATURE_NAMES")
    for name, spec in thresholds["features"].items():
        if len(spec["cuts"]) != len(QUANTILES) or len(spec["labels"]) != N_SLOTS:
            raise ValueError(
                f"feature {name!r} must keep exactly {len(QUANTILES)} cuts and {N_SLOTS} labels"
            )
    relation = thresholds.get("relation")
    if not relation or set(relation) != {"neighbor_following_count", "neighbor_follower_count"}:
        raise ValueError("thresholds must carry both relation count specs")
    for name, spec in relation.items():
        if len(spec["cuts"]) != len(QUANTILES) or len(spec["labels"]) != N_SLOTS:
            raise ValueError(f"relation spec {name!r} must keep {len(QUANTILES)} cuts and {N_SLOTS} labels")
    return thresholds


def slot_of(value: float, spec: dict[str, Any]) -> int:
    """Slot index in ``0..4``.

    Duplicated cut values are resolved through ``tie_map`` (percentile-midpoint rule); every other
    value uses ``right=True``, i.e. slot i = (cuts[i-1], cuts[i]] with ties falling low.
    """
    value = float(value)
    for entry in spec.get("tie_map", ()):
        if value == entry["value"]:
            return int(entry["slot"])
    return int(np.digitize([value], spec["cuts"], right=True)[0])


def tier_of(value: float, spec: dict[str, Any]) -> str:
    return spec["labels"][slot_of(value, spec)]


def tier_for_feature(name: str, value: float, thresholds: dict[str, Any]) -> str:
    return tier_of(value, thresholds["features"][name])


def percentile_bracket(value: float, spec: dict[str, Any]) -> str:
    """Percentile bracket of the slot, for the audit report only. Never rendered into a prompt."""
    slot = slot_of(value, spec)
    return f"p{QUANTILE_EDGES[slot]:g}-p{QUANTILE_EDGES[slot + 1]:g}"
