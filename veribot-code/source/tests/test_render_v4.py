"""S1 gate, second check: the v4 rendering / feature / threshold invariants.

Every test here pins a property that, if it silently broke, would invalidate downstream numbers
rather than crash: a reordered feature vector, a threshold quietly fitted on dev, a digit leaking
into the prompt, an ACTIVITY unit asserting "very low" about an account that has no timeline at
all. None of those would raise on their own.
"""
from __future__ import annotations

import ast
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from verifiable_bot.v4 import features_v4, render_v4, thresholds_v4
from verifiable_bot.v4.features_v4 import FEATURE_NAMES, N_FEATURES, extract_features
from verifiable_bot.v4.render_v4 import build_units_v4, digit_free, covered_feature_indices
from verifiable_bot.v4.thresholds_v4 import (
    DEFAULT_LABELS,
    N_SLOTS,
    QUANTILES,
    fit_thresholds,
    load_thresholds,
    slot_of,
    tier_of,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "twibot20_complete"
THRESHOLDS_PATH = REPO / "outputs" / "v4" / "render_v4" / "bucket_thresholds_v4.json"
SAMPLE = 120  # accounts per split for the value/rendering sweeps; keeps the suite under a minute


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def thresholds() -> dict:
    if not THRESHOLDS_PATH.exists():
        pytest.fail(
            f"missing {THRESHOLDS_PATH}. Run:\n"
            "  python scripts/compute_bucket_thresholds_v4.py"
        )
    return load_thresholds(THRESHOLDS_PATH)


def _load(split: str) -> list[dict]:
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def splits() -> dict[str, list[dict]]:
    return {name: _load(name) for name in ("train", "dev", "test")}


@pytest.fixture(scope="session")
def sampled(splits) -> dict[str, list[dict]]:
    """Deterministic stride sample: covers the whole file without loading a bias toward the head."""
    out = {}
    for name, records in splits.items():
        step = max(1, len(records) // SAMPLE)
        out[name] = records[::step][:SAMPLE]
    return out


# --------------------------------------------------------------------------------------------
# 1. the 33-dim vector must stay identical to the RF baseline
# --------------------------------------------------------------------------------------------
def test_feature_order_matches_rf_baseline():
    """FEATURE_NAMES is parsed straight out of scripts/rf_baseline.py, so a reorder on either side
    fails here rather than silently invalidating the 83.01% / 85.04% baselines."""
    source = (REPO / "scripts" / "rf_baseline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "NAMES" for t in node.targets
        ):
            names = ast.literal_eval(node.value)
    assert names is not None, "scripts/rf_baseline.py no longer defines NAMES"
    assert list(FEATURE_NAMES) == list(names)
    assert N_FEATURES == 33


def _rf_reference_feats(r: dict) -> list[float]:
    """Verbatim copy of scripts/rf_baseline.py::feats — the golden implementation."""
    def s(v):
        return (v or "").strip() if isinstance(v, str) else v

    def num(v):
        try:
            return float(str(v).strip())
        except Exception:
            return 0.0

    def boo(v):
        return 1.0 if str(v).strip() == "True" else 0.0

    p = r["profile"]
    fo, fr = num(p.get("followers_count")), num(p.get("friends_count"))
    st, fa, li = num(p.get("statuses_count")), num(p.get("favourites_count")), num(p.get("listed_count"))
    try:
        age = (datetime(2020, 9, 1) - datetime.strptime(
            s(p.get("created_at")), "%a %b %d %H:%M:%S %z %Y").replace(tzinfo=None)).days
    except Exception:
        age = 0
    age = max(age, 1)
    name, sn, des = s(p.get("name")) or "", s(p.get("screen_name")) or "", s(p.get("description")) or ""
    dig_sn = sum(c.isdigit() for c in sn)
    tw = r.get("tweet") or []
    ntw = len(tw)
    rt = sum(1 for t in tw if isinstance(t, str) and t.startswith("RT @"))
    lens = [len(t) for t in tw if isinstance(t, str)]
    uniq = len(set(tw)) / max(ntw, 1)
    urls = sum(1 for t in tw if isinstance(t, str) and "http" in t)
    hsh = sum(t.count("#") for t in tw if isinstance(t, str))
    men = sum(t.count("@") for t in tw if isinstance(t, str))
    return [
        math.log1p(fo), math.log1p(fr), math.log1p(st), math.log1p(fa), math.log1p(li),
        fr / max(fo, 1), fo / max(fr, 1), st / age, fa / age, fo / age, fr / age,
        math.log1p(age), len(name), len(sn), dig_sn, dig_sn / max(len(sn), 1), len(des),
        1.0 if des else 0.0, boo(p.get("verified")), boo(p.get("default_profile")),
        boo(p.get("default_profile_image")), boo(p.get("geo_enabled")), boo(p.get("protected")),
        1.0 if s(p.get("url")) else 0.0, 1.0 if s(p.get("location")) else 0.0,
        ntw, rt / max(ntw, 1), np.mean(lens) if lens else 0, np.std(lens) if lens else 0,
        uniq, urls / max(ntw, 1), hsh / max(ntw, 1), men / max(ntw, 1),
    ]


def test_feature_values_match_rf_baseline(sampled):
    for split, records in sampled.items():
        for record in records:
            got = extract_features(record)
            want = np.nan_to_num(np.asarray(_rf_reference_feats(record), dtype=np.float64),
                                 posinf=0, neginf=0)
            assert len(got) == N_FEATURES
            np.testing.assert_allclose(
                got, want, rtol=0, atol=0,
                err_msg=f"{split}/{record.get('ID')} diverges from the RF baseline vector",
            )


# --------------------------------------------------------------------------------------------
# 2. thresholds: train-only, five fixed slots
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("split", ["dev", "test"])
def test_fit_refuses_non_train_split(split):
    with pytest.raises(ValueError, match="official train split"):
        fit_thresholds(DATA / f"{split}.json")


def test_load_refuses_thresholds_not_fitted_on_train(tmp_path, thresholds):
    poisoned = dict(thresholds, source_file="data/twibot20_complete/dev.json")
    path = tmp_path / "poisoned.json"
    path.write_text(json.dumps(poisoned), encoding="utf-8")
    with pytest.raises(ValueError, match="not on an official train.json"):
        load_thresholds(path)


def test_thresholds_manifest_shape(thresholds):
    assert thresholds["source_file"].endswith("train.json")
    assert thresholds["quantiles"] == [10.0, 30.0, 70.0, 90.0]
    assert thresholds["n_slots"] == N_SLOTS == 5
    for name, spec in thresholds["features"].items():
        assert len(spec["cuts"]) == len(QUANTILES) == 4, f"{name} must keep the raw four cuts"
        assert len(spec["labels"]) == 5, f"{name} must keep five slots"
        assert spec["cuts"] == sorted(spec["cuts"]), f"{name} cuts must be non-decreasing"
        assert spec["quantiles"] == [10.0, 30.0, 70.0, 90.0]
    # every continuous feature, and only those, is bucketed
    assert set(thresholds["features"]) == {FEATURE_NAMES[i] for i in features_v4.CONTINUOUS_INDICES}


def test_bucket_boundary_semantics_distinct_cuts():
    """Base rule, no ties: slot i = (cuts[i-1], cuts[i]] with right=True, so a value sitting
    exactly on a cut falls into the LOWER slot."""
    spec = {"cuts": [10.0, 30.0, 70.0, 90.0], "labels": list(DEFAULT_LABELS), "tie_map": []}
    assert slot_of(-1.0, spec) == 0
    assert slot_of(9.999, spec) == 0
    assert slot_of(10.0, spec) == 0, "a value exactly on a cut belongs to the lower slot"
    assert slot_of(10.001, spec) == 1
    assert slot_of(30.0, spec) == 1
    assert slot_of(30.001, spec) == 2
    assert slot_of(70.0, spec) == 2
    assert slot_of(70.001, spec) == 3
    assert slot_of(90.0, spec) == 3
    assert slot_of(90.001, spec) == 4
    assert tier_of(90.001, spec) == "very high"
    assert tier_of(10.0, spec) == "very low"


def test_bucket_boundary_semantics_duplicate_cuts():
    """Duplicated cut values resolve through tie_map, and the direction depends on where the tied
    mass sits. Both real TwiBot-20 shapes are pinned here because a single global direction renders
    a falsehood on one of them:

    * digits_in_sn cuts [0,0,0,2], mass at the BOTTOM -> a digit-free handle must not read high;
    * uniq_tweet_ratio cuts [0.99,1,1,1], mass at the TOP -> an all-distinct timeline must not
      read low.
    """
    low_mass = {  # 76% of accounts sit exactly at 0 -> midpoint percentile 38 -> slot 2
        "cuts": [0.0, 0.0, 0.0, 2.0], "labels": list(DEFAULT_LABELS),
        "tie_map": [{"value": 0.0, "slot": 2, "pct_below": 0.0, "pct_at_or_below": 76.0,
                     "pct_midpoint": 38.0}],
    }
    assert slot_of(0.0, low_mass) == 2, "tied mass is resolved by its percentile midpoint"
    assert slot_of(1.0, low_mass) == 3, "non-tied values keep the plain right=True rule"
    assert slot_of(2.0, low_mass) == 3
    assert slot_of(2.5, low_mass) == 4
    assert len(low_mass["cuts"]) == 4, "the four raw cuts are never collapsed"

    high_mass = {  # 75% of accounts sit exactly at 1.0 -> midpoint percentile 62.5 -> slot 2
        "cuts": [0.99, 1.0, 1.0, 1.0], "labels": list(DEFAULT_LABELS),
        "tie_map": [{"value": 1.0, "slot": 2, "pct_below": 25.0, "pct_at_or_below": 100.0,
                     "pct_midpoint": 62.5}],
    }
    assert slot_of(1.0, high_mass) == 2
    assert tier_of(1.0, high_mass) not in ("very low", "low"), (
        "a fully distinct timeline must never be rendered as sitting low in the distribution"
    )
    assert slot_of(0.5, high_mass) == 0


def test_real_tie_maps_are_consistent_with_train(thresholds, splits):
    """Recompute every stored tie_map entry from the raw train data."""
    matrix = np.asarray([extract_features(r) for r in splits["train"]], dtype=np.float64)
    with_tweets = np.asarray([bool(r.get("tweet")) for r in splits["train"]])
    checked = 0
    for name, spec in thresholds["features"].items():
        index = spec["index"]
        column = matrix[with_tweets, index] if index in features_v4.TWEET_STAT_INDICES else matrix[:, index]
        for entry in spec["tie_map"]:
            value = entry["value"]
            below = 100.0 * float((column < value).sum()) / column.size
            at_or_below = 100.0 * float((column <= value).sum()) / column.size
            assert entry["pct_below"] == pytest.approx(below, abs=1e-9)
            assert entry["pct_at_or_below"] == pytest.approx(at_or_below, abs=1e-9)
            midpoint = 0.5 * (below + at_or_below)
            lo = thresholds_v4.QUANTILE_EDGES[entry["slot"]]
            hi = thresholds_v4.QUANTILE_EDGES[entry["slot"] + 1]
            assert lo <= midpoint <= hi, (
                f"{name}: tie slot {entry['slot']} bracket p{lo:g}-p{hi:g} does not contain the "
                f"tied mass midpoint p{midpoint:.2f}"
            )
            checked += 1
    assert checked > 0, "expected at least one duplicated cut in the real thresholds"


def test_tied_features_keep_positional_labels(thresholds):
    """Absolute wording ('almost none', 'all distinct') cannot be trusted once a slot spans a large
    tied mass, so features with duplicated cuts must fall back to the positional vocabulary."""
    for name, spec in thresholds["features"].items():
        if spec["n_distinct_cuts"] < 4:
            assert spec["labels"] == list(DEFAULT_LABELS), (
                f"{name} has tied cuts and must keep positional labels, got {spec['labels']}"
            )


def test_labels_are_digit_free_and_five_wide(thresholds):
    for name, spec in thresholds["features"].items():
        assert len(spec["labels"]) == 5
        for label in spec["labels"]:
            assert not re.search(r"\d", label), f"label {label!r} of {name} contains a digit"
    for name in FEATURE_NAMES:
        assert len(thresholds_v4.labels_for(name)) == 5


def test_thresholds_file_is_read_only_across_splits(sampled, thresholds):
    """Rendering dev/test must not touch the thresholds artifact."""
    before = thresholds_v4.sha256_of(THRESHOLDS_PATH)
    before_mtime = THRESHOLDS_PATH.stat().st_mtime_ns
    for split in ("dev", "test"):
        for record in sampled[split]:
            build_units_v4(record, thresholds)
    assert thresholds_v4.sha256_of(THRESHOLDS_PATH) == before
    assert THRESHOLDS_PATH.stat().st_mtime_ns == before_mtime


def test_dev_test_tiers_come_from_train_cuts(sampled, thresholds):
    """A dev/test account's tier must be reproducible from the frozen cuts alone."""
    for split in ("dev", "test"):
        for record in sampled[split]:
            values = extract_features(record)
            for unit in build_units_v4(record, thresholds):
                if "tier" not in unit.hidden:
                    continue
                index = unit.hidden["feature_indices"][0]
                spec = thresholds["features"][FEATURE_NAMES[index]]
                assert unit.hidden["tier"] == tier_of(values[index], spec)


# --------------------------------------------------------------------------------------------
# 3. rendering invariants
# --------------------------------------------------------------------------------------------
def _has_numeric_glyph(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def test_no_numeric_glyphs_anywhere(sampled, thresholds):
    """The enforced invariant is str.isdigit()-free, stricter than \\d-free: no decimal digit AND
    no superscript/circled/parenthesised digit reaches the model."""
    for split, records in sampled.items():
        for record in records:
            for unit in build_units_v4(record, thresholds):
                assert not _has_numeric_glyph(unit.text), (
                    f"{split}/{record.get('ID')} unit {unit.unit_id} leaked a numeric glyph: {unit.text!r}"
                )


def test_digit_free_covers_unicode_and_special_digits():
    # decimal digits (ASCII / fullwidth / Arabic-Indic)
    assert not _has_numeric_glyph(digit_free("followers 12,345 and ６７８ and ٤٥"))
    # the exact glyphs the full build found (superscripts, subscripts, circled/parenthesised)
    for glyph in ["¹", "²", "⁶", "₂", "①", "②", "③", "❶", "➊"]:
        assert glyph.isdigit(), f"{glyph!r} should be isdigit for this test to be meaningful"
        assert not _has_numeric_glyph(digit_free(f"text {glyph} more")), f"{glyph!r} not folded"
    # real strings from the shards
    assert not _has_numeric_glyph(digit_free("Kings ¹⁶ He answered"))
    assert not _has_numeric_glyph(digit_free("スローループ①〜③巻が重版"))
    # date/separator merging semantics preserved: a run collapses to a single 'number'
    assert digit_free("on 12/34/5678") == "on number"
    # Chinese numeral WORDS must NOT be stripped (isdigit is False for them)
    assert digit_free("第一二三十") == "第一二三十"
    assert digit_free("no digits here") == "no digits here"


def test_unit_ids_valid_unique_and_contiguous(sampled, thresholds):
    pattern = re.compile(r"^(PROFILE|OBS|ACTIVITY|RELATION|TWEET)_\d{2}$")
    for split, records in sampled.items():
        for record in records:
            units = build_units_v4(record, thresholds)
            ids = [u.unit_id for u in units]
            assert len(ids) == len(set(ids)), f"{split}/{record.get('ID')} has duplicate unit ids"
            for uid in ids:
                assert pattern.match(uid), f"malformed unit id {uid!r}"
            by_prefix: dict[str, list[int]] = {}
            for uid in ids:
                prefix, number = uid.split("_")
                by_prefix.setdefault(prefix, []).append(int(number))
            for prefix, numbers in by_prefix.items():
                assert numbers == list(range(1, len(numbers) + 1)), (
                    f"{split}/{record.get('ID')} {prefix} numbering is not contiguous: {numbers}"
                )


# The public unit-ID contract the evidence bank and reward gate are keyed to. Maps each bucketed
# unit id to (feature_name, feature_index) of its PRIMARY (headline) clause. A silent renumber here
# would make ACTIVITY_07 point at std_tweet_len instead of the distinct-post ratio the bank expects,
# which the reward would never notice — so it is pinned mechanically.
CONTRACT: dict[str, tuple[str, int]] = {
    "PROFILE_02": ("log_followers", 0),
    "PROFILE_03": ("log_friends", 1),
    "PROFILE_04": ("log_statuses", 2),
    "PROFILE_05": ("log_listed", 4),
    "PROFILE_06": ("log_favourites", 3),
    "PROFILE_07": ("log_age", 11),
    "OBS_01": ("friends/followers", 5),
    "OBS_02": ("followers/friends", 6),
    "OBS_03": ("statuses/age", 7),
    "OBS_04": ("followers/age", 9),
    "OBS_05": ("fav/age", 8),
    "OBS_06": ("digit_ratio_sn", 15),
    "OBS_07": ("len_desc", 16),
    "OBS_08": ("len_name", 12),
    "ACTIVITY_01": ("retweet_ratio", 26),
    "ACTIVITY_02": ("url_per_tweet", 30),
    "ACTIVITY_03": ("hashtag_per_tweet", 31),
    "ACTIVITY_04": ("mention_per_tweet", 32),
    "ACTIVITY_05": ("mean_tweet_len", 27),
    "ACTIVITY_06": ("std_tweet_len", 28),
    "ACTIVITY_07": ("uniq_tweet_ratio", 29),
}
EXPECTED_ID_ORDER = (
    [f"PROFILE_{i:02d}" for i in range(1, 10)]
    + [f"OBS_{i:02d}" for i in range(1, 9)]
    + [f"ACTIVITY_{i:02d}" for i in range(1, 8)]
)


def test_public_unit_id_contract(thresholds, splits):
    """Every account with a timeline must expose exactly PROFILE_01..09, OBS_01..08, ACTIVITY_01..07
    (in that order, before RELATION/TWEET), and every bucketed unit's primary clause must map to the
    contracted feature index. This is the guard Codex asked for against a silent renumber."""
    for split, records in splits.items():
        for record in records:
            if not record.get("tweet"):
                continue
            units = build_units_v4(record, thresholds)
            by_id = {u.unit_id: u for u in units}
            structured = [u.unit_id for u in units
                          if u.unit_id.split("_")[0] in {"PROFILE", "OBS", "ACTIVITY"}]
            assert structured == EXPECTED_ID_ORDER, (
                f"{split}/{record.get('ID')} unit-id contract broken: {structured}"
            )
            for unit_id, (name, index) in CONTRACT.items():
                clause = by_id[unit_id].clauses[0]
                assert clause["feature_index"] == index, (
                    f"{unit_id} primary clause is index {clause['feature_index']}, contract says {index}"
                )
                assert clause["name"] == name
                assert FEATURE_NAMES[index] == name


def test_contract_pins_the_easily_confused_ids(thresholds, splits):
    """The three ids most likely to silently drift, pinned explicitly."""
    record = next(r for r in splits["train"] if r.get("tweet"))
    by_id = {u.unit_id: u for u in build_units_v4(record, thresholds)}
    assert by_id["ACTIVITY_07"].clauses[0]["feature_index"] == 29  # uniq_tweet_ratio, NOT std
    assert by_id["ACTIVITY_07"].clauses[0]["name"] == "uniq_tweet_ratio"
    assert by_id["OBS_04"].clauses[0]["feature_index"] == 9        # followers/age
    assert by_id["OBS_05"].clauses[0]["feature_index"] == 8        # fav/age


def test_obs08_renders_both_name_and_handle(thresholds, splits):
    record = next(r for r in splits["train"] if r.get("tweet"))
    obs08 = {u.unit_id: u for u in build_units_v4(record, thresholds)}["OBS_08"]
    idxs = [c["feature_index"] for c in obs08.clauses]
    assert idxs == [12, 13], "OBS_08 must present both name length and handle length"
    assert "display-name length" in obs08.text.lower()
    assert "handle length" in obs08.text.lower()


def test_n_tweets_in_vector_but_not_in_prompt(thresholds, splits):
    """index 25 (n_tweets) must stay in the E1 vector yet never reach the prompt (it is the 200-cap,
    not a real activity level)."""
    for split, records in splits.items():
        for record in records[:200]:
            assert len(extract_features(record)) == N_FEATURES  # 25 present in the vector
            covered = covered_feature_indices(build_units_v4(record, thresholds))
            assert 25 not in covered, f"{split}/{record.get('ID')} rendered n_tweets into the prompt"


def test_prompt_covers_every_feature_except_n_tweets(thresholds, splits):
    """Guard against regressing to v3's 7-field prompt (which the RF ablation puts 3.5pp lower):
    every feature except n_tweets must be reachable through the prompt on a full account."""
    record = next(r for r in splits["train"] if r.get("tweet"))
    covered = covered_feature_indices(build_units_v4(record, thresholds))
    assert covered == set(range(N_FEATURES)) - {25}, (
        f"prompt coverage off: missing {sorted(set(range(N_FEATURES)) - {25} - covered)}, "
        f"unexpected {sorted(covered - (set(range(N_FEATURES)) - {25}))}"
    )


def test_relation_uses_train_fitted_tiers(thresholds, splits):
    """RELATION_01 must present both neighbour counts as tiers derived from the frozen relation
    cuts, and be absent when the neighbor field is missing."""
    from verifiable_bot.v4.thresholds_v4 import tier_of
    rel = thresholds["relation"]
    seen_present = seen_absent = 0
    for record in splits["dev"]:
        units = {u.unit_id: u for u in build_units_v4(record, thresholds)}
        counts = features_v4.neighbor_counts(record)
        if counts is None:
            assert "RELATION_01" not in units
            seen_absent += 1
            continue
        seen_present += 1
        clauses = units["RELATION_01"].clauses
        assert [c["name"] for c in clauses] == ["neighbor_following_count", "neighbor_follower_count"]
        assert clauses[0]["tier"] == tier_of(float(counts[0]), rel["neighbor_following_count"])
        assert clauses[1]["tier"] == tier_of(float(counts[1]), rel["neighbor_follower_count"])
    assert seen_present > 0 and seen_absent > 0


def test_exact_facts_use_absolute_wording(thresholds):
    """digit-free handle and fully-distinct timeline read as absolute truths, not positional tiers,
    while the five-slot tier is still recorded in the clause sidecar."""
    record = _synthetic(
        tweet=["alpha beta", "gamma delta", "epsilon zeta"],  # all distinct, no retweets
    )
    record["profile"]["screen_name"] = "plainname "  # no digits
    by_id = {u.unit_id: u for u in build_units_v4(record, thresholds)}
    assert "contains no digits" in by_id["OBS_06"].text
    assert by_id["OBS_06"].clauses  # tier still recorded
    assert "distinct" in by_id["ACTIVITY_07"].text.lower()
    assert "retweets" in by_id["ACTIVITY_01"].text.lower()


def test_verified_is_rendered_first(thresholds, splits):
    """`verified` carries ~19% of the RF's importance, the single strongest feature."""
    units = build_units_v4(splits["train"][0], thresholds)
    assert units[0].unit_id == "PROFILE_01"
    assert "erification status" in units[0].text


# --------------------------------------------------------------------------------------------
# 4. degradation: empty timeline, missing neighbor
# --------------------------------------------------------------------------------------------
def _synthetic(**overrides) -> dict:
    record = {
        "ID": "synthetic_1 ",
        "profile": {
            "name": "Test ", "screen_name": "test01 ", "description": "hello 123 ",
            "followers_count": "10 ", "friends_count": "0 ", "statuses_count": "5 ",
            "favourites_count": "0 ", "listed_count": "0 ",
            "created_at": "Tue Nov 18 10:27:25 +0000 2008 ", "verified": "False ",
            "default_profile": "True ", "default_profile_image": "True ",
            "geo_enabled": "False ", "protected": "False ", "url": None, "location": None,
        },
        "tweet": ["hello world", "RT @someone: hi"],
        "neighbor": {"following": ["a", "b"], "follower": ["c"]},
        "label": "1",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize("empty", [[], None])
def test_empty_timeline_degrades_without_dividing_by_zero(thresholds, empty):
    record = _synthetic(tweet=empty)
    values = extract_features(record)          # must not raise
    assert len(values) == N_FEATURES
    for index in features_v4.TWEET_STAT_INDICES:
        assert values[index] == 0.0, "tweet statistics of an empty timeline are zero, per the baseline"
    units = build_units_v4(record, thresholds)
    assert not [u for u in units if u.unit_id.startswith("ACTIVITY_")], (
        "an account with no timeline must get zero ACTIVITY units: its statistics are undefined, "
        "and rendering them as 'very low' would be a false statement"
    )
    assert not [u for u in units if u.unit_id.startswith("TWEET_")]
    assert [u for u in units if u.unit_id.startswith("PROFILE_")]


@pytest.mark.parametrize("neighbor", [None, {}, {"following": [], "follower": []}])
def test_missing_neighbor_degrades(thresholds, neighbor):
    units = build_units_v4(_synthetic(neighbor=neighbor), thresholds)
    assert not [u for u in units if u.unit_id.startswith("RELATION_")]


def test_zero_denominators_everywhere(thresholds):
    """Followers, friends, screen name, description and timeline all empty at once."""
    record = _synthetic(tweet=[], neighbor=None)
    record["profile"].update({
        "followers_count": "0 ", "friends_count": "0 ", "screen_name": " ",
        "description": " ", "created_at": "not a date ",
    })
    values = extract_features(record)
    assert all(math.isfinite(v) for v in values)
    build_units_v4(record, thresholds)


def test_real_accounts_without_tweets_get_no_activity_units(splits, thresholds):
    """The 80 real accounts (train 55 / dev 15 / test 10) with an empty timeline."""
    checked = 0
    for split, records in splits.items():
        for record in records:
            if record.get("tweet"):
                continue
            checked += 1
            units = build_units_v4(record, thresholds)
            assert not [u for u in units if u.unit_id.startswith("ACTIVITY_")]
    assert checked >= 70, f"expected ~80 tweet-less accounts across the splits, found {checked}"


# --------------------------------------------------------------------------------------------
# 5. canonical prompt formatter, adaptive budget, sidecar (Codex rounds 3 / 3.5)
# --------------------------------------------------------------------------------------------
from verifiable_bot.v4 import prompt_v4                                    # noqa: E402
from verifiable_bot.v4.render_v4 import record_payload, render_units, unit_metadata  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(prompt_v4.default_model_path(), trust_remote_code=True)


def test_canonical_prompt_deterministic_and_uses_chat_template(tokenizer, thresholds, splits):
    units_text = render_units(build_units_v4(splits["train"][0], thresholds))
    a = prompt_v4.canonical_prompt_text(units_text, tokenizer)
    b = prompt_v4.canonical_prompt_text(units_text, tokenizer)
    assert a == b, "canonical prompt must be deterministic"
    assert "forensics analyst" in a, "system prompt must be present"
    assert prompt_v4.count_tokens(units_text, tokenizer) == len(
        tokenizer.apply_chat_template(prompt_v4.canonical_messages(units_text),
                                      tokenize=True, add_generation_prompt=True))


def test_canonical_prompt_hash_stable_and_covers_semantics(tokenizer):
    h1 = prompt_v4.canonical_prompt_sha256(tokenizer)
    h2 = prompt_v4.canonical_prompt_sha256(tokenizer)
    assert h1 == h2 and len(h1) == 64
    fp = prompt_v4.tokenizer_fingerprint(tokenizer, prompt_v4.default_model_path())
    assert fp["snapshot"] and fp["tokenizer_config_sha256"], "fingerprint must carry snapshot + config hash"


def test_adaptive_budget_never_overflows(tokenizer, sampled, thresholds):
    for split, records in sampled.items():
        for record in records:
            units, tokens, cap = prompt_v4.fit_account_prompt(record, thresholds, tokenizer)
            assert tokens <= prompt_v4.PROMPT_BUDGET, (
                f"{split}/{record.get('ID')} fitted to {tokens} > {prompt_v4.PROMPT_BUDGET}")
            assert cap <= prompt_v4.CANDIDATE_MAX_TWEETS


def _forced_reduction_budget(record, thresholds, tokenizer):
    """A budget that provably fits the structural units yet forces at least one tweet to be dropped,
    computed from THIS record's own cap-30 and cap-0 counts so it is independent of tokenizer version."""
    full = prompt_v4.count_tokens(
        render_units(build_units_v4(record, thresholds, max_tweets=prompt_v4.CANDIDATE_MAX_TWEETS)), tokenizer)
    structural = prompt_v4.count_tokens(
        render_units(build_units_v4(record, thresholds, max_tweets=0)), tokenizer)
    assert full > structural, "test record must have tweets that add tokens"
    return max(structural, full - 1)


def test_structural_units_survive_tweet_reduction(tokenizer, thresholds):
    """Under budget pressure the fitter drops TWEET units, never a PROFILE/OBS/ACTIVITY/RELATION."""
    record = _synthetic(tweet=[f"distinct sentence number {i} about weather sports politics" for i in range(200)])
    budget = _forced_reduction_budget(record, thresholds, tokenizer)
    units, tokens, cap = prompt_v4.fit_account_prompt(record, thresholds, tokenizer, budget=budget)
    assert tokens <= budget
    ids = {u.unit_id for u in units}
    for uid in ([f"PROFILE_{i:02d}" for i in range(1, 10)]
                + [f"OBS_{i:02d}" for i in range(1, 9)]
                + [f"ACTIVITY_{i:02d}" for i in range(1, 8)]):
        assert uid in ids, f"structural unit {uid} was dropped under budget pressure"
    assert cap < prompt_v4.CANDIDATE_MAX_TWEETS, "budget pressure must lower the effective cap"
    assert any(u.unit_id.startswith("TWEET_") for u in units), "some tweets should still fit"


def test_unit_metadata_one_to_one_with_units(thresholds, splits):
    record = next(r for r in splits["train"] if r.get("tweet"))
    units = build_units_v4(record, thresholds)
    meta = unit_metadata(units)
    assert [m["unit_id"] for m in meta] == [u.unit_id for u in units], "sidecar must align 1:1"
    for m, u in zip(meta, units):
        assert m["clauses"] == u.clauses
        assert m["feature_indices"] == u.feature_indices


def test_stored_token_count_matches_recount(tokenizer, thresholds, splits):
    """Every stored prompt_token_count must equal a fresh canonical recount of the same units."""
    for split in ("train", "dev", "test"):
        for record in splits[split][::400]:
            units, tokens, cap = prompt_v4.fit_account_prompt(record, thresholds, tokenizer)
            payload = record_payload(record, units, prompt_token_count=tokens,
                                     candidate_max_tweets=prompt_v4.CANDIDATE_MAX_TWEETS,
                                     effective_tweet_cap=cap)
            recount = prompt_v4.count_tokens(render_units(units), tokenizer)
            assert payload["prompt_token_count"] == recount == tokens


def test_three_tweet_fields_have_distinct_semantics(tokenizer, thresholds):
    """candidate (fixed 30) >= effective cap >= n_tweet_units; and the two 'short' cases differ."""
    # (a) naturally short timeline: cap stays at candidate, few tweets, NOT a budget adaptation.
    short = _synthetic(tweet=["hi there", "second post", "third one"])
    u, t, cap = prompt_v4.fit_account_prompt(short, thresholds, tokenizer)
    p = record_payload(short, u, prompt_token_count=t,
                       candidate_max_tweets=prompt_v4.CANDIDATE_MAX_TWEETS, effective_tweet_cap=cap)
    assert p["candidate_max_tweets"] == 30
    assert p["effective_tweet_cap"] == 30, "a short timeline is not a budget adaptation"
    assert p["n_tweet_units"] == 3 < p["effective_tweet_cap"]

    # (b) budget-driven reduction: a dynamically-computed budget lowers the effective cap below candidate.
    longr = _synthetic(tweet=[f"distinct sentence number {i} weather sports politics" for i in range(200)])
    budget = _forced_reduction_budget(longr, thresholds, tokenizer)
    u2, t2, cap2 = prompt_v4.fit_account_prompt(longr, thresholds, tokenizer, budget=budget)
    p2 = record_payload(longr, u2, prompt_token_count=t2,
                        candidate_max_tweets=prompt_v4.CANDIDATE_MAX_TWEETS, effective_tweet_cap=cap2)
    assert p2["effective_tweet_cap"] < p2["candidate_max_tweets"], "budget pressure lowered the cap"
    assert p2["n_tweet_units"] <= p2["effective_tweet_cap"]


def test_payload_units_carry_no_sidecar_fields(tokenizer, thresholds, splits):
    record = next(r for r in splits["train"] if r.get("tweet"))
    units, tokens, cap = prompt_v4.fit_account_prompt(record, thresholds, tokenizer)
    payload = record_payload(record, units, prompt_token_count=tokens,
                             candidate_max_tweets=prompt_v4.CANDIDATE_MAX_TWEETS, effective_tweet_cap=cap)
    for u in payload["units"]:
        assert set(u) == {"unit_id", "text"}, f"unit leaked sidecar fields: {set(u)}"
    prompt = prompt_v4.canonical_prompt_text(render_units(units), tokenizer)
    assert all(u["text"] in prompt for u in payload["units"]), "every unit text is in the prompt"
    assert not re.search(r"\d", "".join(u["text"] for u in payload["units"])), "unit prose has no digits"


def test_empty_timeline_fits_immediately(tokenizer, thresholds):
    record = _synthetic(tweet=[])
    units, tokens, cap = prompt_v4.fit_account_prompt(record, thresholds, tokenizer)
    assert prompt_v4.n_tweet_units(units) == 0
    assert tokens <= prompt_v4.PROMPT_BUDGET
    assert not [u for u in units if u.unit_id.startswith("ACTIVITY_")]
