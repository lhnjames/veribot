"""Enumerate every sampled-tweet deletion through one frozen split-visible model."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import ijson
from lightgbm import Booster
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from audit_twibot20_random_tweet_source_edit_dev import choose
from screen_twibot20_compact20_ab_group_tree_dev import design, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_full_source_audit import direct_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(r"C:\Users\Lenovo\Downloads\Twibot-20.zip")
FACTS = ROOT / "data/processed/twibot20_v3/facts_split_visible_train_dev.npz"
REFERENCE = ROOT / "output/twibot20_compact20_ab_group_tree_dev/reference_split_visible_train_dev.json.gz"
COMPACT = ROOT / "research/certificates/twibot20_compact20_capacity.json"
SCREEN = ROOT / "research/certificates/twibot20_split_visible_graph_ab_dev.json"
MODEL = ROOT / "output/twibot20_compact20_ab_group_tree_dev/seed0_split_visible_train_dev_model.txt"
CARDS = ROOT / "research/certificates/twibot20_split_visible_graph_ab_dev_decisions.json"
RANDOM = ROOT / "research/certificates/twibot20_split_visible_random_tweet_source_edit_dev.json"
OUT = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_responses_dev.json"
DELTAS = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_response_deltas_dev.npz"
CHUNK = 20000
EPS = 1e-12


def all_single_deletions(row: dict, original: np.ndarray,
                         name_index: dict[str, int]) -> np.ndarray:
    tweets = [str(text or "") for text in row.get("tweet") or []]
    n = len(tweets)
    result = np.repeat(original[None, :], n, axis=0)
    if n == 0:
        return result
    frequencies = Counter(tweets)
    flags = {
        "retweet_fraction": np.fromiter(
            (text.lstrip().startswith("RT @") for text in tweets), dtype=float),
        "url_fraction": np.fromiter(
            ("http://" in text or "https://" in text for text in tweets), dtype=float),
        "hashtag_fraction": np.fromiter(
            (re.search(r"(?<!\w)#\w", text) is not None for text in tweets), dtype=float),
        "mention_fraction": np.fromiter(
            (re.search(r"(?<!\w)@\w", text) is not None for text in tweets), dtype=float),
    }
    result[:, name_index["tweet_sample_count"]] = n - 1
    if n == 1:
        for name in (*flags, "mean_tweet_length", "unique_tweet_fraction"):
            result[:, name_index[name]] = np.nan
        return result
    for name, hits in flags.items():
        result[:, name_index[name]] = (float(np.sum(hits)) - hits) / (n - 1)
    lengths = np.asarray([len(text) for text in tweets], dtype=float)
    result[:, name_index["mean_tweet_length"]] = (
        float(np.sum(lengths)) - lengths) / (n - 1)
    unique_after = len(frequencies) - np.asarray(
        [frequencies[text] == 1 for text in tweets], dtype=float)
    result[:, name_index["unique_tweet_fraction"]] = unique_after / (n - 1)
    return result


def selected_record(uid: str, ordinal: int, tweets: list, delta: np.ndarray,
                    slot: int | None) -> dict | None:
    if slot is None:
        return None
    text = str(tweets[slot] or "")
    return {
        "source_locator": f"Twibot-20/dev.json[{ordinal}].tweet[{slot}]",
        "tweet_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "deletion_logit_delta": float(delta[slot]),
    }


def main() -> None:
    cards_artifact = json.loads(CARDS.read_text(encoding="utf-8"))
    random_artifact = json.loads(RANDOM.read_text(encoding="utf-8"))
    compact = json.loads(COMPACT.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (sha256(ARCHIVE) != cards_artifact["source_archive_sha256"]
            or sha256(FACTS) != cards_artifact["fact_cache_sha256"]
            or sha256(REFERENCE) != cards_artifact["reference_artifact_sha256"]
            or sha256(MODEL) != cards_artifact["frozen_model_sha256"]
            or sha256(SCREEN) != cards_artifact["group_screen_sha256"]
            or sha256(CARDS) != random_artifact["cards_sha256"]
            or sha256(MODEL) != random_artifact["model_sha256"]):
        raise ValueError("Official source, frozen model, reference or prior control differs")
    with np.load(FACTS, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        positions = saved["positions"].astype(int)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    dev = np.flatnonzero(split == "dev")
    dev_ids = ids[dev].tolist()
    names = compact["retained_facts"]
    name_index = {name: j for j, name in enumerate(all_names)}
    chosen = next(row for row in screen["rows"]
                  if row["ab_seed"] == cards_artifact["a_seed"])
    signs = np.r_[[chosen["tail_directions_by_fact"][name]["upper"] for name in names],
                  [chosen["tail_directions_by_fact"][name]["lower"] for name in names]]
    reference = LabelBlindReference.load(REFERENCE)
    model = Booster(model_file=str(MODEL))
    keep = [name_index[name] for name in names]
    def encode(matrix: np.ndarray, context: np.ndarray) -> np.ndarray:
        return design(evidence_subset(
            reference.transform(matrix, contexts=context), keep, names,
            matrix[:, name_index["tweet_sample_count"]], True), signs)
    original = values[dev].copy()
    x0 = encode(original, contexts[dev])
    y0 = np.asarray(model.predict(x0, raw_score=True))
    cards = cards_artifact["decision_cards"]
    if (dev_ids != [c["account_id"] for c in cards]
            or np.max(np.abs(y0 - [c["full_logit"] for c in cards])) > EPS):
        raise ValueError("Original frozen predictions differ from cards")
    tweet_cols = source_groups(names)[list(GROUPS).index("sampled_tweet_stats")]
    other_cols = [j for j in range(x0.shape[1]) if j not in tweet_cols]
    threshold = float(cards[0]["threshold"])
    all_offsets = [0]
    flat_delta: list[np.ndarray] = []
    source_rows = []
    pending_values = []
    pending_account = []
    pending_count = 0
    max_non_tweet = 0.0
    max_response = 0.0
    max_formula = 0.0
    def flush() -> None:
        nonlocal pending_count, max_non_tweet, max_response
        if not pending_count:
            return
        matrix = np.concatenate(pending_values)
        account = np.concatenate(pending_account)
        edited_x = encode(matrix, contexts[dev[account]])
        max_non_tweet = max(max_non_tweet, float(np.max(np.abs(
            edited_x[:, other_cols] - x0[account][:, other_cols]))))
        if max_non_tweet > EPS:
            raise ValueError("Tweet deletion changed non-tweet model input")
        edited_y = np.asarray(model.predict(edited_x, raw_score=True))
        hybrid = x0[account].copy()
        hybrid[:, tweet_cols] = edited_x[:, tweet_cols]
        max_response = max(max_response, float(np.max(np.abs(
            edited_y - model.predict(hybrid, raw_score=True)))))
        if max_response > EPS:
            raise ValueError("Edited tweet source does not replay frozen response")
        flat_delta.append(edited_y - y0[account])
        pending_values.clear()
        pending_account.clear()
        pending_count = 0
    with zipfile.ZipFile(ARCHIVE) as zipped, zipped.open("Twibot-20/dev.json") as stream:
        for k, row in enumerate(ijson.items(stream, "item")):
            if (k >= len(dev) or str(row["ID"]) != dev_ids[k]
                    or positions[dev[k]] != k):
                raise ValueError("Official dev source row differs")
            direct = direct_facts(row)
            if not np.allclose(direct, original[k, :len(direct)],
                               rtol=0, atol=EPS, equal_nan=True):
                raise ValueError(f"Original direct source facts differ: {dev_ids[k]}")
            tweets = list(row.get("tweet") or [])
            n = len(tweets)
            edits = all_single_deletions(row, original[k], name_index)
            if n:
                check_slot = choose(dev_ids[k], n)
                changed = dict(row)
                changed["tweet"] = tweets[:check_slot] + tweets[check_slot + 1:]
                actual = direct_facts(changed)
                observed = edits[check_slot, :len(actual)]
                if not np.allclose(observed, actual, rtol=0, atol=EPS,
                                   equal_nan=True):
                    raise ValueError(f"Incremental source edit differs: {dev_ids[k]}")
                finite = np.isfinite(observed) & np.isfinite(actual)
                if finite.any():
                    max_formula = max(max_formula, float(np.max(np.abs(
                        observed[finite] - actual[finite]))))
                pending_values.append(edits)
                pending_account.append(np.full(n, k, dtype=np.int32))
                pending_count += n
            source_rows.append(row)
            all_offsets.append(all_offsets[-1] + n)
            if pending_count >= CHUNK:
                flush()
    flush()
    if len(source_rows) != len(dev):
        raise ValueError("Official dev source population differs")
    delta = np.concatenate(flat_delta)
    offsets = np.asarray(all_offsets, dtype=np.int64)
    if len(delta) != offsets[-1] or not np.isfinite(delta).all():
        raise ValueError("Exhaustive source response population differs")
    summaries = []
    random_error = 0.0
    for k, row in enumerate(source_rows):
        uid = dev_ids[k]
        tweets = list(row.get("tweet") or [])
        d = delta[offsets[k]:offsets[k+1]]
        prior = random_artifact["rows"][k]
        if prior["account_id"] != uid:
            raise ValueError("Random source control order differs")
        if len(d):
            spot = choose(uid, len(d))
            if prior["position"] != spot:
                raise ValueError("Random source control position differs")
            random_error = max(random_error, abs(float(d[spot]) -
                                                 float(prior["logit_change"])))
        else:
            if prior["position"] is not None:
                raise ValueError("Empty tweet list differs from random control")
        nonzero = np.flatnonzero(np.abs(d) > EPS)
        if len(nonzero):
            positive = int(np.argmax(d)) if np.max(d) > EPS else None
            negative = int(np.argmin(d)) if np.min(d) < -EPS else None
            strongest = int(np.argmax(np.abs(d)))
            strongest_ties = int(np.count_nonzero(np.isclose(
                np.abs(d), np.abs(d[strongest]), rtol=0, atol=EPS)))
        else:
            positive = negative = strongest = None
            strongest_ties = 0
        after_class = expit(y0[k] + d) >= threshold
        before_class = bool(cards[k]["prediction"])
        summaries.append({
            "account_id": uid,
            "official_source_locator": f"Twibot-20/dev.json[{k}]",
            "sampled_tweet_count": len(d),
            "nonzero_single_tweet_responses": int(len(nonzero)),
            "single_tweet_deletions_changing_class": int(np.count_nonzero(
                after_class != before_class)),
            "top_positive_deletion_delta": selected_record(uid, k, tweets, d, positive),
            "top_negative_deletion_delta": selected_record(uid, k, tweets, d, negative),
            "strongest_absolute_deletion_delta": selected_record(uid, k, tweets, d, strongest),
            "strongest_absolute_tie_count": strongest_ties,
            "original_prediction": int(before_class),
            "tweet_group_all_zero_logit_effect": float(
                cards[k]["group_logit_effects"]["sampled_tweet_stats"]),
        })
    if random_error > EPS:
        raise ValueError("Independent random-position source responses differ")
    np.savez_compressed(DELTAS, offsets=offsets, deletion_logit_delta=delta,
                        dev_ids=np.asarray(dev_ids),
                        cards_sha256=np.asarray(sha256(CARDS)),
                        source_archive_sha256=np.asarray(sha256(ARCHIVE)),
                        model_sha256=np.asarray(sha256(MODEL)))
    report = {
        "scope": "all sampled tweet deletion responses on frozen split-visible dev model",
        "source_archive_sha256": sha256(ARCHIVE),
        "facts_sha256": sha256(FACTS),
        "reference_sha256": sha256(REFERENCE),
        "cards_sha256": sha256(CARDS),
        "model_sha256": sha256(MODEL),
        "random_control_sha256": sha256(RANDOM),
        "full_deltas_npz_sha256": sha256(DELTAS),
        "candidate_test_evaluated": False,
        "dev_accounts": len(dev),
        "accounts_with_tweets": sum(s["sampled_tweet_count"] > 0 for s in summaries),
        "source_deletions": len(delta),
        "accounts_with_any_nonzero_single_tweet_response": sum(
            s["nonzero_single_tweet_responses"] > 0 for s in summaries),
        "accounts_with_any_single_tweet_class_flip": sum(
            s["single_tweet_deletions_changing_class"] > 0 for s in summaries),
        "nonzero_source_deletions": sum(s["nonzero_single_tweet_responses"]
                                        for s in summaries),
        "class_changing_source_deletions": sum(
            s["single_tweet_deletions_changing_class"] for s in summaries),
        "max_original_to_incremental_formula_error": max_formula,
        "max_non_tweet_model_input_change": max_non_tweet,
        "max_tweet_group_response_error": max_response,
        "max_random_control_response_difference": random_error,
        "rows": summaries,
        "interpretation": "Every published sampled tweet is tested by deletion in its own dev account, with all affected tweet summary facts and support reliability recomputed. The selected maximum is a model-response summary, not semantic evidence truth or a verified natural correction. Zero and opposing responses remain in the full delta array.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
