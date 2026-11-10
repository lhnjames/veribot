"""Replay first/last sampled-tweet removal through the frozen A/B group tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
from lightgbm import Booster
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import design, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_facts import account_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


TWEET_FACTS = GROUPS["sampled_tweet_stats"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--group-screen", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corrected-screen", action="store_true",
                        help="use all-provided sampled-graph refit and cards")
    parser.add_argument("--split-visible-screen", action="store_true",
                        help="use split-visible sampled-graph refit and cards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    screen = json.loads(args.group_screen.read_text(encoding="utf-8"))
    cards = json.loads(args.decision_cards.read_text(encoding="utf-8"))
    if args.corrected_screen and args.split_visible_screen:
        raise ValueError("Choose one graph visibility protocol")
    if (sha256(args.zip) != cards["source_archive_sha256"]
            or sha256(args.facts) != cards["fact_cache_sha256"]
            or sha256(args.reference) != cards["reference_artifact_sha256"]
            or sha256(args.model) != cards["frozen_model_sha256"]
            or sha256(args.group_screen) != cards["group_screen_sha256"]
            or (not (args.corrected_screen or args.split_visible_screen) and
                screen["compact_sha256"] != sha256(args.compact))):
        raise ValueError("Frozen source, reference, model or cards differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        positions = saved["positions"].astype(int)
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if args.corrected_screen or args.split_visible_screen:
        if (screen["corrected_fact_cache_sha256" if args.corrected_screen
                   else "split_visible_facts_sha256"] != sha256(args.facts)
                or screen["corrected_reference_sha256" if args.corrected_screen
                          else "reference_sha256"] != sha256(args.reference)
                or screen["corrected_seed0_model_sha256" if args.corrected_screen
                          else "seed0_model_sha256"] != sha256(args.model)):
            raise ValueError("Source-visible screen does not bind source and model")
        terms = cards["decision_cards"][0]["fact_terms"]
        if [term["fact_name"] for term in terms] != names:
            raise ValueError("Corrected card fact order differs")
        signs = np.r_[[term["upper_monotone_direction"] for term in terms],
                      [term["lower_monotone_direction"] for term in terms]]
    else:
        chosen = next(row for row in screen["rows"]
                      if row["support_aware"] and row["ab_seed"] == cards["a_seed"])
        signs = np.r_[[chosen["a_signs_by_fact"][name]["upper"] for name in names],
                      [chosen["a_signs_by_fact"][name]["lower"] for name in names]]
    source_rows = []
    with zipfile.ZipFile(args.zip) as zipped, zipped.open("Twibot-20/dev.json") as stream:
        for ordinal, row in enumerate(ijson.items(stream, "item")):
            if ordinal >= len(dev) or str(row["ID"]) != ids[dev[ordinal]] or \
                    positions[dev[ordinal]] != ordinal:
                raise ValueError("Official dev row order differs")
            direct, _, context = account_facts(row)
            if context != contexts[dev[ordinal]] or not np.allclose(
                    direct, values[dev[ordinal], :len(direct)],
                    atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError("Official original direct facts differ")
            source_rows.append(row)
    if len(source_rows) != len(dev):
        raise ValueError("Official dev account count differs")
    reference = LabelBlindReference.load(args.reference)
    model = Booster(model_file=str(args.model))
    keep = [all_names.index(name) for name in names]
    groups = source_groups(names)
    tweet_group_index = list(GROUPS).index("sampled_tweet_stats")
    tweet_cols = groups[tweet_group_index]
    original = values[dev].copy()
    original_support = original[:, all_names.index("tweet_sample_count")]
    original_batch = evidence_subset(
        reference.transform(original, contexts=contexts[dev]), keep, names,
        original_support, True)
    original_x = design(original_batch, signs)
    original_logit = np.asarray(model.predict(original_x, raw_score=True))
    card_rows = cards["decision_cards"]
    if ids[dev].tolist() != [card["account_id"] for card in card_rows] or \
            np.max(np.abs(original_logit - [card["full_logit"] for card in card_rows])) > 1e-12:
        raise ValueError("Frozen original dev logits do not replay cards")
    threshold = float(card_rows[0]["threshold"])
    results = {}
    for edit_name, remove_at in (("first", 0), ("last", -1)):
        edited = original.copy()
        source_locator = []
        for k, row in enumerate(source_rows):
            tweets = list(row.get("tweet") or [])
            if not tweets:
                source_locator.append(None)
                continue
            removed_position = remove_at if remove_at >= 0 else len(tweets) - 1
            changed = dict(row)
            changed["tweet"] = tweets[:removed_position] + tweets[removed_position+1:]
            direct, _, context = account_facts(changed)
            if context != contexts[dev[k]]:
                raise ValueError("Context changed after tweet record removal")
            edited[k, :len(direct)] = direct
            source_locator.append(f"Twibot-20/dev.json[{positions[dev[k]]}].tweet[{removed_position}]")
        edited_support = edited[:, all_names.index("tweet_sample_count")]
        edited_batch = evidence_subset(
            reference.transform(edited, contexts=contexts[dev]), keep, names,
            edited_support, True)
        edited_x = design(edited_batch, signs)
        other = [j for j in range(original_x.shape[1]) if j not in tweet_cols]
        unchanged_error = float(np.max(np.abs(edited_x[:, other] - original_x[:, other])))
        if unchanged_error > 1e-12:
            raise ValueError("Source edit unexpectedly changed non-tweet model inputs")
        edited_logit = np.asarray(model.predict(edited_x, raw_score=True))
        tweet_only = original_x.copy()
        tweet_only[:, tweet_cols] = edited_x[:, tweet_cols]
        source_response_error = float(np.max(np.abs(
            edited_logit - model.predict(tweet_only, raw_score=True))))
        if source_response_error > 1e-12:
            raise ValueError("Edited source does not replay through tweet group")
        changed_facts = {}
        for name in TWEET_FACTS:
            j = all_names.index(name)
            before, after = original[:, j], edited[:, j]
            changed_facts[name] = int(np.count_nonzero(
                ~np.isclose(before, after, rtol=0, atol=1e-12, equal_nan=True)))
        original_class = expit(original_logit) >= threshold
        edited_class = expit(edited_logit) >= threshold
        rows = [{
            "account_id": str(ids[dev[k]]),
            "source_locator_removed": source_locator[k],
            "sampled_tweets_before": int(original_support[k]),
            "sampled_tweets_after": int(edited_support[k]),
            "original_logit": float(original_logit[k]),
            "edited_logit": float(edited_logit[k]),
            "logit_change": float(edited_logit[k] - original_logit[k]),
            "original_class": int(original_class[k]),
            "edited_class": int(edited_class[k]),
            "evaluation_label": int(labels[dev[k]]),
        } for k in range(len(dev))]
        bins = {}
        for label, mask in {
                "one": original_support == 1,
                "two_to_five": (original_support >= 2) & (original_support <= 5),
                "six_to_twenty": (original_support >= 6) & (original_support <= 20),
                "twenty_one_to_199": (original_support >= 21) & (original_support < 200),
                "two_hundred": original_support == 200,
        }.items():
            bins[label] = {"accounts": int(np.count_nonzero(mask)),
                           "class_changes": int(np.count_nonzero(
                               original_class[mask] != edited_class[mask])),
                           "mean_abs_logit_change": float(np.mean(
                               np.abs(edited_logit[mask] - original_logit[mask])))
                           if np.any(mask) else None}
        results[edit_name] = {
            "operation": "remove one observed sampled tweet string and recompute all direct facts, support count, tails, reliability, and frozen model output",
            "accounts_with_nonempty_tweet_list": int(np.count_nonzero(original_support > 0)),
            "changed_fact_accounts": changed_facts,
            "any_logit_change_accounts": int(np.count_nonzero(
                np.abs(edited_logit - original_logit) > 1e-12)),
            "class_changes": int(np.count_nonzero(original_class != edited_class)),
            "bot_to_human": int(np.count_nonzero(original_class & ~edited_class)),
            "human_to_bot": int(np.count_nonzero(~original_class & edited_class)),
            "original_errors": int(np.count_nonzero(original_class != labels[dev])),
            "edited_errors": int(np.count_nonzero(edited_class != labels[dev])),
            "max_non_tweet_input_change": unchanged_error,
            "max_source_response_replay_error": source_response_error,
            "sample_count_bins": bins,
            "rows": rows,
        }
    zero_group_changes = sum((card["group_zeroed_logits"]["sampled_tweet_stats"]
                              >= card["threshold_logit"]) != bool(card["prediction"])
                             for card in card_rows)
    report = {
        "scope": "frozen A/B support-aware group-tree official dev source-record sensitivity"
                 + (", split-visible sampled graph" if args.split_visible_screen
                    else ", all provided sampled graph" if args.corrected_screen
                    else ", labeled-only sampled graph"),
        "candidate_test_evaluated": False,
        "source_archive_sha256": sha256(args.zip),
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "decision_cards_sha256": sha256(args.decision_cards),
        "model_sha256": sha256(args.model),
        "coded_tweet_group_zero_class_changes": zero_group_changes,
        "edits": results,
        "interpretation": "Deleting a sampled tweet is an observed-record sensitivity operation, not a correction of a known natural error. Zeroing an entire encoded evidence group is a different model-input substitution. Neither establishes behavioral truth or cost-effective correction.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coded_tweet_group_zero_class_changes": zero_group_changes,
                      "edits": {name: {key: value for key, value in data.items()
                                        if key != "rows"}
                                for name, data in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
