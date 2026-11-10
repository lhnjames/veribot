"""Trace every Fox8 tweet-ratio group flip to raw tweet records."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from gate0_shallow_tree import sha256_file
from lock_fox8_single_fact_baseline import ROOT, population
from verifiable_bot.fox8_twenty_neutral_anchor_runtime import (
    Fox8TwentyNeutralAnchorRuntime)


LEDGER = ROOT / "research/certificates/three_dataset_signed_group_ledger.json"
RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
OUT = ROOT / "research/certificates/fox8_raw_activity_mix_decision_cases.json"
GROUP = "sample_activity_mix"


def raw_rows(wanted: set[str]) -> dict[str, dict]:
    found = {}
    with gzip.open(RAW, "rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            row = json.loads(raw_line)
            uid = str(row["user_id"])
            if uid in wanted and uid not in found:
                tweets = row["user_tweets"]
                ids = [str(tweet.get("id_str", tweet.get("id", ordinal)))
                       for ordinal, tweet in enumerate(tweets)]
                retweets = [ordinal for ordinal, tweet in enumerate(tweets)
                            if (tweet.get("text", "") or "").startswith("RT ")]
                replies = [ordinal for ordinal, tweet in enumerate(tweets)
                           if tweet.get("in_reply_to_status_id") is not None]
                n = len(tweets)
                found[uid] = {
                    "raw_gzip_line": line_number,
                    "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                    "tweet_ids_in_source_order": ids,
                    "sampled_tweet_count": n,
                    "retweet_ordinals": retweets,
                    "reply_ordinals": replies,
                    "retweet_ratio": len(retweets) / n if n else 0.0,
                    "reply_ratio": len(replies) / n if n else 0.0,
                }
            if len(found) == len(wanted):
                break
    if set(found) != wanted:
        raise ValueError(f"Raw Fox8 rows missing {wanted - set(found)}")
    return found


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    selected = {}
    wanted = set()
    for seed in (0, 1, 2):
        row = next(x for x in ledger["runs"]
                   if x["dataset"] == "fox8" and x["seed"] == seed)
        path = Path(row["per_account_group_matrix_path"])
        if sha256_file(path) != row["per_account_group_matrix_sha256"]:
            raise ValueError("Fixed signed ledger changed")
        with np.load(path, allow_pickle=False) as z:
            names = z["source_group_names"].astype(str).tolist()
            j = names.index(GROUP)
            ids = z["account_ids"].astype(str)
            logit = z["logit"].astype(float)
            effect = z["group_logit_effect"][:, j].astype(float)
            pred = z["predicted_bot"].astype(bool)
            threshold = float(z["threshold"])
        flips = (expit(logit - effect) >= threshold) != pred
        chosen = ids[flips].tolist()
        selected[seed] = {uid: (float(logit[i]), float(effect[i]),
                                bool(pred[i]), threshold)
                          for i, uid in enumerate(ids) if flips[i]}
        wanted.update(chosen)
    raw = raw_rows(wanted)
    runs = []
    for seed in (0, 1, 2):
        runtime, labels, split, _ = population(seed)
        source_runtime = Fox8TwentyNeutralAnchorRuntime(ROOT, seed)
        ids = list(selected[seed])
        cards = source_runtime.predict_many(ids)
        cases = []
        for card in cards:
            uid = card["account_id"]
            direct = raw[uid]
            logit, effect, pred, threshold = selected[seed][uid]
            group = next(g for g in card["source_group_effects"]
                         if g["group"] == GROUP)
            if (abs(card["logit"] - logit) > 1e-12 or
                    abs(group["logit_effect"] - effect) > 1e-12 or
                    abs(card["threshold"] - threshold) > 1e-12 or
                    bool(card["prediction"]) != pred or
                    not group["class_changes_without_group"] or
                    abs(group["probability_without_group"] -
                        float(expit(logit - effect))) > 1e-12):
                raise ValueError("Account group response differs from ledger")
            terms = []
            for term in card["concepts"]:
                if not term["concept"].startswith(GROUP + ":"):
                    continue
                fact = term["winning_source_fact"]
                if fact not in {"retweet_ratio", "reply_ratio"}:
                    raise ValueError("Activity mix winner differs")
                source_value = float(np.float32(direct[fact]))
                if abs(float(term["winning_source_value"]) - source_value) > 1e-7:
                    raise ValueError("Tweet fact does not replay from raw tweets")
                terms.append({"concept": term["concept"],
                              "source_fact": fact,
                              "value_from_raw_tweets": direct[fact],
                              "encoded_fact_value": term["winning_source_value"]})
            if len(terms) != 2:
                raise ValueError("Expected high and low activity concepts")
            account_index = source_runtime.index[uid]
            if account_index not in set(map(int, split["test"])):
                raise ValueError("Group-flip account is not in fixed test")
            true_label = int(labels[account_index])
            cases.append({
                "account_id": uid, "label_bot": true_label,
                "predicted_bot": int(pred),
                "prediction_correct": int(pred) == true_label,
                "probability": card["probability"],
                "threshold": threshold,
                "without_activity_mix_probability": group[
                    "probability_without_group"],
                "activity_mix_logit_effect": effect,
                "raw_source": direct,
                "winning_facts": terms,
                "model_sha256": card["model_binding"]["model_sha256"]})
        runs.append({"seed": seed, "cases": cases,
                     "count": len(cases),
                     "correct_before": sum(c["prediction_correct"] for c in cases),
                     "correct_after_removal": sum(
                         not c["prediction_correct"] for c in cases)})
        print({"seed": seed, "raw_verified_flips": len(cases),
               "correct_before": runs[-1]["correct_before"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "Fox8_all_fixed_test_activity_mix_class_flips_raw_tweet_to_frozen_decision",
        "raw_source_sha256": sha256_file(RAW),
        "raw_text_duplicated_in_certificate": False,
        "fact_definition": {
            "retweet_ratio": "tweets with text beginning RT-space / sampled tweets",
            "reply_ratio": "tweets with nonnull in_reply_to_status_id / sampled tweets"},
        "total_cases": sum(row["count"] for row in runs),
        "unique_accounts": len(wanted),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
