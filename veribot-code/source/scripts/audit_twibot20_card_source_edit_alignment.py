"""Check whether card group effects align with actual one-record source edits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates"
CARDS = CERT / "twibot20_split_visible_graph_ab_dev_decisions.json"
TWEETS = CERT / "twibot20_split_visible_graph_tweet_source_edit_dev.json"
EDGES = CERT / "twibot20_split_visible_graph_edge_source_edits_dev.json"
RANDOM_TWEETS = CERT / "twibot20_split_visible_random_tweet_source_edit_dev.json"
OUT = CERT / "twibot20_split_visible_card_source_edit_alignment.json"
EPS = 1e-12


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(name: str, events: list[dict], cards: dict[str, dict], group: str) -> dict:
    per_account: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        uid = event["account_id"]
        if uid not in cards:
            raise ValueError(f"Unknown account in {name}: {uid}")
        card = cards[uid]
        if abs(event["original_logit"] - card["full_logit"]) > EPS:
            raise ValueError(f"Frozen original decision differs in {name}: {uid}")
        per_account[uid].append(event)
    rows = []
    nonzero = aligned = opposed = 0
    for uid, account_events in sorted(per_account.items()):
        card = cards[uid]
        effect = float(card["group_logit_effects"][group])
        deltas = np.asarray([e["logit_change"] for e in account_events])
        if not np.allclose(deltas,
                           [e["edited_logit"] - e["original_logit"]
                            for e in account_events], rtol=0, atol=EPS):
            raise ValueError(f"Recorded source response differs: {uid}")
        if abs(effect) > EPS:
            for delta in deltas:
                if abs(delta) > EPS:
                    nonzero += 1
                    aligned += int(delta * effect < 0)
                    opposed += int(delta * effect > 0)
        rows.append({
            "account_id": uid,
            "source_edit_count": len(account_events),
            "group_effect": effect,
            "mean_abs_single_source_edit_logit_change": float(np.mean(np.abs(deltas))),
            "max_abs_single_source_edit_logit_change": float(np.max(np.abs(deltas))),
            "source_edits_changing_class": sum(e["edited_class"] != e["original_class"]
                                               for e in account_events),
            "group_effect_abs": abs(effect),
        })
    magnitudes = [r["group_effect_abs"] for r in rows]
    mean_edits = [r["mean_abs_single_source_edit_logit_change"] for r in rows]
    max_edits = [r["max_abs_single_source_edit_logit_change"] for r in rows]
    return {
        "scope": name,
        "source_edit_operations": len(events),
        "distinct_accounts": len(rows),
        "nonzero_edit_and_nonzero_group_effect_operations": nonzero,
        "edit_direction_opposes_group_effect_as_removal_intuition": aligned,
        "edit_direction_matches_group_effect": opposed,
        "fraction_opposing_group_effect": aligned / nonzero if nonzero else None,
        "spearman_abs_group_effect_vs_mean_abs_single_edit": float(
            spearmanr(magnitudes, mean_edits).statistic),
        "spearman_abs_group_effect_vs_max_abs_single_edit": float(
            spearmanr(magnitudes, max_edits).statistic),
        "accounts_with_nonzero_group_effect": sum(x > EPS for x in magnitudes),
        "accounts_with_any_nonzero_source_response": sum(x > EPS for x in max_edits),
        "accounts_with_any_source_edit_class_flip": sum(
            r["source_edits_changing_class"] > 0 for r in rows),
        "rows": rows,
    }


def main() -> None:
    cards_artifact = read(CARDS)
    tweet_artifact = read(TWEETS)
    edge_artifact = read(EDGES)
    random_artifact = read(RANDOM_TWEETS)
    cards = {row["account_id"]: row for row in cards_artifact["decision_cards"]}
    if len(cards) != cards_artifact["dev_n"]:
        raise ValueError("Dev card population differs")
    from hashlib import sha256
    cards_sha = sha256(CARDS.read_bytes()).hexdigest()
    if (tweet_artifact["decision_cards_sha256"] != cards_sha
            or edge_artifact["decision_cards_sha256"] != cards_sha
            or random_artifact["cards_sha256"] != cards_sha):
        raise ValueError("Source edits refer to different frozen decision cards")
    for artifact in (tweet_artifact, edge_artifact, random_artifact):
        if (artifact["facts_sha256"] != cards_artifact["fact_cache_sha256"]
                or artifact["reference_sha256"] != cards_artifact["reference_artifact_sha256"]
                or artifact["model_sha256"] != cards_artifact["frozen_model_sha256"]
                or artifact["source_archive_sha256"] != cards_artifact["source_archive_sha256"]):
            raise ValueError("Source edit uses another frozen model or raw source")
    first = [r for r in tweet_artifact["edits"]["first"]["rows"]
             if r["sampled_tweets_before"] > 0]
    last = [r for r in tweet_artifact["edits"]["last"]["rows"]
            if r["sampled_tweets_before"] > 0]
    paired_tweets = list({(r["account_id"], r["source_locator_removed"]): r
                          for r in first + last}.values())
    random_tweets = [r for r in random_artifact["rows"]
                     if r["sampled_tweets_before"] > 0]
    edges = edge_artifact["edits"]
    dev_edges = [e for e in edges if e["declaring_part"] == "dev"]
    support_edges = [e for e in edges if e["declaring_part"] == "support"]
    def one_hash_selected_edge_per_account(events: list[dict]) -> list[dict]:
        selected = {}
        for event in events:
            uid = event["account_id"]
            key = hashlib.sha256(
                f"twibot20-edge-control-2026-11-04|{uid}|{event['source_declaration']}".encode()
            ).hexdigest()
            if uid not in selected or key < selected[uid][0]:
                selected[uid] = (key, event)
        return [value[1] for _, value in sorted(selected.items())]
    report = {
        "scope": "descriptive source-edit alignment of already frozen split-visible dev decisions",
        "cards_sha256": cards_sha,
        "source_edit_artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (TWEETS, EDGES, RANDOM_TWEETS)},
        "tweet_first": summarize("remove first sampled tweet", first, cards,
                                 "sampled_tweet_stats"),
        "tweet_last": summarize("remove last sampled tweet", last, cards,
                                "sampled_tweet_stats"),
        "tweet_both": summarize("remove first or last sampled tweet", paired_tweets,
                                cards, "sampled_tweet_stats"),
        "tweet_random": summarize("remove one fixed hash-selected sampled tweet",
                                  random_tweets, cards, "sampled_tweet_stats"),
        "edge_dev": summarize("remove one dev declaration",
                              dev_edges,
                              cards, "native_follow_graph"),
        "edge_dev_random_one_per_account": summarize(
            "remove one fixed hash-selected eligible dev declaration per account",
            one_hash_selected_edge_per_account(dev_edges),
            cards, "native_follow_graph"),
        "edge_support": summarize("remove one support declaration",
                                  support_edges,
                                  cards, "native_follow_graph"),
        "interpretation": "A group's all-zero-input logit effect and the response to deleting one raw tweet or edge are different interventions. Sign agreement is a descriptive alignment check, not a requirement for local faithfulness. Operations repeat accounts and the deleted records are not known errors. No correction utility or semantic truth follows.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "rows"}
                      if isinstance(value, dict) else value
                      for key, value in report.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
