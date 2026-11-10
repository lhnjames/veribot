"""A prediction-blind hash-position sampled-tweet removal control on frozen dev accounts."""

from __future__ import annotations

import hashlib
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
OUT = ROOT / "research/certificates/twibot20_split_visible_random_tweet_source_edit_dev.json"
SEED = "twibot20-source-position-control-2026-11-04"


def choose(uid: str, n: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}|{uid}".encode()).digest()[:8],
                          "big") % n


def main() -> None:
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    compact = json.loads(COMPACT.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (sha256(ARCHIVE) != cards["source_archive_sha256"]
            or sha256(FACTS) != cards["fact_cache_sha256"]
            or sha256(REFERENCE) != cards["reference_artifact_sha256"]
            or sha256(MODEL) != cards["frozen_model_sha256"]
            or sha256(SCREEN) != cards["group_screen_sha256"]):
        raise ValueError("Frozen source, reference, model or cards differ")
    with np.load(FACTS, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        positions = saved["positions"].astype(int)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    chosen = next(row for row in screen["rows"] if row["ab_seed"] == cards["a_seed"])
    signs = np.r_[[chosen["tail_directions_by_fact"][name]["upper"] for name in names],
                  [chosen["tail_directions_by_fact"][name]["lower"] for name in names]]
    original = values[dev].copy()
    edited = original.copy()
    source_slots = []
    sample_counts = []
    with zipfile.ZipFile(ARCHIVE) as zipped, zipped.open("Twibot-20/dev.json") as stream:
        for k, row in enumerate(ijson.items(stream, "item")):
            if (k >= len(dev) or str(row["ID"]) != ids[dev[k]]
                    or positions[dev[k]] != k):
                raise ValueError("Official dev source row differs")
            direct = direct_facts(row)
            if not np.allclose(direct, original[k, :len(direct)],
                               atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError("Official direct facts differ")
            tweets = list(row.get("tweet") or [])
            sample_counts.append(len(tweets))
            if not tweets:
                source_slots.append(None)
                continue
            slot = choose(str(row["ID"]), len(tweets))
            changed = dict(row)
            changed["tweet"] = tweets[:slot] + tweets[slot + 1:]
            edited[k, :len(direct)] = direct_facts(changed)
            source_slots.append(slot)
    if len(source_slots) != len(dev):
        raise ValueError("Dev source population differs")
    reference = LabelBlindReference.load(REFERENCE)
    keep = [all_names.index(name) for name in names]
    def encode(source: np.ndarray) -> np.ndarray:
        return design(evidence_subset(
            reference.transform(source, contexts=contexts[dev]), keep, names,
            source[:, all_names.index("tweet_sample_count")], True), signs)
    x0 = encode(original)
    x1 = encode(edited)
    tweet_cols = source_groups(names)[list(GROUPS).index("sampled_tweet_stats")]
    other_cols = [j for j in range(x0.shape[1]) if j not in tweet_cols]
    non_tweet_error = float(np.max(np.abs(x0[:, other_cols] - x1[:, other_cols])))
    if non_tweet_error > 1e-12:
        raise ValueError("Random tweet removal changed non-tweet model inputs")
    model = Booster(model_file=str(MODEL))
    y0 = np.asarray(model.predict(x0, raw_score=True))
    y1 = np.asarray(model.predict(x1, raw_score=True))
    card_rows = cards["decision_cards"]
    original_error = float(np.max(np.abs(
        y0 - np.asarray([card["full_logit"] for card in card_rows]))))
    if original_error > 1e-12:
        raise ValueError("Frozen original decision differs")
    only_tweet = x0.copy()
    only_tweet[:, tweet_cols] = x1[:, tweet_cols]
    response_error = float(np.max(np.abs(
        y1 - np.asarray(model.predict(only_tweet, raw_score=True)))))
    if response_error > 1e-12:
        raise ValueError("Random source response does not replay tweet group")
    threshold = float(card_rows[0]["threshold"])
    rows = []
    for k, card in enumerate(card_rows):
        slot = source_slots[k]
        rows.append({
            "account_id": str(ids[dev[k]]),
            "sampled_tweets_before": sample_counts[k],
            "source_locator_removed": (f"Twibot-20/dev.json[{k}].tweet[{slot}]"
                                       if slot is not None else None),
            "position": slot,
            "original_logit": float(y0[k]),
            "edited_logit": float(y1[k]),
            "logit_change": float(y1[k] - y0[k]),
            "original_class": int(expit(y0[k]) >= threshold),
            "edited_class": int(expit(y1[k]) >= threshold),
        })
    report = {
        "scope": "fixed seed, one prediction-blind hash-selected sampled tweet per dev account",
        "selection_seed": SEED,
        "source_archive_sha256": sha256(ARCHIVE),
        "facts_sha256": sha256(FACTS),
        "reference_sha256": sha256(REFERENCE),
        "cards_sha256": sha256(CARDS),
        "model_sha256": sha256(MODEL),
        "dev_accounts": len(rows),
        "eligible_accounts": sum(n > 0 for n in sample_counts),
        "selected_only": sum(n == 1 for n in sample_counts),
        "selected_first_of_multiple": sum(slot == 0 and n > 1
                                           for slot, n in zip(source_slots, sample_counts)
                                           if slot is not None),
        "selected_last_of_multiple": sum(slot == n-1 and n > 1
                                          for slot, n in zip(source_slots, sample_counts)
                                          if slot is not None),
        "selected_internal": sum(0 < slot < n-1 for slot, n in zip(source_slots, sample_counts)
                                 if slot is not None),
        "nonzero_logit_changes": sum(abs(r["logit_change"]) > 1e-12 for r in rows),
        "class_changes": sum(r["original_class"] != r["edited_class"] for r in rows),
        "max_non_tweet_input_change": non_tweet_error,
        "max_original_card_replay_error": original_error,
        "max_tweet_group_response_error": response_error,
        "rows": rows,
        "interpretation": "Source-deletion sensitivity for a prediction-blind hash-selected source position. The raw tweet is not known erroneous. The card has a group effect but does not rank individual tweets.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
