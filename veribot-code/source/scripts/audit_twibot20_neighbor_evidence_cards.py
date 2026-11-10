"""Audit every test evidence-card locator against the official TwiBot records."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ijson
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from train_v3_twibot20 import sha256


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    selected = root / "output/v3_twibot20_neighbor_unigram/selected"
    report = json.loads((selected / "report.json").read_text(encoding="utf-8"))
    cards_path = selected / "evidence_cards.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text(encoding="utf-8").splitlines()]
    if len(cards) != report["test_evidence_cards"] or sha256(cards_path) != report[
            "test_evidence_cards_sha256"]:
        raise ValueError("test evidence cards differ from locked report")
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    if raw_manifest["archive_sha256"] != report["source_archive_sha256"]:
        raise ValueError("official archive differs")
    base_path = root / "data/processed/twibot20_v3/facts.npz"
    neighbor_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_facts.npz"
    incidence_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_incidence.npz"
    with np.load(base_path, allow_pickle=False) as base:
        test_indices = np.flatnonzero(base["split"].astype(str) == "test")
        ids = base["ids"].astype(str)[test_indices]
        positions = base["positions"].astype(int)[test_indices]
        facts = base["values"].astype(float)[test_indices]
        fact_names = base["names"].astype(str).tolist()
    with np.load(neighbor_path, allow_pickle=False) as neighbor:
        values = neighbor["values"].astype(float)[test_indices]
        supports = neighbor["supports"].astype(int)[test_indices]
        neighbor_names = neighbor["names"].astype(str).tolist()
    with np.load(incidence_path, allow_pickle=False) as source:
        source_ids = source["ids"].astype(str)
        source_parts = source["parts"].astype(int)
        source_positions = source["positions"].astype(int)
        source_counts = source["tweet_counts"].astype(int)
        source_incidence = source["incidence"].astype(np.uint8)
        active_words = source["vocabulary"].astype(str)
    source_lookup = {account_id: index for index, account_id in enumerate(source_ids)}
    neighbor_lookup = {name: index for index, name in enumerate(neighbor_names)}
    with np.load(root / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz",
                 allow_pickle=False) as model:
        vocabulary = model["vocabulary"].astype(str).tolist()
    analyzer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", vocabulary=vocabulary).build_analyzer()
    with np.load(selected / "test_predictions.npz", allow_pickle=False) as pred:
        if not np.array_equal(pred["ids"].astype(str), ids):
            raise ValueError("test prediction IDs differ")
        probability = pred["probability"].astype(float)
        raw_score = pred["raw_score"].astype(float)
        reconstructed = pred["contribution_sum"].astype(float)
    counts = {"own_fact": 0, "own_unigram": 0, "neighbor_unigram": 0}
    checked_items = 0
    with zipfile.ZipFile(raw_manifest["archive_path"]) as zipped:
        with zipped.open("Twibot-20/test.json") as stream:
            for row_index, row in enumerate(ijson.items(stream, "item")):
                card = cards[row_index]
                if (row_index != positions[row_index] or str(row["ID"]) != ids[row_index]
                        or card["account_id"] != ids[row_index]
                        or card["prediction"] != int(probability[row_index] >=
                                                      report["threshold"])
                        or abs(card["probability"] - probability[row_index]) > 1e-12
                        or abs(card["raw_logit"] - raw_score[row_index]) > 1e-12
                        or abs(card["contribution_sum"] - reconstructed[row_index]) > 1e-12):
                    raise ValueError("test evidence card prediction or owner locator differs")
                if len(card["top_absolute_contributions"]) != 8:
                    raise ValueError("evidence card has the wrong top contribution count")
                tweets = row.get("tweet") or []
                for item in card["top_absolute_contributions"]:
                    if (item["source_member"] != "Twibot-20/test.json"
                            or item["source_owner_row"] != row_index
                            or item["source_owner_id"] != ids[row_index]):
                        raise ValueError("evidence card owner source pointer differs")
                    feature = item["feature"]
                    if feature.startswith(("own_upper_", "own_lower_")):
                        name = feature.split("_", 2)[2]
                        index = fact_names.index(name)
                        expected = facts[row_index, index]
                        observed = item["raw_fact_value"]
                        if (np.isfinite(expected) and
                                (observed is None or abs(observed - expected) > 1e-12)):
                            raise ValueError("own fact value differs")
                        if not np.isfinite(expected) and observed is not None:
                            raise ValueError("own missing fact differs")
                        counts["own_fact"] += 1
                    elif feature.startswith("own_unigram_"):
                        word = feature[len("own_unigram_"):]
                        expected = [i for i, tweet in enumerate(tweets)
                                    if word in set(analyzer(str(tweet or "")))]
                        if item["raw_tweet_indices_containing_word"] != expected:
                            raise ValueError("own raw tweet token indices differ")
                        counts["own_unigram"] += 1
                    elif feature.startswith(("neighbor_upper_", "neighbor_lower_")):
                        name = (feature[len("neighbor_upper_"):]
                                if feature.startswith("neighbor_upper_") else
                                feature[len("neighbor_lower_"):])
                        fact_index = neighbor_lookup[name]
                        direction_index, word_index = divmod(fact_index, 109)
                        direction = ("following", "follower")[direction_index]
                        word = active_words[word_index]
                        raw_ids = set(map(str, (row.get("neighbor") or {})
                                          .get(direction) or ()))
                        matched = sorted(source_lookup[uid] for uid in raw_ids
                                         if uid in source_lookup and
                                         source_counts[source_lookup[uid]] > 0)
                        positives = [index for index in matched
                                     if source_incidence[index, word_index]]
                        expected_share = len(positives) / len(matched) if matched else None
                        source_examples = [
                            {"account_id": source_ids[index],
                             "member": "Twibot-20/" +
                             ("train", "dev", "test", "support")[source_parts[index]] +
                             ".json", "row": int(source_positions[index])}
                            for index in positives[:5]]
                        if (item["native_relation"] != direction or item["raw_word"] != word
                                or item["neighbor_support"] != len(matched)
                                or supports[row_index, direction_index] != len(matched)
                                or item["neighbor_word_hit_count"] != len(positives)
                                or item["sample_source_neighbors_with_word"] != source_examples):
                            raise ValueError("neighbor word source locator or count differs")
                        if expected_share is None:
                            if item["raw_neighbor_share"] is not None or np.isfinite(
                                    values[row_index, fact_index]):
                                raise ValueError("missing neighbor word share differs")
                        elif (item["raw_neighbor_share"] is None or
                              abs(item["raw_neighbor_share"] - expected_share) > 1e-7 or
                              abs(values[row_index, fact_index] - expected_share) > 1e-7):
                            raise ValueError("neighbor word share differs")
                        counts["neighbor_unigram"] += 1
                    else:
                        raise ValueError("unknown evidence card feature")
                    checked_items += 1
    if len(cards) != len(ids):
        raise ValueError("evidence card count differs from official test")
    certificate = {
        "dataset": "twibot20",
        "status": "all_selected_test_evidence_card_source_locators_verified",
        "accounts_checked": len(cards), "top_items_checked": checked_items,
        "item_types": counts,
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "evidence_cards_sha256": sha256(cards_path),
        "test_predictions_sha256": sha256(selected / "test_predictions.npz"),
    }
    destination = root / "research/certificates/twibot20_neighbor_lightgbm_all_test_card_audit.json"
    destination.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2), flush=True)


if __name__ == "__main__":
    main()
