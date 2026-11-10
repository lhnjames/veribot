"""Audit every compact MGTAB card against frozen evidence and source hashes."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_profile_schema import PROFILE_SCHEMA


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    manifest_path = folder / "evidence_cards_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay_path = root / "research/certificates/mgtab_targeted_20_concept_ab_group_tree_replay.json"
    if manifest["independent_replay_sha256"] != sha256_file(replay_path):
        raise ValueError("Independent source replay changed")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    if any(manifest["source_hashes"][key] != sha256_file(data_dir / filename)
           for key, filename in (("features", "features.pt"),
                                 ("edges", "edge_index.pt"),
                                 ("edge_types", "edge_type.pt"))):
        raise ValueError("Published source hashes changed")
    rows = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        card_path = run_dir / "test_evidence_cards.jsonl.gz"
        expected = manifest["runs"][seed]
        if expected["sha256"] != sha256_file(card_path):
            raise ValueError("Evidence card artifact changed")
        with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as arr:
            ids = arr["ids"].astype(str).tolist()
            fact_names = arr["retained_fact_names"].astype(str).tolist()
            raw = arr["retained_raw_values"].copy()
            concept_names = arr["concept_names"].astype(str).tolist()
            concept_values = arr["concept_scores"].copy()
            winners = arr["concept_winners"].copy()
            probability = arr["probability"].copy()
            logit = arr["logit"].copy()
            shap = arr["shap"].copy()
            group_shap = arr["group_shap"].copy()
            group_delta = arr["exact_group_delta"].copy()
            group_mask = arr["group_mask_probability"].copy()
            all_mask = arr["all_mask_probability"].copy()
            labels = arr["label"].copy()
        group_names = [concept_names[2*k].split(":", 1)[0] for k in range(10)]
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            cards = [json.loads(line) for line in stream]
        if len(cards) != len(ids):
            raise ValueError("Account card count differs")
        located = 0
        for j, card in enumerate(cards):
            threshold = card["threshold"]
            if (card["account_id"] != ids[j]
                    or card["label"] != int(labels[j])
                    or card["prediction"] != int(probability[j] >= threshold)
                    or abs(card["probability"] - probability[j]) > 1e-14
                    or abs(card["logit"] - logit[j]) > 1e-13
                    or abs(card["zero_evidence_probability"] - all_mask[j]) > 1e-14
                    or len(card["evidence"]) != 20):
                raise ValueError("Saved account prediction card differs")
            for k, evidence in enumerate(card["evidence"]):
                winner = int(winners[j, k])
                fact = fact_names[winner] if winner >= 0 else None
                value = (float(raw[j, winner]) if winner >= 0 and
                         np.isfinite(raw[j, winner]) else None)
                if (evidence["concept"] != concept_names[k]
                        or abs(evidence["value"] - concept_values[j, k]) > 1e-14
                        or evidence["winning_fact"] != fact
                        or evidence["winning_fact_value"] != value
                        or abs(evidence["model_logit_attribution"] -
                               float(shap[j, k] + shap[j, k+20])) > 1e-13):
                    raise ValueError("Fact, concept, or attribution card differs")
                locator = evidence["source_locator"]
                if fact is not None:
                    located += 1
                    if locator is None:
                        raise ValueError("Available concept lacks source locator")
                    if fact.startswith("published_profile_coordinate_"):
                        column = int(fact[-2:])
                        if (locator["row"] != int(ids[j]) or locator["column"] != column
                                or locator["paper_table_a5_alias"] != PROFILE_SCHEMA[column][0]):
                            raise ValueError("Profile tensor source locator differs")
                    elif fact.startswith("published_tweet_projection_"):
                        if (locator["row"] != int(ids[j]) or
                                locator["output_coordinate"] != int(fact[-3:])):
                            raise ValueError("Published content tensor locator differs")
                    elif locator["account_row"] != int(ids[j]):
                        raise ValueError("Typed relation source locator differs")
                elif locator is not None:
                    raise ValueError("Missing concept invents a source locator")
            if set(card["group_effects"]) != set(group_names):
                raise ValueError("Card source group names differ")
            for k, name in enumerate(group_names):
                effect = card["group_effects"][name]
                if (abs(effect["model_logit_attribution"] - group_shap[j, k]) > 1e-13
                        or abs(effect["exact_zero_mask_logit_difference"] -
                               group_delta[j, k]) > 1e-13
                        or abs(effect["probability_without_group"] -
                               group_mask[j, k]) > 1e-14
                        or effect["changes_class_without_group"] != bool(
                            (group_mask[j, k] >= threshold) !=
                            (probability[j] >= threshold))):
                    raise ValueError("Card group decision effect differs")
            if (abs(sum(x["model_logit_attribution"] for x in
                        card["group_effects"].values()) + card["intercept_logit"] -
                    card["logit"]) > 1e-9):
                raise ValueError("Card group score does not reconstruct")
        if (len(cards)*20 != expected["concept_slots"]
                or len(cards)*10 != expected["group_mask_decisions"]
                or located != expected["source_locators"]):
            raise ValueError("Evidence card manifest totals differ")
        rows.append({"seed": seed, "account_cards_checked": len(cards),
                     "concept_slots_checked": len(cards)*20,
                     "source_locators_checked": located,
                     "group_mask_decisions_checked": len(cards)*10})
    certificate = {"scope": "MGTAB_all_twenty_concept_account_cards_verified",
                   "manifest_sha256": sha256_file(manifest_path),
                   "runs": rows,
                   "status": "all_account_cards_match_frozen_evidence_and_source_level"}
    path = root / "research/certificates/mgtab_targeted_20_concept_cards.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
