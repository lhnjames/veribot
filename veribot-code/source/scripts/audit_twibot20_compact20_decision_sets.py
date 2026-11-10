"""Enumerate decision-preserving source-group subsets under two frozen references."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


GROUPS = ("profile_counts", "profile_presentation", "sampled_tweet_stats",
          "native_follow_graph")
MASKS = tuple(range(1 << len(GROUPS)))


def selected(mask: int) -> list[str]:
    return [group for bit, group in enumerate(GROUPS) if mask & (1 << bit)]


def classify(margin: float) -> int:
    return int(margin >= 0)


def audit_reference(base: float, effects: dict[str, float], full_class: int,
                    full_margin: float) -> dict:
    margin_by_mask = {mask: base + sum(effects[group] for group in selected(mask))
                      for mask in MASKS}
    if (abs(margin_by_mask[15] - full_margin) > 1e-10
            or classify(margin_by_mask[15]) != full_class):
        raise ValueError("The full subset does not reproduce the frozen decision")
    preserving = [mask for mask in MASKS if classify(margin_by_mask[mask]) == full_class]
    minimal = [mask for mask in preserving if not any(
        other != mask and other & mask == other for other in preserving)]
    leave_one_out = {group: classify(margin_by_mask[15 ^ (1 << bit)]) != full_class
                     for bit, group in enumerate(GROUPS)}
    singleton = {group: classify(margin_by_mask[1 << bit]) == full_class
                 for bit, group in enumerate(GROUPS)}
    return {
        "base_margin": base,
        "base_class": classify(base),
        "preserving_subset_count": len(preserving),
        "minimal_preserving_groups": [selected(mask) for mask in minimal],
        "minimal_preserving_margins": [margin_by_mask[mask] for mask in minimal],
        "full_context_leave_one_group_out_changes_class": leave_one_out,
        "single_group_from_base_preserves_class": singleton,
        "all_subset_margins": [{"included_groups": selected(mask),
                                "margin": margin_by_mask[mask],
                                "class": classify(margin_by_mask[mask])}
                               for mask in MASKS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--reference-sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    anchor = json.loads(args.reference_sensitivity.read_text(encoding="utf-8"))
    if (anchor["decision_cards_sha256"] != sha256(args.cards)
            or anchor["compact_sha256"] != cards["compact_certificate_sha256"]
            or len(cards["decision_cards"]) != len(anchor["rows"])):
        raise ValueError("Frozen decision and anchor certificates differ")
    rows = []
    summaries = {name: {"base_already_predicts_full_class": 0,
                        "unique_minimal_set": 0,
                        "multiple_minimal_sets": 0,
                        "full_context_necessary": Counter(),
                        "singleton_sufficient_from_base": Counter(),
                        "minimal_set_size": Counter(),
                        "predicted_bot_with_empty_minimal_set": 0,
                        "predicted_human_with_empty_minimal_set": 0}
                 for name in ("zero_encoded", "observed_train_account")}
    examples = cards["exemplar_ids_first_in_official_order"]
    example_ids = set(examples.values())
    example_results = {}
    for card, reference_row in zip(cards["decision_cards"], anchor["rows"]):
        if card["account_id"] != reference_row["account_id"]:
            raise ValueError("Decision-card order differs")
        decision = int(card["prediction"])
        full_margin = float(card["full_margin"])
        audits = {
            "zero_encoded": audit_reference(
                float(card["baseline_margin"]), card["group_logit_effects"],
                decision, full_margin),
            "observed_train_account": audit_reference(
                float(anchor["observed_reference_margin"]),
                reference_row["observed_reference_group_differences"],
                decision, full_margin),
        }
        for name, detail in audits.items():
            summary = summaries[name]
            minimum = detail["minimal_preserving_groups"]
            summary["base_already_predicts_full_class"] += int(
                detail["base_class"] == decision)
            summary["unique_minimal_set"] += int(len(minimum) == 1)
            summary["multiple_minimal_sets"] += int(len(minimum) > 1)
            summary["minimal_set_size"].update(len(group_set) for group_set in minimum)
            summary["predicted_bot_with_empty_minimal_set"] += int(
                decision == 1 and [] in minimum)
            summary["predicted_human_with_empty_minimal_set"] += int(
                decision == 0 and [] in minimum)
            for group, changed in detail["full_context_leave_one_group_out_changes_class"].items():
                summary["full_context_necessary"][group] += int(changed)
            for group, sufficient in detail["single_group_from_base_preserves_class"].items():
                summary["singleton_sufficient_from_base"][group] += int(sufficient)
        if card["account_id"] in example_ids:
            example_results[card["account_id"]] = {
                "full_margin": full_margin, "prediction": decision,
                "correct": bool(card["prediction_correct"]),
                "zero_encoded": audits["zero_encoded"],
                "observed_train_account": audits["observed_train_account"],
            }
        rows.append({
            "account_id": card["account_id"],
            "prediction": decision,
            "prediction_correct": bool(card["prediction_correct"]),
            "full_margin": full_margin,
            "zero_encoded": audits["zero_encoded"],
            "observed_train_account": audits["observed_train_account"],
        })
    result = {
        "scope": "Frozen compact 20-fact TwiBot-20 model, every official test account",
        "operation": "Enumerate all 16 combinations of 4 encoded source groups against zero or one label-blind observed train anchor; no source edits",
        "cards_sha256": sha256(args.cards),
        "reference_sensitivity_sha256": sha256(args.reference_sensitivity),
        "groups": GROUPS,
        "accounts": len(rows),
        "subsets_per_account_per_reference": len(MASKS),
        "anchor_account_id": anchor["anchor_account_id"],
        "summaries": {name: {key: dict(value) if isinstance(value, Counter) else value
                             for key, value in summary.items()}
                      for name, summary in summaries.items()},
        "examples": example_results,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "examples")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
