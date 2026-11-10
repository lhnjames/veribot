"""Render complete, dimension-organized evidence cards from verified probes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from verifiable_bot.fox8_evidence_dimensions import FACT_DIMENSIONS, GRAPH_DIMENSIONS
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS


def build_card(account: dict, verification: dict, threshold: float) -> dict:
    views = account["target_selected_view_facts"]
    if set(views) != {"human", "bot"}:
        raise ValueError("both class views are required")
    for view in views.values():
        if set(view) != set(ALL_FACTS):
            raise ValueError("card facts differ from model inputs")
    singles = {item["source"]: item["delta_logit"]
               for item in account["search"]["single_effects"]}
    baseline_margin = account["baseline"]["margin"]
    groups = []
    for item in account["search"]["solutions"]:
        if any(source not in singles for source in item["sources"]):
            raise ValueError("group member lacks a measured single-source effect")
        joint_delta = item["after_removal_logit"] - baseline_margin
        single_sum = sum(singles[source] for source in item["sources"])
        groups.append({
            "source_ids": item["sources"],
            "source_count": len(item["sources"]),
            "after_removal_margin": item["after_removal_logit"],
            "joint_margin_change": joint_delta,
            "sum_of_single_margin_changes": single_sum,
            "nonadditive_margin_residual": joint_delta - single_sum,
            "locally_minimal_checked": item["locally_minimal_checked"],
        })
    return {
        "account_id": account["account_id"],
        "predicted_class": "Bot" if account["predicted_bot"] else "Human",
        "evaluation_label": account["label_for_evaluation_only"],
        "decision_threshold_from_dev": threshold,
        "class_scores": account["baseline"],
        "dimension_facts": {
            dimension: {name: {"human_view": views["human"][name],
                               "bot_view": views["bot"][name]}
                        for name in names}
            for dimension, names in FACT_DIMENSIONS.items()
        },
        "graph_relation_dimensions": list(GRAPH_DIMENSIONS),
        "class_source_closures": account["class_source_closures"],
        "full_path_class_changing_groups": groups,
        "candidate_source_count": account["candidate_source_count"],
        "source_search_evaluations": account["search"]["unique_evaluations"],
        "verification": verification,
        "scope": "model decision dependence under named source-channel removal",
    }


def render_card(card: dict) -> str:
    lines = [f"# Fox8 evidence card: {card['account_id']}", "",
             f"Decision: **{card['predicted_class']}**; dev-set threshold: "
             f"`{card['decision_threshold_from_dev']:.9f}`.", "",
             "The two class hypotheses are scored from their own selected raw-source "
             "views before the final class is known. A fact's numerical value is a "
             "measurement, not a standalone Bot/Human label.", "",
             "## Class scores and source closure", "",
             "| Quantity | Human hypothesis | Bot hypothesis |",
             "|---|---:|---:|",
             f"| RelNet class score | {card['class_scores']['human_score']:.6f} | "
             f"{card['class_scores']['bot_score']:.6f} |",
             f"| Selected profile fields in two-hop closure | "
             f"{card['class_source_closures']['human']['source_counts']['profile_field']} | "
             f"{card['class_source_closures']['bot']['source_counts']['profile_field']} |",
             f"| Selected tweet observations in two-hop closure | "
             f"{card['class_source_closures']['human']['source_counts']['tweet']} | "
             f"{card['class_source_closures']['bot']['source_counts']['tweet']} |",
             f"| Selected native relation links in two-hop closure | "
             f"{card['class_source_closures']['human']['source_counts']['native_relation_event']} | "
             f"{card['class_source_closures']['bot']['source_counts']['native_relation_event']} |",
             "", "## All named node facts", ""]
    for dimension, facts in card["dimension_facts"].items():
        lines += [f"### {dimension.replace('_', ' ').title()}", "",
                  "| Raw-derived fact | Human view | Bot view |",
                  "|---|---:|---:|"]
        lines += [f"| `{name}` | {values['human_view']:.6g} | "
                  f"{values['bot_view']:.6g} |" for name, values in facts.items()]
        lines.append("")
    lines += ["## Native graph dimensions", "",
              "The selected class views separately rebuild directed reply and "
              "retweet relations, with reverse edges for RelNet message passing. "
              "Raw relation locators, profile-value hashes, and tweet-record "
              "hashes are in the JSON card.", "",
              "## Full-path source interventions", "",
              f"Candidate pool: {card['candidate_source_count']} source channels "
              f"from this account and its incident relation links; "
              f"{card['source_search_evaluations']} distinct model evaluations. "
              "The search is budgeted and the candidate pool excludes "
              "neighbour account fields and posts.", "",
              "| Removed source-channel IDs | Margin after removal | Joint change | Nonadditive residual | Local minimality checked |",
              "|---|---:|---:|---:|---|"]
    for group in card["full_path_class_changing_groups"]:
        labels = ", ".join(f"`{item}`" for item in group["source_ids"])
        lines.append(f"| {labels} | {group['after_removal_margin']:.6f} | "
                     f"{group['joint_margin_change']:.6f} | "
                     f"{group['nonadditive_margin_residual']:.6f} | "
                     f"{'yes' if group['locally_minimal_checked'] else 'no'} |")
    lines += ["", "## Verification boundary", "",
              "The independent verifier checked raw record positions and values, "
              "the two-hop RelNet source closure, class scores, full-path class "
              "changes, and claimed local minimality. It did not establish that "
              "these source channels prove the account's real-world identity; "
              "the HGT selector context still needs its own exhaustive "
              "raw-source closure.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT / "output/two_stage_fox8_evidence/pilot_replay")
    args = parser.parse_args()
    probe = json.loads((args.run / "full_path_source_group_probe.json").read_text(
        encoding="utf-8"))
    verification = json.loads((args.run /
        "full_path_source_group_probe_verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "verified_raw_sources_two_hop_scores_and_full_path_groups":
        raise ValueError("independent source verification is missing")
    if verification["probe_sha256"] != sha256(
            (args.run / "full_path_source_group_probe.json").read_bytes()).hexdigest():
        raise ValueError("verified probe differs from card source")
    cards = [build_card(account, verification,
                        probe["threshold_dev_only"]) for account in probe["accounts"]]
    output = args.run / "evidence_cards.json"
    output.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    for card in cards:
        (args.run / f"evidence_card_{card['account_id']}.md").write_text(
            render_card(card), encoding="utf-8")
    print(json.dumps({"cards": len(cards), "dimensions": len(FACT_DIMENSIONS) + 1,
                      "facts_per_class": len(ALL_FACTS), "output": str(output)}))


if __name__ == "__main__":
    main()
