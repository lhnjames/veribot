"""Summarize context-dependent evidence-group rankings from measured dev queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def kendall_tau_b(left: dict[str, float], right: dict[str, float],
                  tolerance: float = 1e-6) -> float | None:
    names = sorted(left)
    if set(names) != set(right):
        raise ValueError("Rankings must cover the same groups")
    concordant = discordant = tied_left = tied_right = 0
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            dx = left[first] - left[second]
            dy = right[first] - right[second]
            x_tied = abs(dx) <= tolerance
            y_tied = abs(dy) <= tolerance
            if x_tied and y_tied:
                continue
            if x_tied:
                tied_left += 1
            elif y_tied:
                tied_right += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + tied_left) *
                            (concordant + discordant + tied_right))
    return ((concordant - discordant) / denominator
            if denominator else None)


def summarize_case(case: dict, interaction_cutoff: float) -> dict:
    audit = case["coalition_audit"]
    effects = audit["ranked_group_effects"]
    alone = {row["group"]: row["signed_gain_over_empty"]
             for row in effects}
    full = {row["group"]: row["signed_loss_when_removed_from_full"]
            for row in effects}
    rank_alone = audit["ranked_by_singleton_gain"]
    rank_full = audit["ranked_by_full_context_removal_loss"]
    if set(rank_alone) != set(alone) or set(rank_full) != set(full):
        raise ValueError("Ranking/group effect disagreement")
    pairs = audit["pair_interactions"]
    strong_positive = [row for row in pairs
                       if row["signed_pair_interaction"] > interaction_cutoff]
    strong_negative = [row for row in pairs
                       if row["signed_pair_interaction"] < -interaction_cutoff]
    positive = max(pairs, key=lambda row: row["signed_pair_interaction"],
                   default=None)
    negative = min(pairs, key=lambda row: row["signed_pair_interaction"],
                   default=None)
    top = min(3, len(effects))
    return {
        "account_id": case["account_id"],
        "predicted_bot": case["predicted_bot"],
        "group_count": len(effects),
        "oracle_query_count": audit["oracle_query_count"],
        "query_budget": audit["query_budget"],
        "ranking_alone": rank_alone,
        "ranking_full_context_removal": rank_full,
        "top3_overlap_count": len(set(rank_alone[:top]) & set(rank_full[:top])),
        "kendall_tau_b": kendall_tau_b(alone, full),
        "group_effects": effects,
        "negative_singleton_gain_count": sum(
            value < -1e-6 for value in alone.values()),
        "negative_full_context_loss_count": sum(
            value < -1e-6 for value in full.values()),
        "observed_pair_count": len(pairs),
        "observed_positive_interactions_over_cutoff": len(strong_positive),
        "observed_negative_interactions_below_cutoff": len(strong_negative),
        "largest_positive_observed_pair": positive,
        "largest_negative_observed_pair": negative,
        "observed_nondominated_sufficient_group_sets":
            audit["observed_nondominated_sufficient_group_sets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--interaction-cutoff", type=float, default=0.05)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.interaction_cutoff < 0:
        raise ValueError("Interaction cutoff must be nonnegative")
    rows = []
    inputs = []
    for path in args.reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        if report["split"] != "dev":
            raise ValueError("Only development-set queries belong in this audit")
        inputs.append({"path": str(path),
                       "sha256": hashlib.sha256(raw).hexdigest()})
        for case in report["cases"]:
            rows.append({"report": str(path),
                         **summarize_case(case, args.interaction_cutoff)})
    def aggregate(items: list[dict]) -> dict:
        defined = [item["kendall_tau_b"] for item in items
                   if item["kendall_tau_b"] is not None]
        return {
            "account_count": len(items),
            "median_kendall_tau_b": median(defined) if defined else None,
            "top3_zero_overlap_count": sum(
                item["top3_overlap_count"] == 0 for item in items),
            "negative_singleton_gain_account_count": sum(
                item["negative_singleton_gain_count"] > 0 for item in items),
            "negative_full_context_loss_account_count": sum(
                item["negative_full_context_loss_count"] > 0
                for item in items),
            "observed_both_interaction_signs_account_count": sum(
                item["observed_positive_interactions_over_cutoff"] > 0 and
                item["observed_negative_interactions_below_cutoff"] > 0
                for item in items),
        }

    result = {"input_reports": inputs,
              "interaction_cutoff": args.interaction_cutoff,
              "interpretation": "Only queried source-group interventions; not a population estimate or causal proof.",
              "by_predicted_class": {
                  "human": aggregate([item for item in rows
                                      if not item["predicted_bot"]]),
                  "bot": aggregate([item for item in rows
                                    if item["predicted_bot"]]),
              },
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) +
                           "\n", encoding="utf-8")
    for row in rows:
        print(row["account_id"], "Bot" if row["predicted_bot"] else "Human",
              "tau_b", round(row["kendall_tau_b"], 3)
              if row["kendall_tau_b"] is not None else None,
              "top3_overlap", row["top3_overlap_count"],
              "observed_interactions", row["observed_pair_count"],
              "positive", row["observed_positive_interactions_over_cutoff"],
              "negative", row["observed_negative_interactions_below_cutoff"])


if __name__ == "__main__":
    main()
