"""Summarize all sampled dev accounts in a budgeted source-coalition audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def summarize(report: dict, *, require_complete: bool = True,
              margin_buffer: float = 0.05) -> dict:
    if margin_buffer < 0:
        raise ValueError("Margin buffer must be nonnegative")
    if report.get("split") != "dev":
        raise ValueError("Only development-set searches belong in this summary")
    selected = report["selected_account_ids"]
    cases = report["cases"]
    if len(selected) != len(set(selected)):
        raise ValueError("Sampled account IDs overlap")
    if len(cases) != report["completed_cases"]:
        raise ValueError("Completed-case counter disagrees with case records")
    if require_complete and len(cases) != len(selected):
        raise ValueError("Audit is incomplete")
    if [row["account_id"] for row in cases] != list(selected[:len(cases)]):
        raise ValueError("Case order differs from the locked sample")
    rows = []
    for case in cases:
        audit = case["coalition_audit"]
        if (audit["account_id"] != case["account_id"] or
                audit["predicted_bot"] != case["predicted_bot"] or
                audit["oracle_query_count"] > audit["query_budget"]):
            raise ValueError("Per-account search contract differs")
        sufficient = audit["observed_nondominated_sufficient_group_sets"]
        bidirectional = audit["searched_bidirectional_group_sets"]
        directional = audit.get("directional_support_group_sets")
        direction = 1 if case["predicted_bot"] else -1
        threshold = audit["threshold"]
        robust_bidirectional = [item for item in bidirectional if
            direction * (item["retained_margin"] - threshold) >= margin_buffer
            and direction * (item["full_without_candidate_margin"] -
                             threshold) <= -margin_buffer]
        robust_directional = ([item for item in directional if
            item["signed_gain_over_empty"] >= margin_buffer and
            item["signed_loss_when_removed_from_full"] >= margin_buffer]
            if directional is not None else None)
        rows.append({
            "account_id": case["account_id"],
            "label_bot": bool(case["label"]),
            "predicted_bot": bool(case["predicted_bot"]),
            "correct": bool(case["label"]) == bool(case["predicted_bot"]),
            "group_count": len(audit["group_names"]),
            "dependency_source_count": audit["full_dependency_source_count"],
            "oracle_query_count": audit["oracle_query_count"],
            "query_budget_exhausted": audit["query_budget_exhausted"],
            "empty_input_preserves_prediction":
                audit["empty_input_preserves_original_class"],
            "observed_nonempty_nondominated_sufficient_count": sum(
                bool(item["groups"]) for item in sufficient),
            "observed_nonempty_class_preserving_count": sum(
                bool(item["groups"]) and item["preserves_original_class"]
                for item in audit["coalitions"]),
            "observed_bidirectional_count": len(bidirectional),
            "observed_buffered_bidirectional_count": len(robust_bidirectional),
            "observed_directional_support_count": (
                len(directional) if directional is not None else None),
            "observed_buffered_directional_support_count": (
                len(robust_directional) if robust_directional is not None
                else None),
            "smallest_observed_bidirectional_source_count":
                min((item["source_count"] for item in bidirectional),
                    default=None),
            "smallest_observed_directional_source_count":
                (min((item["source_count"] for item in directional),
                     default=None) if directional is not None else None),
            "smallest_observed_buffered_bidirectional_source_count":
                min((item["source_count"] for item in robust_bidirectional),
                    default=None),
        })

    def aggregate(items: list[dict]) -> dict:
        return {
            "sampled": len(items),
            "correct": sum(item["correct"] for item in items),
            "empty_input_preserves_prediction": sum(
                item["empty_input_preserves_prediction"] for item in items),
            "found_nonempty_nondominated_sufficient": sum(
                item["observed_nonempty_nondominated_sufficient_count"] > 0
                for item in items),
            "found_nonempty_class_preserving": sum(
                item["observed_nonempty_class_preserving_count"] > 0
                for item in items),
            "found_bidirectional": sum(
                item["observed_bidirectional_count"] > 0 for item in items),
            "found_buffered_bidirectional": sum(
                item["observed_buffered_bidirectional_count"] > 0
                for item in items),
            "found_directional_support": (sum(
                item["observed_directional_support_count"] > 0
                for item in items) if all(
                    item["observed_directional_support_count"] is not None
                    for item in items) else None),
            "found_buffered_directional_support": (sum(
                item["observed_buffered_directional_support_count"] > 0
                for item in items) if all(
                    item["observed_buffered_directional_support_count"]
                    is not None for item in items) else None),
            "query_budget_exhausted": sum(
                item["query_budget_exhausted"] for item in items),
            "median_dependency_sources": (median(
                item["dependency_source_count"] for item in items)
                if items else None),
            "median_smallest_observed_bidirectional_sources": (median(
                item["smallest_observed_bidirectional_source_count"]
                for item in items if
                item["smallest_observed_bidirectional_source_count"] is not None)
                if any(item["smallest_observed_bidirectional_source_count"]
                       is not None for item in items) else None),
            "median_smallest_observed_buffered_bidirectional_sources":
                (median(item[
                    "smallest_observed_buffered_bidirectional_source_count"]
                    for item in items if item[
                    "smallest_observed_buffered_bidirectional_source_count"]
                    is not None) if any(item[
                    "smallest_observed_buffered_bidirectional_source_count"]
                    is not None for item in items) else None),
        }

    return {
        "split": "dev",
        "sampling_seed": report["sampling_seed"],
        "sample_per_class": report["sample_per_class"],
        "margin_buffer": margin_buffer,
        "expected_count": len(selected),
        "completed_count": len(cases),
        "complete": len(cases) == len(selected),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_sha256": report["source_sha256"],
        "interpretation": (
            "Counts are budget-limited discoveries among the fixed sample, "
            "not exhaustive minimal-set prevalence or semantic validity."),
        "all": aggregate(rows),
        "by_true_class": {
            "human": aggregate([item for item in rows if not item["label_bot"]]),
            "bot": aggregate([item for item in rows if item["label_bot"]]),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--margin-buffer", type=float, default=0.05)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = summarize(json.loads(raw),
                       require_complete=not args.allow_partial,
                       margin_buffer=args.margin_buffer)
    result["input_report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("expected_count", "completed_count", "complete",
                       "by_true_class")}), flush=True)


if __name__ == "__main__":
    main()
