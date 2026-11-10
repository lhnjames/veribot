"""Validate the final v3 cold-start file that SFT will actually consume."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.format_check import check_format
from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, hidden_unit_metadata, render_units, unit_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manual-sample", type=Path, required=True)
    args = parser.parse_args()

    counts = Counter()
    by_stratum = Counter()
    status = defaultdict(Counter)
    bad_examples = []
    bot_candidates = []
    sufficiency = defaultdict(list)
    for line_number, line in enumerate(args.input.open(encoding="utf-8"), 1):
        counts["rows"] += 1
        row = json.loads(line)
        record = AccountRecord.model_validate(row["record"])
        target = row["target"]
        quality = row.get("quality", {})
        units = build_units(record)
        texts = unit_map(units)
        metadata = hidden_unit_metadata(units)
        checked = check_format(json.dumps(target, ensure_ascii=False), texts, metadata)
        if checked.ok:
            counts["format_valid"] += 1
        else:
            bad_examples.append({"line": line_number, "source": record.source,
                                 "account_id": record.account_id, "reason": checked.reason})
            continue
        required = "agree" if record.upstream_verdict == record.label else "disagree"
        if target["agreement"] != required:
            counts["agreement_mismatch"] += 1
        else:
            counts["agreement_consistent"] += 1
        by_stratum[(record.source, record.label)] += 1
        status[(record.source, record.label)][target["evidence_status"]] += 1
        graph_ids = {unit.unit_id for unit in units if unit.unit_id.startswith("GRAPH_")}
        cited = set(target["evidence_ids"])
        if graph_ids:
            counts["graph_available"] += 1
            if cited & graph_ids:
                counts["graph_cited"] += 1
        claim_prose = " ".join(
            f"{claim['claim']} {claim.get('pattern', '')}" for claim in target["claims"]
        )
        if re.search(r"\d", claim_prose):
            counts["rows_with_digit_in_generated_prose"] += 1
        categories = {source.split("_", 1)[0] for source in cited}
        if record.label == "bot" and target["evidence_status"] == "sufficient":
            counts["bot_sufficient_rows"] += 1
            if len(categories) >= 2:
                counts["bot_sufficient_multicategory_rows"] += 1
        if "sufficiency_k5" in quality:
            sufficiency[record.label].append(float(quality["sufficiency_k5"]))
        if record.label == "bot":
            bot_candidates.append({
                "source": record.source, "account_id": record.account_id,
                "trusted_label": record.label, "upstream_verdict": record.upstream_verdict,
                "target": target, "quality": quality, "visible_units": render_units(units),
            })

    n = counts["rows"]
    available = counts["graph_available"]
    bot_sufficient = counts["bot_sufficient_rows"]
    report = {
        "input": str(args.input), "total_rows": n,
        "format_valid": counts["format_valid"],
        "format_valid_rate": counts["format_valid"] / n if n else 0.0,
        "agreement_consistent": counts["agreement_consistent"],
        "agreement_mismatch": counts["agreement_mismatch"],
        "rows_with_digit_in_generated_prose": counts["rows_with_digit_in_generated_prose"],
        "digit_residual_rate": counts["rows_with_digit_in_generated_prose"] / n if n else 0.0,
        "graph_available": available, "graph_cited": counts["graph_cited"],
        "graph_citation_rate_of_available": counts["graph_cited"] / available if available else 0.0,
        "bot_sufficient_rows": bot_sufficient,
        "bot_sufficient_multicategory_rate": (
            counts["bot_sufficient_multicategory_rows"] / bot_sufficient if bot_sufficient else 0.0
        ),
        "mean_k5_sufficiency": {
            label: sum(values) / len(values) if values else 0.0
            for label, values in sufficiency.items()
        },
        "bot_k5_sufficiency_at_least_three_of_five_rate": (
            sum(value >= 0.6 for value in sufficiency["bot"]) / len(sufficiency["bot"])
            if sufficiency["bot"] else 0.0
        ),
        "counts_by_source_label": {
            f"{source}/{label}": count for (source, label), count in sorted(by_stratum.items())
        },
        "status_by_source_label": {
            f"{source}/{label}": dict(value) for (source, label), value in sorted(status.items())
        },
        "sample_bad_examples": bad_examples[:30], "n_bad_examples": len(bad_examples),
        "gates": {
            "format_all_valid": counts["format_valid"] == n,
            "agreement_all_consistent": counts["agreement_mismatch"] == 0,
            "bot_sufficiency_above_half": (
                bool(sufficiency["bot"])
                and sum(value >= 0.6 for value in sufficiency["bot"]) / len(sufficiency["bot"]) > 0.5
            ),
            "graph_citation_above_thirty_percent": available > 0 and counts["graph_cited"] / available > 0.30,
            "digit_residual_near_zero": counts["rows_with_digit_in_generated_prose"] == 0,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rng = random.Random(42)
    sample = rng.sample(bot_candidates, k=min(30, len(bot_candidates)))
    args.manual_sample.parent.mkdir(parents=True, exist_ok=True)
    args.manual_sample.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not all(report["gates"].values()):
        raise SystemExit("one or more cold-start quality gates failed")


if __name__ == "__main__":
    main()
