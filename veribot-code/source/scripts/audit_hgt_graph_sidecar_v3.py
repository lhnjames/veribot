"""Recompute GRAPH prose from raw attention-sidecar rows and audit a deterministic sample."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def tier(fraction: float) -> str:
    if fraction > 0.6:
        return "majority"
    if fraction < 0.4:
        return "a small minority"
    return "about half"


def verify(row: dict) -> list[str]:
    errors = []
    neighbors = row["top_neighbors"]
    facts = row["graph_facts"]
    if row["upstream_verdict"] != (
        "bot" if row["target_probability"] >= row["frozen_threshold"] else "human"
    ):
        errors.append("target verdict does not match probability/threshold")
    if not neighbors:
        if facts:
            errors.append("isolated account has GRAPH facts")
        return errors
    ids = [neighbor["neighbor_account_id"] for neighbor in neighbors]
    if len(ids) != len(set(ids)):
        errors.append("neighbor identity was not deduplicated")
    weights = [neighbor["ranking_attention"] for neighbor in neighbors]
    if weights != sorted(weights, reverse=True):
        errors.append("neighbors are not attention-ranked")
    expected_tier = tier(sum(n["predicted_label"] == "bot" for n in neighbors) / len(neighbors))
    expected_first = (
        "Among the neighbors the upstream classifier attended to most strongly, "
        f"{expected_tier} are themselves predicted as bot."
    )
    top = neighbors[0]
    direction = "mutual follow" if top["direction"] == "mutual" else top["direction"]
    expected_second = (
        f"The single most attended-to neighbor (edge type: {direction}) is predicted "
        f"{top['predicted_label']}."
    )
    if facts != [expected_first, expected_second]:
        errors.append("GRAPH prose does not recompute from raw ranked neighbors")
    if any(re.search(r"\d", fact) for fact in facts):
        errors.append("model-visible GRAPH prose contains a digit")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    rng = random.Random(42)
    reservoir = []
    total = eligible = invalid = 0
    examples = []
    with args.sidecar.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            if not row["top_neighbors"]:
                continue
            eligible += 1
            errors = verify(row)
            if errors:
                invalid += 1
                if len(examples) < 20:
                    examples.append({"account_id": row["account_id"], "errors": errors})
            if len(reservoir) < args.sample_size:
                reservoir.append(row)
            else:
                replacement = rng.randrange(eligible)
                if replacement < args.sample_size:
                    reservoir[replacement] = row
    report = {
        "sidecar": str(args.sidecar), "total_accounts": total, "eligible_accounts": eligible,
        "checked_eligible_accounts": eligible, "invalid_accounts": invalid,
        "all_recomputed_consistent": invalid == 0, "sample_size": len(reservoir),
        "sample_invalid": sum(bool(verify(row)) for row in reservoir),
        "invalid_examples": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    args.sample.write_text(json.dumps(reservoir, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if invalid:
        raise SystemExit("GRAPH sidecar audit failed")


if __name__ == "__main__":
    main()
