"""Count what source and decision depth the final readable reasons support."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
BALANCE = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
CHAIN = ROOT / "research/certificates/bidirectional_report_source_model_chain.json"
OUT = ROOT / "research/certificates/readable_reason_source_depth.json"


def main() -> None:
    chain = json.loads(CHAIN.read_text(encoding="utf-8"))
    if (chain["status"] !=
            "all_readable_items_join_to_source_and_exact_frozen_effect" or
            chain["balance_rows_sha256"] != sha256_file(BALANCE)):
        raise ValueError("readable report and source chain differ")
    by_run: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "accounts": 0,
        "claims": 0,
        "source_levels": Counter(),
        "single_input_class_critical_levels": Counter(),
        "single_input_class_critical_accounts": 0,
        "opposing_displayed_claims": 0,
        "accounts_with_only_projection_displayed": 0,
        "accounts_with_only_projection_class_critical": 0,
    })
    projection = "published_averaged_tweet_tensor_train_only_projection"
    with gzip.open(BALANCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["dataset"], row["seed"])
            tally = by_run[key]
            items = row["balanced_displayed_items"]
            if len(items) not in (5, 6):
                raise ValueError("unexpected reason count")
            critical = [item for item in items
                        if item["class_changes_when_withheld"]]
            sign = 1 if row["prediction"] else -1
            tally["accounts"] += 1
            tally["claims"] += len(items)
            tally["source_levels"].update(item["source_level"]
                                          for item in items)
            tally["single_input_class_critical_levels"].update(
                item["source_level"] for item in critical)
            tally["single_input_class_critical_accounts"] += bool(critical)
            tally["opposing_displayed_claims"] += sum(
                float(item["logit_difference_when_withheld"]) * sign <
                -1e-10 for item in items)
            tally["accounts_with_only_projection_displayed"] += all(
                item["source_level"] == projection for item in items)
            tally["accounts_with_only_projection_class_critical"] += (
                bool(critical) and all(item["source_level"] == projection
                                       for item in critical))
    expected = {(run["dataset"], run["seed"]): run for run in chain["runs"]}
    if set(by_run) != set(expected):
        raise ValueError("run population differs")
    output_runs = []
    totals = Counter()
    dataset_levels: dict[str, Counter] = defaultdict(Counter)
    dataset_critical_levels: dict[str, Counter] = defaultdict(Counter)
    for key in sorted(by_run):
        row = by_run[key]
        bound = expected[key]
        if (row["accounts"] != bound["accounts"] or
                row["claims"] != bound["claims"]):
            raise ValueError("report totals differ")
        dataset_levels[key[0]].update(row["source_levels"])
        dataset_critical_levels[key[0]].update(
            row["single_input_class_critical_levels"])
        totals.update({k: v for k, v in row.items() if isinstance(v, int)})
        output_runs.append({"dataset": key[0], "seed": key[1], **row,
                            "source_levels": dict(row["source_levels"]),
                            "single_input_class_critical_levels": dict(
                                row["single_input_class_critical_levels"])})
    if totals["accounts"] != 5400 or totals["claims"] != 28007:
        raise ValueError("full population differs")
    OUT.write_text(json.dumps({
        "status": "final_readable_reason_scope_counted",
        "meaning": "single input mask class changes are not raw-source edits",
        "chain_certificate_sha256": sha256_file(CHAIN),
        "balance_rows_sha256": sha256_file(BALANCE),
        "totals": dict(totals),
        "source_levels_by_dataset": {name: dict(counts) for name, counts
                                     in sorted(dataset_levels.items())},
        "single_input_class_critical_levels_by_dataset": {
            name: dict(counts) for name, counts
            in sorted(dataset_critical_levels.items())},
        "runs": output_runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": dict(totals),
                      "source_levels_by_dataset": {
                          name: dict(counts) for name, counts
                          in sorted(dataset_levels.items())},
                      "single_input_class_critical_levels_by_dataset": {
                          name: dict(counts) for name, counts
                          in sorted(dataset_critical_levels.items())}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
