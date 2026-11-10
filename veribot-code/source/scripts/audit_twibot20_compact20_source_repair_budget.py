"""Compare certified source-record repair with direct labels at equal account actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


def hash_order(seed: int, account_id: str) -> str:
    return hashlib.sha256(f"compact20_repair_v1:{seed}:{account_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--source-edit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    edited = json.loads(args.source_edit.read_text(encoding="utf-8"))
    if (edited["compact_certificate_sha256"] != sha256(args.compact)
            or edited["source_archive_sha256"] != compact["source_archive_sha256"]
            or edited["base_facts_sha256"] != compact["base_facts_sha256"]):
        raise ValueError("The source-edit certificate does not bind to the frozen model")
    original = compact["rows"]
    altered = edited["rows"]
    if (len(original) != len(altered) or [x["account_id"] for x in original]
            != [x["account_id"] for x in altered]):
        raise ValueError("Official test account order differs")
    baseline_errors = sum(int(row["compact_class"] != edit["label_for_evaluation"])
                          for row, edit in zip(original, altered))
    if baseline_errors != edited["original_errors"]:
        raise ValueError("Original error count differs")
    eligible = [k for k, row in enumerate(altered) if row["first_tweet_locator"] is not None]
    if len(eligible) != edited["eligible_nonempty_lists"]:
        raise ValueError("Source-edit eligibility differs")
    budget = len(eligible) // 5
    seeds = []
    for seed in range(3):
        selected = sorted(eligible, key=lambda k: hash_order(seed, altered[k]["account_id"]))[:budget]
        selected_set = set(selected)
        truth = [int(row["label_for_evaluation"]) for row in altered]
        original_class = [int(row["original_class"]) for row in altered]
        injected_class = [int(row["first_removed_class"]) if k in selected_set else original_class[k]
                          for k, row in enumerate(altered)]
        if any(original_class[k] != int(original[k]["compact_class"])
               for k in range(len(altered))):
            raise ValueError("Frozen original classes differ")
        corrupt_errors = sum(int(p != y) for p, y in zip(injected_class, truth))
        corrected = list(injected_class)
        directly_labeled = list(injected_class)
        curve = [{"account_actions": 0, "source_repair_errors": corrupt_errors,
                  "direct_label_errors": corrupt_errors}]
        repair_removed = 0
        repair_added = 0
        for count, k in enumerate(selected, start=1):
            before = corrected[k] != truth[k]
            corrected[k] = original_class[k]
            after = corrected[k] != truth[k]
            repair_removed += int(before and not after)
            repair_added += int(not before and after)
            directly_labeled[k] = truth[k]
            curve.append({
                "account_actions": count,
                "source_repair_errors": sum(int(p != y) for p, y in zip(corrected, truth)),
                "direct_label_errors": sum(int(p != y) for p, y in zip(directly_labeled, truth)),
            })
        if curve[-1]["source_repair_errors"] != baseline_errors:
            raise ValueError("Repair failed to restore all clean predictions")
        selected_rows = [{
            "account_id": altered[k]["account_id"],
            "removed_source_locator": altered[k]["first_tweet_locator"],
            "original_tweet_count": altered[k]["original_tweet_count"],
            "original_logit": altered[k]["original_logit"],
            "injected_logit": altered[k]["first_removed_logit"],
            "original_class": original_class[k],
            "injected_class": injected_class[k],
            "label_for_evaluation": truth[k],
        } for k in selected]
        seeds.append({
            "seed": seed,
            "selection": "first floor(20% of eligible accounts) by SHA256('compact20_repair_v1:seed:account_id'); label-blind",
            "selected_accounts": budget,
            "selected_class_changes": sum(int(original_class[k] != injected_class[k])
                                          for k in selected),
            "clean_errors": baseline_errors,
            "injected_errors": corrupt_errors,
            "source_repair_errors_at_full_budget": curve[-1]["source_repair_errors"],
            "direct_label_errors_at_full_budget": curve[-1]["direct_label_errors"],
            "repair_removed_injected_errors": repair_removed,
            "repair_added_errors_when_restoring_original": repair_added,
            "direct_label_removed_errors": corrupt_errors - curve[-1]["direct_label_errors"],
            "budget_curve": curve,
            "selected_rows": selected_rows,
        })
    output = {
        "scope": "Frozen TwiBot-20 compact 20-fact classifier, official test population",
        "source_edit_certificate_sha256": sha256(args.source_edit),
        "compact_certificate_sha256": sha256(args.compact),
        "operation": "Known measurement-error injection deletes the first sampled tweet record for a label-blind hash-selected 20% of eligible test accounts. Source repair restores the exact original record and all derived tweet facts; direct labeling sets the same account prediction to its evaluation label. Costs are counted as account actions, not minutes.",
        "eligibility": len(eligible),
        "budget_per_seed": budget,
        "clean_errors": baseline_errors,
        "limitations": "This is a known artificial record deletion, not a naturally discovered error or bot-like behavior. The selected accounts have already been explored in other diagnostics. Direct label is an oracle control; equal account counts do not imply equal time or monetary cost. No external semantic truth is established.",
        "seeds": seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"eligibility": len(eligible), "budget_per_seed": budget,
                      "clean_errors": baseline_errors,
                      "seeds": [{key: value for key, value in row.items()
                                 if key not in ("budget_curve", "selected_rows")}
                                for row in seeds]}, indent=2))


if __name__ == "__main__":
    main()
