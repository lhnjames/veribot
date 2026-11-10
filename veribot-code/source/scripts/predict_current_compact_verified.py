"""Label-free source-to-evidence-to-readable-decision entry for compact models."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.current_compact_live_report import render_live
from verifiable_bot.fox8_twenty_concept_runtime import Fox8TwentyConceptRuntime
from verifiable_bot.mgtab_positive_human_report import render_positive_human
from verifiable_bot.mgtab_positive_human_runtime import MGTABPositiveHumanRuntime
from verifiable_bot.mgtab_twenty_concept_runtime import MGTABTwentyConceptRuntime


def load_runtime(root: Path, dataset: str, seed: int,
                 *, positive_human: bool = False):
    if dataset == "fox8":
        if positive_human:
            raise ValueError("positive Human evidence model is available for MGTAB")
        return Fox8TwentyConceptRuntime(root, seed)
    elif dataset == "mgtab":
        return (MGTABPositiveHumanRuntime(root, seed) if positive_human
                else MGTABTwentyConceptRuntime(root, seed))
    else:
        raise ValueError("choose fox8 or mgtab")


def render_account(runtime, dataset: str, seed: int, account_id: str,
                   *, atomic_facts: bool = False,
                   positive_human: bool = False) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if atomic_facts:
            if dataset != "fox8":
                raise ValueError("atomic fact withholding is available for fox8")
            result = runtime.predict_with_atomic_fact_effects(account_id)
            report = render_live(result, dataset, seed)
            report["atomic_fact_effects"] = result["atomic_fact_effects"]
            report["class_critical_source_facts"] = result[
                "class_critical_source_facts"]
            report["score_relevant_source_facts"] = result[
                "score_relevant_source_facts"]
            report["atomic_decision_basis"] = {
                "class_critical": [row for row in result["atomic_fact_effects"]
                                   if row["effect_level"] == "class_critical"],
                "score_relevant": [row for row in result["atomic_fact_effects"]
                                   if row["effect_level"] == "score_relevant"],
                "context_only_fact_count": sum(
                    row["effect_level"] == "context_only"
                    for row in result["atomic_fact_effects"]),
                "class_critical_source_groups": [
                    row["group"] for row in report["source_groups"]
                    if row["changes_class_without_group"]],
                "zero_evidence_prediction": int(
                    result["all_evidence_zero_probability"] >= result["threshold"]),
                "class_depends_on_evidence_vs_zero_input": report[
                    "all_evidence_zero_changes_class"],
                "effect_scope": "one cited fact withheld from the frozen evidence builder; released records and graph remain unchanged",
            }
            return report
        if positive_human:
            if dataset != "mgtab":
                raise ValueError("positive Human evidence model is available for MGTAB")
            return render_positive_human(runtime.predict(account_id), seed)
        return render_live(runtime.predict(account_id), dataset, seed)


def predict(root: Path, dataset: str, seed: int, account_id: str,
            *, atomic_facts: bool = False,
            positive_human: bool = False) -> dict:
    if atomic_facts and positive_human:
        raise ValueError("choose atomic facts or positive Human evidence")
    return render_account(load_runtime(root, dataset, seed,
                                       positive_human=positive_human),
                          dataset, seed, account_id, atomic_facts=atomic_facts,
                          positive_human=positive_human)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("fox8", "mgtab"))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--atomic-facts", action="store_true",
                        help="for Fox8, recompute each cited fact's frozen-model effect")
    parser.add_argument("--positive-human", action="store_true",
                        help="for MGTAB, use the frozen source-present positive Human evidence model")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = predict(root, arguments.dataset, arguments.seed,
                     arguments.account_id, atomic_facts=arguments.atomic_facts,
                     positive_human=arguments.positive_human)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(content, encoding="utf-8")
        print(str(arguments.out))
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
