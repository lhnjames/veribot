"""Compare source closure at the target with the current source-node closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from train_aaai_veribot_fox8 import load_inputs
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.fox8_hgt_context_closure import hgt_context_closure
from verifiable_bot.fox8_two_stage_inputs import Fox8TwoStageInputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    _, _, _, ids, graph, _ = load_inputs()
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    bank = Fox8TwoStageInputs.build(
        load_account_atoms(raw, ids), load_relation_atoms(events, raw, ids), ids)
    rows = []
    for case in sample["cases"]:
        uid = case["account_id"]
        closure, _, nodes = hgt_context_closure(bank, graph, [uid])
        profile_count = sum(len(bank.accounts[i].profile_field_ids) for i in nodes)
        tweet_count = sum(len(bank.accounts[i].tweet_ordinals) for i in nodes)
        total = profile_count + tweet_count + closure["native_relation_source_count"]
        rows.append({"account_id": uid,
                     "predicted_bot": case["baseline"]["predicted_bot"],
                     "target_hgt_context_sources": total,
                     "target_hgt_context_accounts": len(nodes),
                     "current_full_closure_sources": case[
                         "candidate_source_counts"]["with_hgt_context"],
                     "source_reduction_ratio": 1 - total / case[
                         "candidate_source_counts"]["with_hgt_context"]})
        print(json.dumps(rows[-1]), flush=True)
    args.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
