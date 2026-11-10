"""Diagnosis (step 1 of user's 5-step plan): across all 40 GRPO rounds, for rollouts where the
TRUE label is bot and the model's prediction was correct (bot), did ANY candidate ever exceed the
base_correct reward floor (reward > 1.0, i.e. cleared sufficiency+consistency+necessity hard
gates and got some r_hybrid bonus)? Track this per-round to see if it's "always zero" (never
learned) vs "some early, then collapsed" (mid-training regression).

Method: the model's prompt never contains the true label (it must predict it), so we recover the
true label by exact-matching the EVIDENCE_UNITS text block in each rollout's prompt against a
precomputed hash->label map built from the actual training data file (train_twibot20_only.jsonl),
using the identical deterministic rendering function (prompting.py::render_record). Since
build_units/render_units are deterministic given the same account record, this is an exact match,
not a fuzzy one.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from verifiable_bot.prompting import render_record
from verifiable_bot.schema import AccountRecord

DATA_FILE = "outputs/unified_full/train_twibot20_only.jsonl"
ROUND_GLOB = "outputs/veriroute_lite/grpo_0.6b_twibot20only_20260722_0419/round_*/completions/*.parquet"

EVID_RE = re.compile(r"EVIDENCE_UNITS\n(.*?)\n\nRespond with", re.DOTALL)


def build_hash_to_label() -> dict[str, str]:
    mapping = {}
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record = AccountRecord.model_validate(row)
            rendered = render_record(record)
            m = EVID_RE.search(rendered + "\n\nRespond with")
            text = m.group(1) if m else rendered[len("EVIDENCE_UNITS\n"):]
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            mapping[h] = record.label
    return mapping


def main() -> None:
    hash_to_label = build_hash_to_label()
    print(f"built hash map for {len(hash_to_label)} accounts", flush=True)

    files = sorted(glob.glob(ROUND_GLOB))
    print(f"found {len(files)} completion parquet files", flush=True)

    per_round = {}  # round_idx -> dict(counts)
    unmatched = 0
    total_rows = 0

    round_re = re.compile(r"round_(\d+)")

    for fp in files:
        m = round_re.search(fp)
        round_idx = int(m.group(1)) if m else -1
        df = pd.read_parquet(fp)
        bucket = per_round.setdefault(round_idx, {
            "bot_correct_total": 0, "bot_correct_reward_gt1": 0,
            "bot_correct_strong_or_perfect": 0,
            "human_correct_total": 0, "human_correct_reward_gt1": 0,
            "bot_wrong": 0, "human_wrong": 0, "invalid_format": 0, "unmatched": 0,
        })
        for _, row in df.iterrows():
            total_rows += 1
            prompt = row["prompt"]
            m2 = EVID_RE.search(prompt)
            if not m2:
                bucket["unmatched"] += 1
                unmatched += 1
                continue
            h = hashlib.sha256(m2.group(1).encode("utf-8")).hexdigest()
            label = hash_to_label.get(h)
            if label is None:
                bucket["unmatched"] += 1
                unmatched += 1
                continue
            reward = float(row["reward"])
            completion = row["completion"]
            try:
                parsed = json.loads(completion.strip().strip("`"))
                prediction = parsed.get("prediction")
            except Exception:
                prediction = None

            if reward <= -1.5:
                bucket["invalid_format"] += 1
                continue

            correct = (prediction == label)
            if label == "bot":
                if correct:
                    bucket["bot_correct_total"] += 1
                    if reward > 1.0 + 1e-6:
                        bucket["bot_correct_reward_gt1"] += 1
                    if reward >= 2.999:
                        bucket["bot_correct_strong_or_perfect"] += 1
                else:
                    bucket["bot_wrong"] += 1
            else:
                if correct:
                    bucket["human_correct_total"] += 1
                    if reward > 1.0 + 1e-6:
                        bucket["human_correct_reward_gt1"] += 1
                else:
                    bucket["human_wrong"] += 1

    print(f"\ntotal rows scanned: {total_rows}, unmatched (hash miss): {unmatched}")
    print("\n=== per-round breakdown ===")
    print(f"{'round':>5} | {'bot_ok':>7} {'bot_ok>1.0':>10} {'bot_strong/perfect':>18} | "
          f"{'human_ok':>8} {'human_ok>1.0':>12} | {'bot_wrong':>9} {'human_wrong':>11} {'invalid':>7} {'unmatched':>9}")
    total_bot_correct = 0
    total_bot_gt1 = 0
    total_bot_strong = 0
    for r in sorted(per_round):
        b = per_round[r]
        total_bot_correct += b["bot_correct_total"]
        total_bot_gt1 += b["bot_correct_reward_gt1"]
        total_bot_strong += b["bot_correct_strong_or_perfect"]
        print(f"{r:>5} | {b['bot_correct_total']:>7} {b['bot_correct_reward_gt1']:>10} "
              f"{b['bot_correct_strong_or_perfect']:>18} | "
              f"{b['human_correct_total']:>8} {b['human_correct_reward_gt1']:>12} | "
              f"{b['bot_wrong']:>9} {b['human_wrong']:>11} {b['invalid_format']:>7} {b['unmatched']:>9}")

    print(f"\n=== GRAND TOTAL across all rounds ===")
    print(f"true_label=bot, prediction=bot (correct) rollouts: {total_bot_correct}")
    print(f"  of those, reward > 1.0 (cleared sufficiency+consistency+necessity gates): {total_bot_gt1}")
    print(f"  of those, reward >= 3.0 (strong/perfect tier): {total_bot_strong}")


if __name__ == "__main__":
    main()
