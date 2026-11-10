"""Step 1 (design doc section 4.3): is the necessity judge (label_approve_rate on full vs.
evidence-masked text) actually sensitive to removing the cited evidence, or does it mostly return
the same verdict either way (S_nec = approve_full - approve_masked == 0)?

Samples real accounts + real teacher claims/evidence_ids from the existing cold-start data (not
hand-constructed), rebuilds units the same way training does, and reports the S_nec distribution.
Covers a real mix of evidence-combo strength, per the design doc's "10-15 accounts, different
evidence-strength combinations" instruction, rather than reusing the same 2 accounts the earlier
round of testing used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.llm_client import DeepSeekJudge
from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, render_masked, render_units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                     default=Path("/path/to/data_root/outputs/veriroute_lite/cold_start_sft_v4_final.jsonl"))
    ap.add_argument("--judge-cache", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["target"]["evidence_status"] == "sufficient" and len(row["target"]["claims"]) >= 1:
                rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Stratify by number of distinct evidence categories cited (rough proxy for "evidence
    # strength combination" per the design doc) so the sample isn't dominated by single-PROFILE-fact
    # accounts.
    def n_categories(row) -> int:
        cats = {sid.split("_")[0] for c in row["target"]["claims"] for sid in c["source_ids"]}
        return len(cats)

    rows.sort(key=lambda r: -n_categories(r))
    sample = rows[: args.n // 2] + rng.sample(rows[args.n // 2:], min(len(rows) - args.n // 2, args.n - args.n // 2))
    sample = sample[: args.n]

    judge = DeepSeekJudge(cache_dir=args.judge_cache, k=5, max_workers=16)

    print(f"{'account_id':22s} {'label':6s} {'pred':6s} {'n_cat':>5s} {'approve_full':>12s} "
          f"{'approve_masked':>14s} {'s_nec':>7s}")
    s_necs = []
    for row in sample:
        record = AccountRecord.model_validate(row["record"])
        units = build_units(record)
        target = row["target"]
        evidence_ids = sorted({sid for c in target["claims"] for sid in c["source_ids"]})
        full_text = render_units(units)
        masked_text = render_masked(units, set(evidence_ids))
        approve_full = judge.label_approve_rate(full_text, target["prediction"])
        approve_masked = judge.label_approve_rate(masked_text, target["prediction"])
        s_nec = max(0.0, min(1.0, approve_full - approve_masked))
        s_necs.append(s_nec)
        print(f"{row['record']['account_id']:22s} {row['record']['label']:6s} {target['prediction']:6s} "
              f"{n_categories(row):5d} {approve_full:12.2f} {approve_masked:14.2f} {s_nec:7.2f}")

    n = len(s_necs)
    zero_rate = sum(1 for s in s_necs if s == 0.0) / n
    print("\n===== AGGREGATE =====")
    print(f"n accounts = {n}")
    print(f"mean S_nec = {sum(s_necs)/n:.3f}")
    print(f"S_nec == 0 rate = {zero_rate:.1%}  (design doc concern threshold: >50% is a problem)")
    print(f"S_nec distribution: {sorted(s_necs)}")


if __name__ == "__main__":
    main()
