"""Recover rollouts.jsonl for E4 (and any GRPO run) from TRL's completion parquets — ZERO GPU.

TRL logs completions to <run>/completions/completions_<step>.parquet with columns
[step, prompt, completion, three_gate_offline, advantage] (16 rows/log = 2 prompts x group_size 8).
train_grpo_v4 never wrote rollouts.jsonl, but everything is recoverable offline:

  1. prompt -> account_id/label: build_dataset set prompt = canonical_prompt_text(units, tok); we
     rebuild the same map over TRAIN accounts and reverse-lookup (normalized to absorb special-token
     decoding differences). Uniqueness is verified.
  2. gate/order/veri/step/Q/R: recompute with reward_v4.score, reconstructing the curriculum
     (evidence_relaxed = step < relaxed_steps) so the recomputed R matches TRL's logged
     three_gate_offline EXACTLY — that agreement is the correctness proof.
  3. group_id = index of the unique prompt within a step's parquet; round = training step.

Validation mode (--limit-files N) reports match% and reward-agreement% WITHOUT writing.

  python scripts/v5/recover_rollouts.py --run outputs/v4/grpo/run_0.6b --limit-files 3   # validate
  python scripts/v5/recover_rollouts.py --run outputs/v4/grpo/run_0.6b                    # full write
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer  # noqa: E402
from verifiable_bot.v4.prompt_v4 import canonical_prompt_text  # noqa: E402
from verifiable_bot.v4.render_v4 import Unit, render_units  # noqa: E402
from scripts.reward_v4 import score, EvidenceBank  # noqa: E402

SPECIAL = re.compile(r"<\|[^|]*\|>")


def norm(p):
    """Normalize a prompt for matching: drop chat special tokens, strip edges."""
    return SPECIAL.sub("", p).strip()


def units_text(row):
    return render_units([Unit(u["unit_id"], u["text"]) for u in row["units"]])


def build_map(render_path, bank, tok):
    m, collisions = {}, 0
    for l in open(render_path, encoding="utf-8"):
        r = json.loads(l)
        if not bank.has(r["account_id"]):
            continue
        key = norm(canonical_prompt_text(units_text(r), tok))
        if key in m and m[key][0] != r["account_id"]:
            collisions += 1
        m[key] = (r["account_id"], r["label"])
    return m, collisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/v4/grpo/run_0.6b")
    ap.add_argument("--config", default="configs/grpo_v4_0.6b.yaml")
    ap.add_argument("--bank", default="outputs/v4/evidence_bank_v1/evidence_bank_train.jsonl")
    ap.add_argument("--render", default="outputs/v4/render_v4/train.jsonl")
    ap.add_argument("--limit-files", type=int, default=0, help=">0 = validation only, no write")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    relaxed_steps = int(cfg.get("relaxed_steps", 0))
    near_miss = float(cfg.get("near_miss", 0.0))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    bank = EvidenceBank(args.bank)
    pmap, collisions = build_map(args.render, bank, tok)
    print(f"[map] train prompts={len(pmap)} collisions={collisions}")

    files = sorted(glob.glob(f"{args.run}/completions/*.parquet"))
    if args.limit_files:
        files = files[:args.limit_files]
    print(f"[files] {len(files)} parquet(s)")

    n_rows = n_matched = n_reward_ok = 0
    recs = []
    unmatched_example = None
    for f in files:
        df = pd.read_parquet(f)
        uniq = list(dict.fromkeys(df["prompt"].tolist()))
        gid = {p: i for i, p in enumerate(uniq)}
        for _, row in df.iterrows():
            n_rows += 1
            key = norm(row["prompt"])
            hit = pmap.get(key)
            if hit is None:
                if unmatched_example is None:
                    unmatched_example = key[:200]
                continue
            n_matched += 1
            aid, label = hit
            logged = float(row["three_gate_offline"])
            # reconstruct the exact curriculum state by matching the logged reward: try the step-based
            # guess first, then the other (handles the relaxed_steps boundary off-by-one).
            guess = int(row["step"]) < relaxed_steps
            r = score(row["completion"], aid, label, bank, near_miss=near_miss, evidence_relaxed=guess)
            used_relaxed = guess
            if abs(r.total - logged) >= 1e-4:
                r2 = score(row["completion"], aid, label, bank, near_miss=near_miss,
                           evidence_relaxed=not guess)
                if abs(r2.total - logged) < 1e-4:
                    r, used_relaxed = r2, not guess
            reward_match = abs(r.total - logged) < 1e-4
            if reward_match:
                n_reward_ok += 1
            recs.append({
                "account_id": aid, "label": label, "round": int(row["step"]), "group_id": gid[row["prompt"]],
                "completion_text": row["completion"],
                "gate1_ok": r.gate1_ok, "gate2_ok": r.gate2_ok, "gate3_ok": r.gate3_ok,
                "order": r.order, "veri": r.veri, "step": r.step, "Q": r.Q, "R": r.total,
                "advantage": float(row["advantage"]), "reward_logged": logged, "tier": r.tier,
                "evidence_relaxed": used_relaxed, "reward_match": reward_match,
            })
    mrate = 100 * n_matched / max(n_rows, 1)
    rrate = 100 * n_reward_ok / max(n_matched, 1)
    print(f"[validate] rows={n_rows} matched={n_matched} ({mrate:.2f}%) "
          f"reward_recompute_agree={n_reward_ok}/{n_matched} ({rrate:.2f}%)")
    if unmatched_example:
        print(f"[validate] first UNMATCHED (normalized head): {unmatched_example!r}")
    if args.limit_files:
        print("[validate] validation-only (no write).")
        return
    if mrate < 100.0:
        print("[abort] account-match < 100% -> NOT writing (investigate prompt normalization).")
        return
    if rrate < 100.0:
        print(f"[warn] {n_matched - n_reward_ok} rows still disagree after boundary retry; "
              "flagged with reward_match=false (kept for audit, not dropped).")
    out = args.out or f"{args.run}/rollouts_recovered.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {len(recs)} rollouts -> {out}")


if __name__ == "__main__":
    main()
