"""Reward-mode dispatch for the E3/E4/E5 GRPO comparison (doc §8). ZERO API.

Only the REWARD differs across E3/E4/E5 — same E2 best_lora, same seed, same dataset, same dev.
This module wraps the canonical three-gate reward (scripts/reward_v4.score) so E4 stays byte-identical.

  three_gate (E4): r = reward_v4.score(...).total   (format+label+evidence gates + Q)
  class_only (E3): +2 if <answer> == ground truth else -1  (ignores format & evidence entirely)
  random     (E5): keep the three-gate JUDGMENTS but PERMUTE the reward values across the batch,
                   so the reward distribution matches E4 while being decorrelated from each
                   completion's quality (negative control — must do significantly worse than E4).

Every mode can append per-completion records to a rollouts.jsonl (account_id, gates, rewards, mode).
"""
from __future__ import annotations

import json
import random
import re

from scripts.reward_v4 import score  # canonical three-gate reward (train-only bank)

ANSWER_RE = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.IGNORECASE)


def _text(comp):
    return comp if isinstance(comp, str) else (comp[0]["content"] if comp else "")


def _answer(text):
    m = ANSWER_RE.search(text)
    return m.group(1).lower() if m else None


def make_reward_func(mode, bank, near_miss=0.0, rollouts_path=None,
                     relaxed_getter=lambda: False, seed=0):
    assert mode in ("three_gate", "class_only", "random"), f"bad reward_mode {mode}"
    state = {"calls": 0}

    def _emit(records):
        if not rollouts_path:
            return
        with open(rollouts_path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reward(prompts, completions, account_id, label, **kwargs):
        texts = [_text(c) for c in completions]
        # always compute the true three-gate judgment (for logging + three_gate/random)
        trues = [score(t, aid, lb, bank, near_miss=near_miss, evidence_relaxed=relaxed_getter())
                 for t, aid, lb in zip(texts, account_id, label)]
        if mode == "three_gate":
            used = [float(r.total) for r in trues]
        elif mode == "class_only":
            used = [2.0 if _answer(t) == lb else -1.0 for t, lb in zip(texts, label)]
        else:  # random: permute the true reward values across the batch (distribution preserved)
            base = [float(r.total) for r in trues]
            rng = random.Random(seed * 1_000_003 + state["calls"])
            idx = list(range(len(base)))
            rng.shuffle(idx)
            used = [base[j] for j in idx]
        state["calls"] += 1
        _emit([
            {"account_id": aid, "mode": mode, "reward": u,
             "true_three_gate": float(r.total), "tier": getattr(r, "tier", None),
             "gate1": r.gate1_ok, "gate2": r.gate2_ok, "gate3": r.gate3_ok,
             "pred": _answer(t), "label": lb, "completion": t[:500]}
            for aid, lb, t, r, u in zip(account_id, label, texts, trues, used)
        ])
        return used

    reward.__name__ = f"reward_{mode}"
    return reward
