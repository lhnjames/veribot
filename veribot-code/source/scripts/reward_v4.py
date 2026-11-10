"""v4 offline reward: three gates + quality bonus Q. ZERO API — pure lookup against the TRAIN
evidence bank + programmatic string/set ops.

Priority, by design (design doc section 5): classification > verifiable evidence > format. Encoded
as gate ORDER and penalty magnitude, not partial credit:

    gate 1 (format):  parse ok + exactly 2 evidence groups + all ids valid & unique +
                      both chain keywords in the closed vocab + three <think> steps present
    gate 2 (label):   <answer> == ground_truth
    gate 3 (evidence):the model's 2 chain keywords == the bank's top-2 keywords (set)

    all three pass  -> R = 2.0 + Q,  Q in [0, 1.0]        (only path to a positive reward)
    gate 1 fails    -> R = -1.0   (and the bank is NOT consulted)
    gate 2 fails    -> R = -1.0
    gates 1&2 pass, gate 3 fails -> R = 0.0 (near-miss; configurable to -1.0 via strict)

Q = 0.4*order + 0.4*veri + 0.2*step   (order = rank-1 keyword emitted first; veri = mean bank
verifiability of the two keywords; step = 3-step gated reasoning check).

Hard offline guarantees, asserted at construction:
  * the bank path must end in '_train.jsonl'; anything containing 'eval' is refused;
  * this module imports NO http/socket/requests/urllib client and must never make a network call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---- offline guard: refuse to even be importable alongside a network client in this module ----
# (kept as a name-check; the trainer also runs with no API key mounted.)
_FORBIDDEN_NET = ("requests", "urllib.request", "httpx", "aiohttp", "socket")

ANSWER_RE = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.IGNORECASE)
CHAIN_RE = re.compile(r"<chain>(.*?)</chain>", re.IGNORECASE | re.DOTALL)
EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
STEP_RE = re.compile(r"step\s*([123])\b", re.IGNORECASE)
UNIT_ID_RE = re.compile(r"\b([A-Z]+_\d{2})\b")

NEAR_MISS_DEFAULT = 0.0
BASE_CORRECT = 2.0

# Closed keyword vocabulary — must match build_evidence_bank.KEYWORDS exactly.
KEYWORDS = [
    "follow imbalance", "low follower base", "no profile description", "default avatar",
    "account name entropy", "recent account age", "burst posting", "uniform posting interval",
    "duplicate text", "near-duplicate across accounts", "high retweet ratio", "no original content",
    "hashtag stuffing", "url spam", "mention flooding", "promotional template",
    "neighbor bot cluster", "low degree isolation", "cross-account co-posting", "topic monotony",
]
KEYWORD_SET = set(KEYWORDS)


@dataclass
class Reward:
    total: float
    gate1_ok: bool = False
    gate2_ok: bool = False
    gate3_ok: bool = False
    order: float = 0.0
    veri: float = 0.0
    step: float = 0.0
    Q: float = 0.0
    tier: str = ""
    detail: dict = field(default_factory=dict)


class EvidenceBank:
    """Loads the TRAIN bank and the per-keyword verifiability weights (train-frequency based)."""

    def __init__(self, bank_path: str | Path):
        path = Path(bank_path)
        name = path.name.lower()
        if "eval" in name or not name.endswith("_train.jsonl"):
            raise ValueError(
                f"reward may only read the TRAIN evidence bank; refused {path.name!r}. "
                "The eval bank (dev/test) must never be visible to training."
            )
        self.by_account: dict[str, dict] = {}
        kw_pos = {k: 0 for k in KEYWORDS}   # times keyword is the label-supporting evidence
        kw_tot = {k: 0 for k in KEYWORDS}
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            self.by_account[str(row["account_id"])] = row
            label = row["label"]
            for item in row.get("core_evidence", []):
                kw = item.get("keyword")
                if kw in kw_tot:
                    kw_tot[kw] += 1
                    kw_pos[kw] += 1  # bank evidence always supports the true label
        # veri weight: fraction present, smoothed by coverage (>=30 supports full weight)
        self.veri_weight = {}
        n_accounts = max(len(self.by_account), 1)
        for k in KEYWORDS:
            support = kw_tot[k]
            base = support / n_accounts
            self.veri_weight[k] = min(1.0, support / 30.0) * (0.5 + 0.5 * min(1.0, base * 5))
        # normalise to [0,1] by max so weights are comparable
        mx = max(self.veri_weight.values()) or 1.0
        self.veri_weight = {k: v / mx for k, v in self.veri_weight.items()}

    def top2_keywords(self, account_id: str) -> list[str]:
        row = self.by_account.get(str(account_id))
        if not row:
            return []
        return [item["keyword"] for item in row.get("core_evidence", [])][:2]

    def has(self, account_id: str) -> bool:
        return str(account_id) in self.by_account


def _parse(completion: str):
    ans = ANSWER_RE.search(completion)
    chain = CHAIN_RE.search(completion)
    ev = EVIDENCE_RE.search(completion)
    think = THINK_RE.search(completion)
    return ans, chain, ev, think


def _chain_keywords(chain_text: str) -> list[str]:
    # split on ' - ' but keywords contain hyphens too; match against vocab greedily
    parts = [p.strip() for p in re.split(r"\s+-\s+", chain_text.strip())]
    return parts


def score(completion: str, account_id: str, true_label: str, bank: EvidenceBank,
          *, near_miss: float = NEAR_MISS_DEFAULT, evidence_relaxed: bool = False) -> Reward:
    ans, chain, ev, think = _parse(completion)

    # ---- gate 1: format (no bank access) ----
    if not (ans and chain and ev and think):
        return Reward(-1.0, tier="invalid_format", detail={"reason": "missing tags"})
    ev_groups = [g.strip() for g in ev.group(1).split("|")]
    if len(ev_groups) != 2:
        return Reward(-1.0, tier="invalid_format", detail={"reason": "need exactly 2 evidence groups"})
    ev_ids = [UNIT_ID_RE.findall(g) for g in ev_groups]
    flat_ids = [i for g in ev_ids for i in g]
    if not all(ev_ids) or len(flat_ids) != len(set(flat_ids)):
        return Reward(-1.0, tier="invalid_format", detail={"reason": "empty/duplicate evidence ids"})
    kws = _chain_keywords(chain.group(1))
    if len(kws) != 2 or any(k not in KEYWORD_SET for k in kws):
        return Reward(-1.0, tier="invalid_format", detail={"reason": "chain keywords not 2-in-vocab"})
    steps = set(STEP_RE.findall(think.group(1)))
    if not {"1", "2", "3"}.issubset(steps):
        return Reward(-1.0, tier="invalid_format", detail={"reason": "need 3 think steps"})

    # ---- gate 2: classification ----
    pred = ans.group(1).lower()
    if pred != true_label:
        return Reward(-1.0, gate1_ok=True, tier="wrong_label", detail={"pred": pred})

    # ---- gate 3: evidence (bank lookup, TRAIN only) ----
    top2 = bank.top2_keywords(account_id)
    if not top2:
        # account not in bank (e.g. insufficient) — cannot verify; treat as near-miss
        return Reward(near_miss, gate1_ok=True, gate2_ok=True, tier="no_bank_evidence")
    if evidence_relaxed:
        gate3 = top2[0] in kws  # relaxed curriculum: only rank-1 keyword required
    else:
        gate3 = set(kws) == set(top2)
    if not gate3:
        return Reward(near_miss, gate1_ok=True, gate2_ok=True, tier="evidence_mismatch",
                      detail={"model_kws": kws, "bank_top2": top2})

    # ---- all three gates pass: base 2.0 + Q ----
    order = 1.0 if (kws and kws[0] == top2[0]) else 0.3
    veri = sum(bank.veri_weight.get(k, 0.0) for k in kws) / 2.0
    # step gate: 1 present, 2 cites a DIFFERENT unit category than step 1, 3 present & no new id
    step = _step_quality(think.group(1))
    Q = 0.4 * order + 0.4 * veri + 0.2 * step
    return Reward(BASE_CORRECT + Q, gate1_ok=True, gate2_ok=True, gate3_ok=True,
                  order=order, veri=veri, step=step, Q=Q, tier="strong",
                  detail={"model_kws": kws, "bank_top2": top2})


def _step_quality(think_text: str) -> float:
    """0..1: step1 cites a unit; step2 cites a DIFFERENT category; step3 present, no fresh id."""
    lines = think_text.strip().splitlines()
    by_step: dict[str, str] = {}
    cur = None
    for line in lines:
        m = STEP_RE.search(line)
        if m:
            cur = m.group(1)
            by_step[cur] = line
        elif cur:
            by_step[cur] = by_step.get(cur, "") + " " + line
    def cats(s):
        return {i.split("_")[0] for i in UNIT_ID_RE.findall(s)}
    q = 0.0
    if "1" in by_step and cats(by_step["1"]):
        q += 0.4
        if "2" in by_step and cats(by_step["2"]) and cats(by_step["2"]) - cats(by_step["1"]):
            q += 0.3
    if "3" in by_step:
        q += 0.3
    return q
