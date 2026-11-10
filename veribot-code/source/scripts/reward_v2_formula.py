"""Standalone implementation of the v2 reward formula (design doc "VeriRoute-Lite v2", section 4),
for offline unit testing BEFORE it is wired into rewards.py/train_grpo.py for real training.

This intentionally reuses the existing judge primitives unchanged (claim_approval, sufficiency_vote,
consistency_vote, label_approve_rate -- all in llm_client.py) and the existing binary-search length
reward (_length_reward in rewards.py). Only the AGGREGATION on top of those primitives is new here,
matching design doc section 4 exactly:

  - 4.0  agree branch never looks at the true label; disagree branch always does.
  - 4.1  gate cascade: format -> agreement -> (insufficient: 4.2.1) / (sufficient: 4.2.2) -> disagree: 4.3
  - 4.2.1 sampling-audited honesty check for declared "insufficient"
  - 4.2.2 continuous multi-signal-gated scoring for declared "sufficient", with the 3 documented
          loophole fixes: R_len conditioned on s_suf/s_nec, consistency only counted at n_claims>=2
          (naturally true here since it only appears inside the eligible_multisignal branch, which
          itself requires n_claims>=2), and R_div counted over *approved, per-claim* categories
          rather than over all source_ids appearing anywhere.
  - 4.3  disagree branch, asymmetric HGT-right vs HGT-wrong reward.
  - 4.4  the four invariants this script's test harness checks.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from verifiable_bot.llm_client import DeepSeekJudge
from verifiable_bot.rewards import _length_reward
from verifiable_bot.schema import Claim
from verifiable_bot.units import Unit, render_masked, render_units, unit_map

STRONG_SUF_MIN = 2 / 3          # same "1/3, 2/3" discrete-vote cutoffs used throughout the project
WEAK_SUF_MAX = 1 / 3
CLAIM_APPROVE_MIN = 2 / 3        # a claim counts as "approved" (for R_div's category set) at this vote fraction

HONEST_INSUFFICIENT_REWARD = 1.0 + 0.3
NO_BONUS_INSUFFICIENT_REWARD = 1.0
SUFFICIENT_BUT_UNSUPPORTED_PENALTY = 1.0 - 0.3
DEFAULT_AUDIT_PROB = 0.175       # doc: "建议取 0.15~0.2"


def unit_category(unit_id: str) -> str:
    """PROFILE_03 -> PROFILE, TWEET_12 -> TWEET, OBS_02 -> OBS, ... (section 4.2.2 item 3: R_div
    must be computed over the categories of APPROVED claims, not over raw source_id prefixes seen
    anywhere in the output)."""
    return unit_id.split("_")[0]


@dataclass
class RewardV2Breakdown:
    total: float
    branch: str
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4.2.1: declared evidence_status = "insufficient"
# ---------------------------------------------------------------------------


def score_agree_insufficient(
    judge: DeepSeekJudge,
    claims_text: str,
    prediction: str,
    audit_prob: float = DEFAULT_AUDIT_PROB,
    rng: random.Random | None = None,
    force_audit: bool | None = None,
) -> RewardV2Breakdown:
    rng = rng or random
    audited = force_audit if force_audit is not None else (rng.random() < audit_prob)
    if not audited:
        return RewardV2Breakdown(
            total=HONEST_INSUFFICIENT_REWARD, branch="insufficient_unaudited", detail={"audited": False},
        )
    s_suf = judge.sufficiency_vote(claims_text, prediction)
    if s_suf < STRONG_SUF_MIN:
        return RewardV2Breakdown(
            total=HONEST_INSUFFICIENT_REWARD, branch="insufficient_audited_honest",
            detail={"audited": True, "s_suf": s_suf},
        )
    return RewardV2Breakdown(
        total=NO_BONUS_INSUFFICIENT_REWARD, branch="insufficient_audited_dishonest",
        detail={"audited": True, "s_suf": s_suf},
    )


# ---------------------------------------------------------------------------
# 4.2.2: declared evidence_status = "sufficient"
# ---------------------------------------------------------------------------


def score_agree_sufficient(
    judge: DeepSeekJudge, units: list[Unit], claims: list[Claim], prediction: str,
) -> RewardV2Breakdown:
    units_map = unit_map(units)
    claims_text = "\n".join(f"- {c.claim}" for c in claims)

    s_suf = judge.sufficiency_vote(claims_text, prediction)
    if s_suf < WEAK_SUF_MAX:
        return RewardV2Breakdown(
            total=SUFFICIENT_BUT_UNSUPPORTED_PENALTY, branch="sufficient_but_unsupported",
            detail={"s_suf": s_suf},
        )

    claim_scores = []
    approved_categories: set[str] = set()
    for c in claims:
        score = judge.claim_approval(c.claim, [units_map[sid] for sid in c.source_ids], c.pattern)
        claim_scores.append(score)
        if score >= CLAIM_APPROVE_MIN:
            approved_categories.update(unit_category(sid) for sid in c.source_ids)
    q = 0.5 * min(claim_scores) + 0.5 * (sum(claim_scores) / len(claim_scores))

    evidence_ids = sorted({sid for c in claims for sid in c.source_ids})
    full_text = render_units(units)
    masked_text = render_masked(units, set(evidence_ids))
    approve_full = judge.label_approve_rate(full_text, prediction)
    approve_masked = judge.label_approve_rate(masked_text, prediction)
    s_nec = max(0.0, min(1.0, approve_full - approve_masked))

    s_consistency = None
    if len(claims) >= 2:
        s_consistency = judge.consistency_vote(claims_text)

    eligible_multisignal = len(claims) >= 2 and len(approved_categories) >= 2
    r_div = min(len(approved_categories), 3) / 3.0

    detail = {
        "s_suf": s_suf, "q": q, "s_nec": s_nec, "s_consistency": s_consistency,
        "approved_categories": sorted(approved_categories), "r_div": r_div,
        "eligible_multisignal": eligible_multisignal,
        "approve_full": approve_full, "approve_masked": approve_masked,
    }

    if not eligible_multisignal:
        total = 1.0 + 0.6 * q + 0.8 * s_suf + 0.8 * s_nec
        return RewardV2Breakdown(total=total, branch="single_signal_cap", detail=detail)

    r_len = 0.0
    # Loophole fix #1: R_len only counts when the evidence has already cleared both quality gates;
    # s_suf >= 1/3 is guaranteed here (WEAK_SUF_MAX gate above already returned otherwise), so this
    # reduces to gating on s_nec, but both conditions are checked explicitly to match the spec text.
    if s_suf >= WEAK_SUF_MIN_FOR_LEN and s_nec >= WEAK_SUF_MIN_FOR_LEN:
        r_len = _length_reward(judge, units, evidence_ids, prediction, approve_full, s_nec)
    detail["r_len"] = r_len

    total = 1.0 + 0.6 * q + 0.8 * s_suf + 0.8 * s_nec + 0.4 * s_consistency + 0.5 * r_div + 0.3 * r_len
    return RewardV2Breakdown(total=total, branch="eligible_multisignal", detail=detail)


WEAK_SUF_MIN_FOR_LEN = 1 / 3  # doc 4.2.2 item 1: "S_suf >= 1/3 且 S_nec >= 1/3"


# ---------------------------------------------------------------------------
# 4.3: agreement = "disagree"
# ---------------------------------------------------------------------------


def score_disagree(
    judge: DeepSeekJudge, units: list[Unit], claims: list[Claim], alt_prediction: str,
    y_hgt: str, y_true: str,
) -> RewardV2Breakdown:
    units_map = unit_map(units)
    claims_text = "\n".join(f"- {c.claim}" for c in claims)
    evidence_ids = sorted({sid for c in claims for sid in c.source_ids})
    full_text = render_units(units)
    masked_text = render_masked(units, set(evidence_ids))

    s_suf = judge.sufficiency_vote(claims_text, alt_prediction)
    approve_full = judge.label_approve_rate(full_text, alt_prediction)
    approve_masked = judge.label_approve_rate(masked_text, alt_prediction)
    s_nec = max(0.0, min(1.0, approve_full - approve_masked))

    detail = {"s_suf": s_suf, "s_nec": s_nec, "y_hgt": y_hgt, "y_true": y_true}
    if y_hgt != y_true:
        total = 2.0 + 1.0 * s_suf + 1.0 * s_nec
        return RewardV2Breakdown(total=total, branch="disagree_hgt_wrong", detail=detail)
    total = -0.5 - 1.0 * s_suf - 0.5 * s_nec
    return RewardV2Breakdown(total=total, branch="disagree_hgt_right", detail=detail)


# ---------------------------------------------------------------------------
# format-level check (programmatic, no judge call): category-description vs. hidden raw value
# ---------------------------------------------------------------------------


def bucket_from_percentile(percentile: float) -> str:
    if percentile < 10:
        return "very low"
    if percentile < 30:
        return "low"
    if percentile < 70:
        return "moderate"
    if percentile < 90:
        return "high"
    return "very high"


def check_category_consistency(stated_bucket: str, hidden_percentile: float) -> bool:
    """Section 2/4.1: a claim's categorical description must match the hidden raw percentile it
    was derived from. Purely programmatic, same layer as format_check.py's other checks."""
    return bucket_from_percentile(hidden_percentile) == stated_bucket
