"""VeriRoute-Lite v3 reward: agreement-aware continuous reward v2.

The agree branch deliberately treats the upstream HGT verdict as the classification anchor and
scores evidence honesty/quality without consulting the dataset label.  The disagree branch uses
the hidden trusted label asymmetrically: well-supported correction of a wrong HGT decision is
rewarded, while contradicting a correct HGT decision is penalized.  All semantic scores are K-vote
fractions from the same DeepSeek judge configuration; format, references, digits, and qualitative
tier consistency are deterministic gates evaluated before any API call.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .format_check import check_format
from .llm_client import DeepSeekJudge
from .schema import AccountRecord
from .units import build_units, hidden_unit_metadata, render_masked, render_units, unit_map

STRONG_SUF_MIN = 2 / 3
WEAK_SUF_MAX = 1 / 3
CLAIM_APPROVE_MIN = 2 / 3
WEAK_NEC_MIN_FOR_LEN = 1 / 3
DEFAULT_AUDIT_PROB = 0.175
LENGTH_MIN_THRESHOLD = 0.05


@dataclass
class RewardBreakdown:
    total: float
    format_ok: bool
    r_cls: float | None = None
    q_claims: float | None = None
    s_suf: float | None = None
    s_nec: float | None = None
    s_consistency: float | None = None
    r_len: float | None = None
    r_hybrid: float | None = None
    reason: str = ""
    detail: dict = field(default_factory=dict)


def unit_category(unit_id: str) -> str:
    return unit_id.split("_", 1)[0]


def opposite(label: str) -> str:
    return "human" if label == "bot" else "bot"


def _necessity_at_prefix(judge, units, prediction, approve_rate_full, excluded_ids, cache):
    key = frozenset(excluded_ids)
    if key in cache:
        return cache[key]
    approve_rate_masked = judge.label_approve_rate(render_masked(units, excluded_ids), prediction)
    value = max(0.0, min(1.0, approve_rate_full - approve_rate_masked))
    cache[key] = value
    return value


def _length_reward(judge, units, evidence_ids, prediction, approve_rate_full, s_nec_full):
    n = len(evidence_ids)
    if n <= 1:
        return 1.0 if s_nec_full >= LENGTH_MIN_THRESHOLD else 0.0
    target = max(LENGTH_MIN_THRESHOLD, 0.5 * s_nec_full)
    cache = {}

    def reaches(k: int) -> bool:
        return _necessity_at_prefix(
            judge, units, prediction, approve_rate_full, set(evidence_ids[:k]), cache
        ) >= target

    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches(mid):
            hi = mid
        else:
            lo = mid + 1
    return max(0.0, min(1.0, 1.0 - (lo - 1) / n))


def _score_agree_insufficient(judge, claims_text: str, prediction: str) -> RewardBreakdown:
    audited = random.random() < DEFAULT_AUDIT_PROB
    if not audited:
        return RewardBreakdown(
            total=1.3, format_ok=True, r_cls=1.0, r_hybrid=0.3,
            reason="declared insufficient; sampled honesty audit not selected",
            detail={"branch": "insufficient_unaudited", "audited": False},
        )
    s_suf = judge.sufficiency_vote(claims_text, prediction)
    honest = s_suf < STRONG_SUF_MIN
    return RewardBreakdown(
        total=1.3 if honest else 1.0, format_ok=True, r_cls=1.0,
        s_suf=s_suf, r_hybrid=0.3 if honest else 0.0,
        reason="honest insufficient" if honest else "dishonest insufficient",
        detail={
            "branch": "insufficient_audited_honest" if honest else "insufficient_audited_dishonest",
            "audited": True,
        },
    )


def _score_agree_sufficient(judge, units, output, prediction: str) -> RewardBreakdown:
    units_map = unit_map(units)
    claims_text = "\n".join(f"- {claim.claim}" for claim in output.claims)
    s_suf = judge.sufficiency_vote(claims_text, prediction)
    if s_suf < WEAK_SUF_MAX:
        return RewardBreakdown(
            total=0.7, format_ok=True, r_cls=1.0, s_suf=s_suf, r_hybrid=-0.3,
            reason="declared sufficient but evidence is unsupported",
            detail={"branch": "sufficient_but_unsupported"},
        )

    claim_scores = []
    approved_categories = set()
    for claim in output.claims:
        score = judge.claim_approval(
            claim.claim, [units_map[source] for source in claim.source_ids], claim.pattern
        )
        claim_scores.append(score)
        if score >= CLAIM_APPROVE_MIN:
            approved_categories.update(unit_category(source) for source in claim.source_ids)
    q_claims = 0.5 * min(claim_scores) + 0.5 * (sum(claim_scores) / len(claim_scores))

    full_text = render_units(units)
    approve_full = judge.label_approve_rate(full_text, prediction)
    approve_masked = judge.label_approve_rate(
        render_masked(units, set(output.evidence_ids)), prediction
    )
    s_nec = max(0.0, min(1.0, approve_full - approve_masked))
    s_consistency = None
    if len(output.claims) >= 2:
        s_consistency = judge.consistency_vote(claims_text)
    eligible = len(output.claims) >= 2 and len(approved_categories) >= 2
    r_div = min(len(approved_categories), 3) / 3.0
    detail = {
        "s_suf": s_suf, "q_claims": q_claims, "s_nec": s_nec,
        "s_consistency": s_consistency, "approved_categories": sorted(approved_categories),
        "r_div": r_div, "eligible_multisignal": eligible,
        "approve_full": approve_full, "approve_masked": approve_masked,
    }
    if not eligible:
        total = 1.0 + 0.6 * q_claims + 0.8 * s_suf + 0.8 * s_nec
        detail["branch"] = "single_signal_cap"
        return RewardBreakdown(
            total=total, format_ok=True, r_cls=1.0, q_claims=q_claims, s_suf=s_suf,
            s_nec=s_nec, s_consistency=s_consistency, r_hybrid=total - 1.0, detail=detail,
        )

    r_len = 0.0
    if s_suf >= WEAK_SUF_MAX and s_nec >= WEAK_NEC_MIN_FOR_LEN:
        r_len = _length_reward(
            judge, units, output.evidence_ids, prediction, approve_full, s_nec
        )
    total = (
        1.0 + 0.6 * q_claims + 0.8 * s_suf + 0.8 * s_nec
        + 0.4 * float(s_consistency) + 0.5 * r_div + 0.3 * r_len
    )
    detail.update({"branch": "eligible_multisignal", "r_len": r_len})
    return RewardBreakdown(
        total=total, format_ok=True, r_cls=1.0, q_claims=q_claims, s_suf=s_suf,
        s_nec=s_nec, s_consistency=s_consistency, r_len=r_len, r_hybrid=total - 1.0,
        detail=detail,
    )


def _score_disagree(judge, units, output, y_hgt: str, y_true: str) -> RewardBreakdown:
    alternative = opposite(y_hgt)
    claims_text = "\n".join(f"- {claim.claim}" for claim in output.claims)
    s_suf = judge.sufficiency_vote(claims_text, alternative)
    approve_full = judge.label_approve_rate(render_units(units), alternative)
    approve_masked = judge.label_approve_rate(
        render_masked(units, set(output.evidence_ids)), alternative
    )
    s_nec = max(0.0, min(1.0, approve_full - approve_masked))
    hgt_wrong = y_hgt != y_true
    if hgt_wrong:
        total = 2.0 + s_suf + s_nec
        branch = "disagree_hgt_wrong"
        r_cls = 1.0
    else:
        total = -0.5 - s_suf - 0.5 * s_nec
        branch = "disagree_hgt_right"
        r_cls = -1.0
    evidence_approved = s_suf >= STRONG_SUF_MIN and s_nec >= WEAK_NEC_MIN_FOR_LEN
    return RewardBreakdown(
        total=total, format_ok=True, r_cls=r_cls, s_suf=s_suf, s_nec=s_nec,
        r_hybrid=total - (2.0 if hgt_wrong else -0.5),
        detail={
            "branch": branch, "agreement": "disagree", "y_hgt": y_hgt, "y_true": y_true,
            "derived_prediction": alternative, "derived_prediction_correct": hgt_wrong,
            "approve_full": approve_full, "approve_masked": approve_masked,
            "record_hgt_suspect": bool(hgt_wrong and evidence_approved),
        },
    )


def score_completion(record: AccountRecord, raw_text: str, judge: DeepSeekJudge) -> RewardBreakdown:
    if record.upstream_verdict not in {"bot", "human"}:
        return RewardBreakdown(total=-2.0, format_ok=False, reason="missing upstream HGT verdict")
    units = build_units(record)
    checked = check_format(raw_text, unit_map(units), hidden_unit_metadata(units))
    if not checked.ok:
        return RewardBreakdown(total=-2.0, format_ok=False, reason=checked.reason)
    output = checked.output
    if output.agreement == "agree":
        claims_text = "\n".join(f"- {claim.claim}" for claim in output.claims)
        if output.evidence_status == "insufficient":
            result = _score_agree_insufficient(judge, claims_text, record.upstream_verdict)
        else:
            result = _score_agree_sufficient(judge, units, output, record.upstream_verdict)
        result.detail.update({
            "agreement": "agree", "derived_prediction": record.upstream_verdict,
            # Logging/calibration only; this value never changes the agree-branch reward above.
            "derived_prediction_correct": record.upstream_verdict == record.label,
        })
        return result
    return _score_disagree(judge, units, output, record.upstream_verdict, record.label)
