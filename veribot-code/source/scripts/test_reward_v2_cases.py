"""Offline unit test for the v2 reward formula (reward_v2_formula.py), per design doc section 6.

Builds the 7 hand-constructed test cases (A-G) against two real TwiBot-20 training accounts,
scores them with the real DeepSeek judge (disk-cached, same llm_client.py used in training), and
checks the relative-ordering invariants the design doc asks for:

  A > B, A > C, E > D (D <= E), disagree(HGT wrong) > disagree(HGT right), G == -2 always.

This performs no model training and no rollout generation -- only judge calls, on the order of a
few dozen total (each K=5 internally, disk-cached).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.llm_client import DeepSeekJudge
from verifiable_bot.schema import AccountRecord, Claim
from verifiable_bot.units import build_units

from reward_v2_formula import (
    check_category_consistency,
    score_agree_insufficient,
    score_agree_sufficient,
    score_disagree,
)

DATA_PATH = "/path/to/data_root/outputs/unified_full/train_twibot20_only.jsonl"
BOT_ID = "843514885644271616"     # strong multi-signal bot: extreme follow/follower skew + near-duplicate tweets
HUMAN_ID = "1273053994986934273"  # genuinely thin record: 1 tweet, empty bio, tiny follow counts


def load_record(account_id: str) -> AccountRecord:
    with open(DATA_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["account_id"] == account_id:
                return AccountRecord.model_validate(row)
    raise KeyError(account_id)


def find_unit(units, predicate) -> str:
    for u in units:
        if predicate(u):
            return u.unit_id
    raise LookupError(f"no unit matched predicate among {[u.unit_id for u in units]}")


def main() -> None:
    judge = DeepSeekJudge(cache_dir=Path(__file__).resolve().parents[1] / "cache" / "judge_reward_v2_test")

    bot_record = load_record(BOT_ID)
    human_record = load_record(HUMAN_ID)
    bot_units = build_units(bot_record)
    human_units = build_units(human_record)

    ratio_id = find_unit(bot_units, lambda u: u.unit_id.startswith("OBS") and "times the follow" in u.text)
    verified_id = find_unit(bot_units, lambda u: u.unit_id.startswith("PROFILE") and u.text.startswith("verified"))
    # pick the two TWEET ids the OBS similarity fact names, so a claim can cite the tweets directly
    sim_obs = find_unit(bot_units, lambda u: u.unit_id.startswith("OBS") and "similarity above" in u.text)
    sim_text = next(u.text for u in bot_units if u.unit_id == sim_obs)
    import re
    tweet_ids = re.findall(r"TWEET_\d+", sim_text)
    assert len(tweet_ids) == 2, sim_text

    results: dict[str, object] = {}

    # ---- Case A: genuine multi-signal strong evidence (3 categories: OBS, PROFILE, TWEET) ----
    # Wording matters to the judge: vague paraphrase ("roughly six times") was found insufficient in
    # calibration, while pointed, concrete numbers were consistently approved -- so all claims here
    # state exact figures, while staying extractive (no "therefore this is a bot" language).
    claims_a = [
        Claim(source_ids=[ratio_id],
              claim="The account follows 3639 accounts but has only 598 followers, a following-to-follower "
                    "ratio above 6x.",
              pattern="Following far exceeds followers"),
        Claim(source_ids=[verified_id],
              claim="The account is not verified, and its bio is a single generic promotional line.",
              pattern="Unverified account status"),
        Claim(source_ids=tweet_ids,
              claim=f"{tweet_ids[0]} and {tweet_ids[1]} are near-verbatim duplicates of each other, indicating "
                    "templated/copy-paste posting rather than individually composed content.",
              pattern="Repetitive templated tweet content"),
    ]
    results["A_multisignal"] = score_agree_sufficient(judge, bot_units, claims_a, "bot")

    # ---- Case B: single OBS-tier signal, written cleanly, declared sufficient ----
    claims_b = [
        Claim(source_ids=[ratio_id],
              claim="The account follows far more accounts than follow it back, with a following-to-follower "
                    "ratio above 6x, a pattern associated with indiscriminate mass-following.",
              pattern="Following far exceeds followers"),
    ]
    results["B_single_signal"] = score_agree_sufficient(judge, bot_units, claims_b, "bot")

    # ---- Case C: one incoherent PROFILE+TWEET mashup claim, plus one normal (concretely-worded) claim ----
    desc_id = find_unit(bot_units, lambda u: u.unit_id.startswith("PROFILE") and u.text.startswith("description"))
    claims_c = [
        Claim(source_ids=[desc_id, tweet_ids[0]],
              claim=("The bio advertises discount codes, and combined with one unrelated tweet this shows the "
                     "account's follow ratio is inconsistent with its listed count."),
              pattern="Cross-field narrative inconsistency"),
        Claim(source_ids=[ratio_id],
              claim="The account follows 3639 accounts but has only 598 followers, a following-to-follower "
                    "ratio above 6x.",
              pattern="Following far exceeds followers"),
    ]
    results["C_mashup"] = score_agree_sufficient(judge, bot_units, claims_c, "bot")

    # ---- Case D: dishonest insufficient (evidence is actually strong; audit forced on) ----
    claims_d_text = "\n".join(f"- {c.claim}" for c in claims_a)
    results["D_dishonest_insufficient"] = score_agree_insufficient(
        judge, claims_d_text, "bot", force_audit=True,
    )

    # ---- Case E: honest insufficient (evidence is genuinely thin; audit forced on) ----
    # NOTE: an earlier version of this case cited followers_count/following_count (3 followers, 17
    # following) as the "thin" evidence. That was found (calibration review) to NOT reliably be
    # non-diagnostic: a judge asked "does a small follow footprint alone justify calling this
    # account human" approves often enough (s_suf swung between 0.33 and 0.8 across K=3/K=5 reruns)
    # that it isn't a stable example of "genuinely insufficient" evidence -- small-but-nonzero
    # follow counts DO carry some discriminative signal a judge can lean on. Replaced with two facts
    # that are close to maximally non-diagnostic and equally consistent with either label: "not
    # verified" (true of the vast majority of both bot and human accounts) and "empty bio" (common
    # in both populations, not itself a bot signal). Verified low and stable at K=5 with a direct
    # probe (scripts/calibrate_necessity_sensitivity.py's sibling ad-hoc check, s_suf=0.00) before
    # being wired in here -- see outputs/veriroute_lite/audit_v3/STEP1_AUDIT_REPORT.md section 4b.
    human_verified_id = find_unit(human_units, lambda u: u.unit_id.startswith("PROFILE") and u.text.startswith("verified"))
    human_desc_id = find_unit(human_units, lambda u: u.unit_id.startswith("PROFILE") and u.text.startswith("description"))
    assert human_units[[u.unit_id for u in human_units].index(human_desc_id)].text == "description: ", \
        "case E relies on this account's bio actually being empty"
    claims_e_text = "- The account is not verified.\n- The account has an empty bio."
    results["E_honest_insufficient"] = score_agree_insufficient(
        judge, claims_e_text, "human", force_audit=True,
    )

    # ---- Case F: disagree, HGT was actually right (bot account, model confidently dissents to "human") ----
    claims_f = [
        Claim(source_ids=[desc_id],
              claim="The account's bio offers real-world discount codes for restaurants and stores.",
              pattern="Locally-oriented promotional bio content"),
    ]
    results["F_disagree_hgt_right"] = score_disagree(
        judge, bot_units, claims_f, "human", y_hgt="bot", y_true="bot",
    )

    # ---- Case F2: disagree, HGT was actually wrong (thin human account misclassified as bot by HGT) ----
    fcount_id = find_unit(human_units, lambda u: u.unit_id.startswith("PROFILE") and u.text.startswith("followers_count"))
    following_id = find_unit(human_units, lambda u: u.unit_id.startswith("PROFILE") and u.text.startswith("following_count"))
    claims_f2 = [
        Claim(source_ids=[fcount_id, following_id],
              claim="The account follows 17 accounts and has 3 followers, a small personal-scale footprint.",
              pattern="Small-scale personal follow footprint"),
    ]
    results["F2_disagree_hgt_wrong"] = score_disagree(
        judge, human_units, claims_f2, "human", y_hgt="bot", y_true="human",
    )

    # ---- Case G: category-description vs hidden-raw-value mismatch (pure format check, no judge call) ----
    stated_bucket = "very low"       # claim says "extremely low follower count"
    hidden_percentile = 50.0         # but the account is actually mid-percentile
    g_ok = check_category_consistency(stated_bucket, hidden_percentile)
    results["G_category_mismatch"] = {
        "total": -2.0 if not g_ok else "FORMAT CHECK BUG: should have failed",
        "branch": "invalid_format" if not g_ok else "BUG",
        "detail": {"stated_bucket": stated_bucket, "hidden_percentile": hidden_percentile},
    }

    print("\n===== RESULTS =====")
    for name, r in results.items():
        if isinstance(r, dict):
            total, branch, detail = r["total"], r["branch"], r["detail"]
        else:
            total, branch, detail = r.total, r.branch, r.detail
        print(f"{name:28s} total={total!s:>8s} branch={branch:28s} {detail}")

    def total_of(name: str) -> float:
        r = results[name]
        return r["total"] if isinstance(r, dict) else r.total

    print("\n===== INVARIANT CHECKS =====")
    checks = [
        ("A > B", total_of("A_multisignal") > total_of("B_single_signal")),
        ("A > C", total_of("A_multisignal") > total_of("C_mashup")),
        ("E >= D", total_of("E_honest_insufficient") >= total_of("D_dishonest_insufficient")),
        ("E > D (strict)", total_of("E_honest_insufficient") > total_of("D_dishonest_insufficient")),
        ("disagree(HGT wrong) > disagree(HGT right)",
         total_of("F2_disagree_hgt_wrong") > total_of("F_disagree_hgt_right")),
        ("G == -2", total_of("G_category_mismatch") == -2.0),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        all_ok = all_ok and ok
    print(f"\nOVERALL: {'ALL INVARIANTS HOLD' if all_ok else 'AT LEAST ONE INVARIANT FAILED'}")


if __name__ == "__main__":
    main()
