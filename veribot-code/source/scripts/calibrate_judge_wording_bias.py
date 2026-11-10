"""Step 1 (design doc section 4.2): measure whether claim_approval / sufficiency_vote systematically
favor precise-number wording over calibrated-descriptor wording, isolating ONLY the presence/absence
of a raw digit while holding everything else (including the decision-relevant tier framing) fixed.

History of this script's design, each superseded by a review finding:
  v0: both number and bucket claim checked against the SAME bare-number source -> confounded a
      grounding rejection (bucket claim asserts a percentile not present in a bare number) with
      wording bias.
  v1: separate matched (source, claim) pairs, but bucket_source still spelled out a percentile in
      digits ("lower than roughly 90%...") -> violated the v3 target format (bucket text must have
      zero digits) and wasn't actually testing the right two conditions.
  v2 (digit-free bucket side): removed all digits from the bucket side. claim_approval became well
      calibrated, but sufficiency_vote showed a large, direction-consistent gap. Root cause found by
      direct probe: the numeric claim ("The account has 890000 followers.") carries ONLY a raw
      magnitude, with no dataset-relative framing, while the bucket claim ("...is unusually high.")
      is *by construction* a pre-judged, dataset-relative claim. That is a real information-content
      difference, not a pure wording-form difference -- so v2 was not isolating the variable it
      claimed to isolate.
  v3 (this version): BOTH numeric_claim and bucket_claim assert the SAME fixed tier word (e.g. "is
      an extreme imbalance" / "is unusually high" / "is a moderate burst") -- the ONLY difference
      between the two pipelines is whether a raw figure is additionally present. This isolates
      "does stating a digit change the verdict" while holding the decision-relevant semantic framing
      constant, which is the actual question 4.2 is about.

Two negative controls per pair:
  - fabricated_number_claim: tier word is correct, but the appended raw figure is NOT the one in
    numeric_source -> should be REJECTed (tests the number is actually checked, not just present)
  - unsupported_bucket_claim: asserts a DIFFERENT tier word than bucket_source states -> should be
    REJECTed (tests tier claims are still grounding-checked, not rubber-stamped)

Reports mean AND max |gap| per pair -- a near-zero signed mean can hide large per-pair spread that
happens to cancel out, so signed mean alone is not sufficient evidence of "no bias."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.llm_client import DeepSeekJudge

# numeric_source/numeric_claim state the SAME tier word as bucket_source/bucket_claim, plus a raw
# figure. bucket_source/bucket_claim/unsupported_bucket_claim contain zero digits. The only
# difference between the numeric and bucket pipelines is the presence of the raw figure.
PAIRS = [
    dict(
        name="ratio_extreme",
        numeric_source="Following-to-follower ratio: extreme imbalance (following is "
                       "approximately 46.7 times the follower count).",
        numeric_claim="The account's following-to-follower ratio is an extreme imbalance, with "
                      "following approximately 46.7 times the follower count.",
        bucket_source="Following-to-follower ratio: extreme imbalance.",
        bucket_claim="The account's following-to-follower ratio is an extreme imbalance.",
        fabricated_number_claim="The account's following-to-follower ratio is an extreme "
                                 "imbalance, with following approximately 120 times the follower "
                                 "count.",
        unsupported_bucket_claim="The account's following-to-follower ratio is a mild imbalance.",
        pattern="Following far exceeds followers",
    ),
    dict(
        name="ratio_mild",
        numeric_source="Following-to-follower ratio: mild imbalance (following is approximately "
                       "2.3 times the follower count).",
        numeric_claim="The account's following-to-follower ratio is a mild imbalance, with "
                      "following approximately 2.3 times the follower count.",
        bucket_source="Following-to-follower ratio: mild imbalance.",
        bucket_claim="The account's following-to-follower ratio is a mild imbalance.",
        fabricated_number_claim="The account's following-to-follower ratio is a mild imbalance, "
                                 "with following approximately 9.1 times the follower count.",
        unsupported_bucket_claim="The account's following-to-follower ratio is an extreme "
                                  "imbalance.",
        pattern="Following mildly exceeds followers",
    ),
    dict(
        name="follower_very_low",
        numeric_source="Follower count: unusually low (3 followers).",
        numeric_claim="The account's follower count is unusually low, with only 3 followers.",
        bucket_source="Follower count: unusually low.",
        bucket_claim="The account's follower count is unusually low.",
        fabricated_number_claim="The account's follower count is unusually low, with only 41 "
                                 "followers.",
        unsupported_bucket_claim="The account's follower count is unusually high.",
        pattern="Very low follower count",
    ),
    dict(
        name="follower_very_high",
        numeric_source="Follower count: unusually high (890000 followers).",
        numeric_claim="The account's follower count is unusually high, with 890000 followers.",
        bucket_source="Follower count: unusually high.",
        bucket_claim="The account's follower count is unusually high.",
        fabricated_number_claim="The account's follower count is unusually high, with 12000 "
                                 "followers.",
        unsupported_bucket_claim="The account's follower count is unusually low.",
        pattern="Very high follower count",
    ),
    dict(
        name="tweet_count_high",
        numeric_source="Tweet count: unusually high (128000 tweets).",
        numeric_claim="The account's total tweet count is unusually high, at 128000 tweets.",
        bucket_source="Tweet count: unusually high.",
        bucket_claim="The account's total tweet count is unusually high.",
        fabricated_number_claim="The account's total tweet count is unusually high, at 999999 "
                                 "tweets.",
        unsupported_bucket_claim="The account's total tweet count is moderate.",
        pattern="Very high posting volume",
    ),
    dict(
        name="listed_count_low",
        numeric_source="Listed count: low (0 lists).",
        numeric_claim="The account's listed count is low, appearing on 0 public lists.",
        bucket_source="Listed count: low.",
        bucket_claim="The account's listed count is low.",
        fabricated_number_claim="The account's listed count is low, appearing on 6 public lists.",
        unsupported_bucket_claim="The account's listed count is high.",
        pattern="Low list membership",
    ),
    dict(
        name="burst_moderate",
        numeric_source="Posting burst intensity: moderate burst (14 posts within a 15-minute "
                       "interval).",
        numeric_claim="A moderate-intensity posting burst occurred, with 14 posts within a "
                      "15-minute window.",
        bucket_source="Posting burst intensity: moderate burst.",
        bucket_claim="A moderate-intensity posting burst occurred.",
        fabricated_number_claim="A moderate-intensity posting burst occurred, with 22 posts "
                                 "within a 15-minute window.",
        unsupported_bucket_claim="A high-intensity posting burst occurred.",
        pattern="Moderate posting burst",
    ),
    dict(
        name="burst_high",
        numeric_source="Posting burst intensity: high-intensity burst (37 posts within a "
                       "15-minute interval).",
        numeric_claim="A high-intensity posting burst occurred, with 37 posts within a 15-minute "
                      "window.",
        bucket_source="Posting burst intensity: high-intensity burst.",
        bucket_claim="A high-intensity posting burst occurred.",
        fabricated_number_claim="A high-intensity posting burst occurred, with 9 posts within a "
                                 "15-minute window.",
        unsupported_bucket_claim="A low-level posting burst occurred.",
        pattern="High-intensity posting burst",
    ),
    dict(
        name="following_count_moderate",
        numeric_source="Following count: moderate (512 accounts followed).",
        numeric_claim="The account's following count is moderate, following 512 accounts.",
        bucket_source="Following count: moderate.",
        bucket_claim="The account's following count is moderate.",
        fabricated_number_claim="The account's following count is moderate, following 9,800 "
                                 "accounts.",
        unsupported_bucket_claim="The account's following count is unusually high.",
        pattern="Moderate following count",
    ),
    dict(
        name="active_years_short",
        numeric_source="Account age: unusually short (0.4 years).",
        numeric_claim="The account's account age is unusually short, at 0.4 years.",
        bucket_source="Account age: unusually short.",
        bucket_claim="The account's account age is unusually short.",
        fabricated_number_claim="The account's account age is unusually short, at 0.05 years.",
        unsupported_bucket_claim="The account's account age is unusually long.",
        pattern="Very short account age",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-cache", type=Path, required=True)
    ap.add_argument("--prediction", default="bot")
    args = ap.parse_args()

    judge = DeepSeekJudge(cache_dir=args.judge_cache, k=5, max_workers=16)

    header = (f"{'pair':26s} {'ca_num':>7s} {'ca_buck':>7s} {'ca_gap':>7s}  "
              f"{'ca_fab':>7s} {'ca_unsup':>8s}   "
              f"{'suf_num':>7s} {'suf_buck':>8s} {'suf_gap':>7s}")
    print(header)
    ca_gaps, suf_gaps = [], []
    ca_fab_scores, ca_unsup_scores = [], []
    for p in PAIRS:
        ca_num = judge.claim_approval(p["numeric_claim"], [p["numeric_source"]], p["pattern"])
        ca_buck = judge.claim_approval(p["bucket_claim"], [p["bucket_source"]], p["pattern"])
        ca_fab = judge.claim_approval(p["fabricated_number_claim"], [p["numeric_source"]], p["pattern"])
        ca_unsup = judge.claim_approval(p["unsupported_bucket_claim"], [p["bucket_source"]], p["pattern"])
        suf_num = judge.sufficiency_vote(f"- {p['numeric_claim']}", args.prediction)
        suf_buck = judge.sufficiency_vote(f"- {p['bucket_claim']}", args.prediction)
        ca_gap = ca_num - ca_buck
        suf_gap = suf_num - suf_buck
        ca_gaps.append(ca_gap)
        suf_gaps.append(suf_gap)
        ca_fab_scores.append(ca_fab)
        ca_unsup_scores.append(ca_unsup)
        print(f"{p['name']:26s} {ca_num:7.2f} {ca_buck:7.2f} {ca_gap:+7.2f}  "
              f"{ca_fab:7.2f} {ca_unsup:8.2f}   "
              f"{suf_num:7.2f} {suf_buck:8.2f} {suf_gap:+7.2f}")

    n = len(PAIRS)
    mean_ca_gap = sum(ca_gaps) / n
    mean_abs_ca_gap = sum(abs(g) for g in ca_gaps) / n
    max_abs_ca_gap = max(abs(g) for g in ca_gaps)
    worst_ca_pair = PAIRS[max(range(n), key=lambda i: abs(ca_gaps[i]))]["name"]
    mean_suf_gap = sum(suf_gaps) / n
    mean_abs_suf_gap = sum(abs(g) for g in suf_gaps) / n
    max_abs_suf_gap = max(abs(g) for g in suf_gaps)
    worst_suf_pair = PAIRS[max(range(n), key=lambda i: abs(suf_gaps[i]))]["name"]
    print("\n===== AGGREGATE (tier word held fixed; ONLY raw-figure presence differs) =====")
    print(f"claim_approval:   mean signed gap = {mean_ca_gap:+.3f}; mean |gap| = {mean_abs_ca_gap:.3f}; "
          f"MAX |gap| = {max_abs_ca_gap:.3f} (worst pair: {worst_ca_pair})  (n={n}, K=5 each)")
    print(f"sufficiency_vote: mean signed gap = {mean_suf_gap:+.3f}; mean |gap| = {mean_abs_suf_gap:.3f}; "
          f"MAX |gap| = {max_abs_suf_gap:.3f} (worst pair: {worst_suf_pair})")
    print("\n===== NEGATIVE CONTROLS (should be low) =====")
    print(f"fabricated_number_claim mean claim_approval = {sum(ca_fab_scores)/n:.3f} "
          f"(ideal: near 0 -- number contradicts numeric_source)")
    print(f"unsupported_bucket_claim mean claim_approval = {sum(ca_unsup_scores)/n:.3f} "
          f"(ideal: near 0 -- tier contradicts bucket_source)")
    print("\n(positive gap = judge favors the claim WITH the raw figure over the claim without one, "
          "for the SAME stated tier word. Report MAX |gap|, not just mean signed gap.)")


if __name__ == "__main__":
    main()
