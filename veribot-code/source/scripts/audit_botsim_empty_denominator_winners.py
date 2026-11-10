"""Find model evidence winners whose numeric average has no observed items."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimLiteralZeroRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/botsim_empty_denominator_winners.json"
POST_MEANS = {"avg_post_score", "avg_upvote_ratio", "avg_post_len"}
COMMENT_MEANS = {"avg_comment_score", "avg_comment_len"}


def main() -> None:
    activity_path = ROOT / "data/raw/botsim_24/user_post_comment.json"
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimLiteralZeroRuntime(ROOT, seed)
        saved_path = ROOT / f"output/v3_botsim_literal_upvote_zero_12/seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        decisions = runtime.predict_many([row["account_id"] for row in saved])
        affected = []
        for decision in decisions:
            uid = decision["account_id"]
            record = activity[uid]
            for term in decision["terms"]:
                name = term["source_fact_name"]
                missing = ((name in POST_MEANS and not record.get("posts")) or
                           (name in COMMENT_MEANS and not record.get("comment_1")))
                if missing:
                    affected.append({
                        "account_id": uid, "concept": term["name"],
                        "winning_fact": name,
                        "logit_contribution": term["logit_contribution"],
                        "prediction": decision["prediction"],
                    })
        runs.append({
            "seed": seed, "test_accounts": len(saved),
            "no_post_test_accounts": sum(not activity[row["account_id"]].get("posts")
                                         for row in saved),
            "no_comment1_test_accounts": sum(
                not activity[row["account_id"]].get("comment_1") for row in saved),
            "imputed_average_winner_slots": len(affected),
            "accounts_with_imputed_average_winner": len({row["account_id"] for row in affected}),
            "nonzero_effect_slots": sum(abs(row["logit_contribution"]) > 1e-10
                                        for row in affected),
            "winners": affected,
            "test_predictions_sha256": sha256_file(saved_path),
        })
        print({key: runs[-1][key] for key in (
            "seed", "no_post_test_accounts", "no_comment1_test_accounts",
            "imputed_average_winner_slots", "nonzero_effect_slots")}, flush=True)
    OUT.write_text(json.dumps({
        "status": "botsim_empty_denominator_concept_winners_audited",
        "raw_activity_sha256": sha256_file(activity_path),
        "population_no_posts": sum(not row.get("posts") for row in activity.values()),
        "population_no_comment1": sum(not row.get("comment_1") for row in activity.values()),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
