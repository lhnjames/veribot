"""Lock the most paper-named MGTAB 20-term budget within dev tolerances."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    screen_path = root / "output/v3_mgtab_named_content_budget/dev_only.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if (screen["dataset"] != "mgtab" or screen["test_predictions_accessed"]
            or len(screen["runs"]) != 3):
        raise ValueError("dev-only budget screen differs")
    eligible = []
    tolerance_acc = 0.005
    tolerance_macro_f1 = 0.006
    for named_count in (4, 8, 12, 16, 20):
        checks = []
        for row in screen["runs"]:
            candidate = next(b for b in row["budgets"]
                             if b["paper_named_profile_or_relation_count"] == named_count)
            baseline = row["locked_4_plus_16_validation"]
            checks.append({
                "seed": row["seed"],
                "acc_loss": baseline["acc"] - candidate["validation"]["acc"],
                "macro_f1_loss": (baseline["macro_f1"] -
                                  candidate["validation"]["macro_f1"]),
            })
        if all(check["acc_loss"] <= tolerance_acc and
               check["macro_f1_loss"] <= tolerance_macro_f1
               for check in checks):
            eligible.append({"named_count": named_count, "checks": checks})
    chosen = max(eligible, key=lambda row: row["named_count"])
    if chosen["named_count"] != 16:
        raise ValueError("validation-budget selection changed")
    lock = {
        "dataset": "mgtab",
        "scope": "named16_content4_20_term_evidence_budget",
        "selected_on": "train_dev_only_before_this_candidate_test_evaluation",
        "selection_rule": "largest paper-named fact count with each seed ACC loss <=0.5pp and Macro-F1 loss <=0.6pp relative to frozen 4+16 model",
        "named_count": 16, "content_count": 4,
        "selected_l2": 1e-5,
        "validation_screen_sha256": sha256_file(screen_path),
        "validation_screen": str(screen_path),
        "per_seed_checks": chosen["checks"],
        "test_predictions_accessed_for_selection": False,
        "content_source_level": "published_averaged_tweet_tensor",
    }
    target = root / "output/v3_mgtab_named_content_budget/selection_lock_16_4.json"
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"named_count": 16, "content_count": 4,
                      "max_acc_loss_pp": max(c["acc_loss"] for c in chosen["checks"]) * 100,
                      "max_macro_f1_loss_pp": max(c["macro_f1_loss"] for c in chosen["checks"]) * 100,
                      "lock_sha256": sha256_file(target)}, indent=2))


if __name__ == "__main__":
    main()
