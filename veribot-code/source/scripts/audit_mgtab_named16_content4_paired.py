"""Compare independently replayed 16+4 MGTAB with frozen HypBot and 4+16."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    new_dir = root / "output/v3_mgtab_named16_content4/selected"
    old_dir = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    base_dir = root / "output/hypbot_mgtab_paired"
    verification = read(new_dir / "independent_verification.json")
    verified = {row["seed"]: row for row in verification["runs"]}
    runs = []
    for seed in (0, 1, 2):
        new_report = read(new_dir / f"seed{seed}" / "report.json")
        new = read(new_dir / f"seed{seed}" / "test_predictions.json")
        old = read(old_dir / f"seed{seed}" / "test_predictions.json")
        baseline = read(base_dir / f"seed{seed}" / "test_predictions.json")
        if (verified[seed]["status"] !=
                "all_published_tensor_to_evidence_to_decision_cards_replayed"
                or verified[seed]["model_report_sha256"] !=
                sha256_file(new_dir / f"seed{seed}" / "report.json")
                or new_report["manifest"] != new["manifest"]
                or new["manifest"]["split_sha256"] != old["manifest"]["split_sha256"]
                or new["manifest"]["split_sha256"] != baseline["manifest"]["split_sha256"]):
            raise ValueError("candidate replay, split or manifest differs")
        rows = [new["predictions"], old["predictions"], baseline["predictions"]]
        if not all(len(group) == 1020 for group in rows):
            raise ValueError("paired test population differs")
        maps = [{row["account_id"]: row for row in group} for group in rows]
        if not all(len(group) == 1020 for group in maps):
            raise ValueError("duplicate account in paired test")
        ids = sorted(maps[0])
        if any(sorted(group) != ids for group in maps[1:]):
            raise ValueError("paired account sets differ")
        labels = np.asarray([maps[0][uid]["label"] for uid in ids])
        if any(np.any(np.asarray([group[uid]["label"] for uid in ids]) != labels)
               for group in maps[1:]):
            raise ValueError("paired labels differ")
        thresholds = [new["manifest"]["threshold"],
                      old["manifest"]["threshold"],
                      baseline["manifest"]["threshold"]]
        correct = [(np.asarray([group[uid]["probability"] for uid in ids]) >=
                    threshold) == labels
                   for group, threshold in zip(maps, thresholds, strict=True)]
        runs.append({
            "seed": seed, "n": len(ids),
            "named16_content4_acc": float(correct[0].mean()),
            "named4_content16_acc": float(correct[1].mean()),
            "hypbot_acc": float(correct[2].mean()),
            "candidate_minus_hypbot_correct_accounts": int(
                correct[0].sum() - correct[2].sum()),
            "candidate_only_correct_vs_hypbot": int(np.sum(correct[0] & ~correct[2])),
            "hypbot_only_correct_vs_candidate": int(np.sum(correct[2] & ~correct[0])),
            "candidate_minus_current_correct_accounts": int(
                correct[0].sum() - correct[1].sum()),
            "candidate_predictions_sha256": sha256_file(
                new_dir / f"seed{seed}" / "test_predictions.json"),
            "hypbot_predictions_sha256": sha256_file(
                base_dir / f"seed{seed}" / "test_predictions.json"),
        })
    output = {"dataset": "mgtab", "scope": "same_account_fixed_test_paired",
              "selection_lock_sha256": sha256_file(
                  root / "output/v3_mgtab_named_content_budget/selection_lock_16_4.json"),
              "candidate_independent_verification_sha256": sha256_file(
                  new_dir / "independent_verification.json"),
              "candidate_selected_from_test": False,
              "candidate_replaces_current_model": False,
              "runs": runs}
    target = root / "research/mgtab_named16_content4_paired_audit_2026-11-03.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(runs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
