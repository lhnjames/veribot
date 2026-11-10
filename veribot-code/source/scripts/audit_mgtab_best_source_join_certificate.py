"""Independently recount the frozen-best MGTAB source-correction certificate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = ROOT / "research/certificates/mgtab_best_source_join_correction.json"
OUTPUT = ROOT / "research/certificates/mgtab_best_source_join_correction_audit.json"


def hash_order(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda uid: hashlib.sha256(
        f"{salt}:{uid}".encode()).digest())


def main() -> None:
    source = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    if (source["status"] != "frozen_best_mgtab_known_content_row_join_repair" or
            len(source["runs"]) != 3):
        raise ValueError("correction certificate differs")
    audited = []
    for run in source["runs"]:
        seed = run["seed"]
        saved_path = ROOT / ("output/v3_mgtab_ab_group_additive_tree/"
                             f"seed{seed}/test_predictions.json")
        if sha256_file(saved_path) != run["saved_predictions_sha256"]:
            raise ValueError("frozen saved predictions changed")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        all_rows = saved["predictions"]
        ids = [row["account_id"] for row in all_rows]
        if (len(ids) != run["test_accounts"] or
                saved["manifest"]["split_sha256"] != run["split_sha256"] or
                saved["manifest"]["model_sha256"] != run["model_sha256"]):
            raise ValueError("frozen test population differs")
        by_id = {row["account_id"]: row for row in all_rows}
        selected_ids = hash_order(
            ids, "veribot-mgtab-best-content-recipient-v1")[:
                run["injected_accounts"]]
        rows = run["rows"]
        if ([row["account_id"] for row in rows] != selected_ids or
                len(set(selected_ids)) != len(rows)):
            raise ValueError("recipient hash selection differs")
        threshold = float(saved["manifest"]["threshold"])
        clean_errors = 0
        corrupt_errors = 0
        class_changes = 0
        new_errors = 0
        removed_errors = 0
        row_by_id = {row["account_id"]: row for row in rows}
        for frozen in all_rows:
            uid = frozen["account_id"]
            label = int(frozen["label"])
            clean_p = float(frozen["probability"])
            injected = row_by_id.get(uid)
            bad_p = clean_p if injected is None else float(
                injected["corrupted_probability"])
            if injected is not None:
                if (float(injected["clean_probability"]) != clean_p or
                        float(injected["repaired_probability"]) != clean_p or
                        injected["source_row"] != int(uid) or
                        injected["wrong_join_source_row"] == int(uid) or
                        str(injected["wrong_join_source_row"]) not in by_id):
                    raise ValueError("source row or clean/repaired probability differs")
            clean_wrong = int(clean_p >= threshold) != label
            bad_wrong = int(bad_p >= threshold) != label
            clean_errors += clean_wrong
            corrupt_errors += bad_wrong
            class_changes += (clean_p >= threshold) != (bad_p >= threshold)
            new_errors += bad_wrong and not clean_wrong
            removed_errors += clean_wrong and not bad_wrong
        if (clean_errors != run["clean_errors"] or
                clean_errors != run["repaired_errors"] or
                corrupt_errors != run["corrupted_errors"] or
                class_changes != run["corruption_changed_classes"] or
                new_errors != run["corruption_new_errors"] or
                removed_errors != run["corruption_removed_errors"]):
            raise ValueError("corrected class/error counts differ")
        for strategy, points in run["curves"].items():
            queue = (hash_order(selected_ids,
                     "veribot-mgtab-best-repair-order-v1")
                     if strategy == "hash_random" else
                     sorted(selected_ids, key=lambda uid: (
                         abs(float(row_by_id[uid]["corrupted_probability"]) -
                             threshold), uid)))
            for point in points:
                chosen = set(queue[:point["account_actions"]])
                repaired_count = corrupt_errors
                labeled_count = corrupt_errors
                for uid in chosen:
                    frozen = by_id[uid]
                    row = row_by_id[uid]
                    label = int(frozen["label"])
                    clean_wrong = int(frozen["probability"] >= threshold) != label
                    bad_wrong = int(row["corrupted_probability"] >= threshold) != label
                    repaired_count += clean_wrong - bad_wrong
                    labeled_count -= bad_wrong
                if (repaired_count != point["source_corrected_total_errors"] or
                        labeled_count != point["direct_account_labeled_total_errors"]):
                    raise ValueError("budget curve does not match account rows")
        audited.append({"seed": seed, "test_accounts": len(ids),
                        "injected_accounts": len(rows),
                        "clean_errors": clean_errors,
                        "corrupted_errors": corrupt_errors,
                        "corrected_errors": clean_errors,
                        "budget_points_checked": sum(len(p) for p in
                                                     run["curves"].values())})
    result = {
        "status": "independent_account_and_budget_recount_passed",
        "source_correction_certificate_sha256": sha256_file(CERTIFICATE),
        "scope": "frozen saved test labels/probabilities, source row IDs, account-level injected probabilities, and all budget points; source-level projection independently checked by generation script",
        "runs": audited,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
