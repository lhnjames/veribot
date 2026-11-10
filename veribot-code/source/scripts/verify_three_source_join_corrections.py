"""Replay label-blind selection and account-budget curves for three repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG = {
    "botsim": {
        "file": "output/v3_computable/botsim_group_tail_12/profile_source_join_correction.json",
        "model": "output/v3_computable/botsim_group_tail_12",
        "recipient_salt": "veribot-botsim-profile-join-recipient-v1",
        "budget_salt": "veribot-botsim-correction-order-v1",
    },
    "fox8": {
        "file": "output/v3_fox8_native_spline_l2/group_tail_20_selected/tweet_source_join_correction.json",
        "model": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
        "recipient_salt": "veribot-fox8-tweet-join-recipient-v1",
        "budget_salt": "veribot-fox8-correction-order-v1",
    },
    "mgtab": {
        "file": "output/v3_mgtab_ab_spline_content/direct_selected_20/content_source_join_correction.json",
        "model": "output/v3_mgtab_ab_spline_content/direct_selected_20",
        "recipient_salt": "veribot-mgtab-content-join-recipient-v1",
        "budget_salt": "veribot-mgtab-correction-order-v1",
    },
}


def sorted_ids(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda uid: hashlib.sha256(
        f"{salt}:{uid}".encode()).digest())


def verify_dataset(root: Path, name: str, config: dict) -> dict:
    source = json.loads((root / config["file"]).read_text(encoding="utf-8"))
    if (source["dataset"] != name or source["injection_uses_bot_human_labels"]
            or source["account_selection_uses_bot_human_labels"]
            or source["source_correction_uses_bot_human_labels"]):
        raise ValueError(f"{name}: source-correction scope differs")
    runs = []
    for run in source["runs"]:
        seed = run["seed"]
        predictions = json.loads((root / config["model"] / f"seed{seed}" /
                                  "test_predictions.json").read_text(encoding="utf-8"))
        all_ids = [row["account_id"] for row in predictions["predictions"]]
        selected_ids = sorted_ids(all_ids, config["recipient_salt"])[
            :round(len(all_ids) * 0.25)]
        rows = run["rows"]
        if ([row["account_id"] for row in rows] != selected_ids
                or len(rows) != run["selected_accounts"]
                or run["test_accounts"] != len(all_ids)
                or run["split_sha256"] != predictions["manifest"]["split_sha256"]):
            raise ValueError(f"{name} seed {seed}: label-blind recipient selection differs")
        threshold = run["threshold"]
        labels = np.asarray([row["label_for_evaluation"] for row in rows], dtype=int)
        clean = np.asarray([row["clean_probability"] for row in rows], dtype=float)
        bad = np.asarray([row["corrupted_probability"] for row in rows], dtype=float)
        repaired = np.asarray([row["repaired_probability"] for row in rows], dtype=float)
        clean_error = (clean >= threshold) != labels
        bad_error = (bad >= threshold) != labels
        outside_errors = run["clean_errors"] - int(clean_error.sum())
        if (outside_errors < 0
                or outside_errors + int(bad_error.sum()) != run["corrupted_errors"]
                or outside_errors + int(((repaired >= threshold) != labels).sum()) !=
                run["source_corrected_errors"]
                or np.max(np.abs(repaired - clean)) > 1e-12
                or run["corruption_changed_predictions"] != int(np.count_nonzero(
                    (bad >= threshold) != (clean >= threshold)))
                or run["corruption_new_errors"] != int(np.sum(bad_error & ~clean_error))
                or run["corruption_removed_errors"] != int(np.sum(clean_error & ~bad_error))):
            raise ValueError(f"{name} seed {seed}: correction effect differs")
        orderings = {
            "hash_random": sorted(range(len(rows)), key=lambda i: hashlib.sha256(
                f"{config['budget_salt']}:{selected_ids[i]}".encode()).digest()),
            "model_uncertainty": sorted(range(len(rows)),
                                        key=lambda i: (abs(bad[i] - threshold),
                                                       selected_ids[i])),
        }
        for strategy, points in run["curves"].items():
            if strategy not in orderings:
                raise ValueError("unexpected correction strategy")
            for point in points:
                selected = orderings[strategy][:point["account_actions"]]
                corrected = bad_error.copy()
                corrected[selected] = clean_error[selected]
                direct = bad_error.copy()
                direct[selected] = False
                if (point["source_correction_total_errors"] !=
                        outside_errors + int(corrected.sum())
                        or point["direct_account_label_total_errors"] !=
                        outside_errors + int(direct.sum())
                        or point["source_correction_net_errors_removed"] !=
                        int(bad_error.sum() - corrected.sum())
                        or point["direct_account_label_net_errors_removed"] !=
                        int(bad_error.sum() - direct.sum())):
                    raise ValueError(f"{name} seed {seed}: budget curve differs")
        runs.append({"seed": seed, "test_accounts": len(all_ids),
                     "selected_accounts": len(rows),
                     "clean_errors": run["clean_errors"],
                     "corrupted_errors": run["corrupted_errors"],
                     "source_corrected_errors": run["source_corrected_errors"],
                     "verified_budget_strategies": len(orderings),
                     "status": "label_blind_selection_and_budget_curves_replayed"})
    if [row["seed"] for row in runs] != [0, 1, 2]:
        raise ValueError(f"{name}: fixed seeds differ")
    return {"dataset": name, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = [verify_dataset(args.project_root, name, config)
                for name, config in CONFIG.items()]
    result = {"scope": "three_dataset_known_source_join_repair_curve_replay",
              "datasets": datasets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
