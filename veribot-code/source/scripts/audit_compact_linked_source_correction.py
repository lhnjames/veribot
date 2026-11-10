"""Independently check source truth and action curves in linked repair reports."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fox_path = root / "output/v3_fox8_targeted_neighbor_20_concepts/linked_source_correction.json"
    mgtab_path = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree/linked_profile_source_correction.json"
    fox = json.loads(fox_path.read_text(encoding="utf-8"))
    mgtab = json.loads(mgtab_path.read_text(encoding="utf-8"))
    raw_path = root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    if fox["raw_source_truth_sha256"] != sha256_file(raw_path):
        raise ValueError("Fox8 raw source changed")
    fox_truth = {}
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            uid = str(record["user_id"])
            fox_truth.setdefault(uid, (number, float(record["user_tweets"][0]
                                                   ["user"].get("statuses_count", 0) or 0)))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    for filename, digest in mgtab["source_files_sha256"].items():
        if digest != sha256_file(data_dir / filename):
            raise ValueError("MGTAB published source changed")
    features = torch.load(data_dir / "features.pt", map_location="cpu",
                          weights_only=True).numpy()
    summary = []
    for dataset, output in (("fox8", fox), ("mgtab", mgtab)):
        for seed, run in enumerate(output["runs"]):
            if run["seed"] != seed:
                raise ValueError("Repair seed order differs")
            rows = run["source_rows"]
            if len(rows) != (run["wrong_source_record_count"] if dataset == "fox8"
                             else run["wrong_profile_rows"]):
                raise ValueError("Corrupted source row count differs")
            if len({row.get("source_account_id", row.get("account_id")) for row in rows}) != len(rows):
                raise ValueError("Corrupted source rows are duplicated")
            for row in rows:
                if dataset == "fox8":
                    original = fox_truth[row["source_account_id"]]
                    donor = fox_truth[row["wrong_join_donor_id"]]
                    if (row["raw_gzip_line"] != original[0]
                            or row["source_statuses_count"] != original[1]
                            or row["wrong_join_statuses_count"] != donor[1]):
                        raise ValueError("Fox8 source row disagrees with original gzip")
                else:
                    account, donor = int(row["account_id"]), int(row["wrong_donor_id"])
                    if (row["published_tensor_row"] != account
                            or row["source_ratio"] != float(features[account, 14])
                            or row["wrong_join_ratio"] != float(features[donor, 14])):
                        raise ValueError("MGTAB source row disagrees with published tensor")
            curve = run["budget_curve"]
            if (curve[0]["source_record_actions"] != 0
                    or curve[0]["source_corrected_errors"] != run["corrupted_errors"]
                    or curve[-1]["source_record_actions"] != len(rows)
                    or curve[-1]["source_corrected_errors"] != run["clean_errors"]
                    or run["fully_source_corrected_errors"] != run["clean_errors"]
                    or run["corruption_new_errors"] - run["corruption_removed_errors"] !=
                    run["corrupted_errors"] - run["clean_errors"]):
                raise ValueError("Repair action curve endpoints differ")
            previous = -1
            for point in curve:
                budget = point["source_record_actions"]
                if (budget <= previous or budget != point["direct_account_label_actions"]
                        or not 0 <= point["direct_labeled_errors"] <= run["test_accounts"]
                        or not 0 <= point["source_corrected_errors"] <= run["test_accounts"]):
                    raise ValueError("Repair budget curve is malformed")
                previous = budget
            fixed_path = (root / f"output/v3_fox8_targeted_neighbor_20_concepts/seed{seed}/ab_group_tree_test_predictions.json"
                          if dataset == "fox8" else
                          root / f"output/v3_mgtab_targeted_20_concept_ab_group_tree/seed{seed}/test_predictions.json")
            fingerprint = (run["fixed_test_prediction_sha256"] if dataset == "fox8"
                           else run["test_prediction_sha256"])
            if fingerprint != sha256_file(fixed_path):
                raise ValueError("Frozen fixed-test prediction artifact changed")
            prediction = json.loads(fixed_path.read_text(encoding="utf-8"))
            expected_clean = sum((p["probability"] >= prediction["manifest"]["threshold"])
                                 != p["label"] for p in prediction["predictions"])
            if expected_clean != run["clean_errors"]:
                raise ValueError("Clean repair endpoint differs from frozen test")
            summary.append({"dataset": dataset, "seed": seed,
                            "source_records_checked": len(rows),
                            "known_error_net_increase": run["corrupted_errors"] -
                                run["clean_errors"],
                            "linked_neighbor_fact_cells_changed": (
                                run["linked_neighbor_fact_cells_changed"]
                                if dataset == "mgtab" else None),
                            "status": "source_truth_and_action_endpoints_verified"})
    out = {
        "scope": "compact_Fox8_and_MGTAB_linked_source_repair_report_integrity",
        "fox8_report_sha256": sha256_file(fox_path),
        "mgtab_report_sha256": sha256_file(mgtab_path),
        "runs": summary,
        "status": "source_rows_frozen_predictions_and_repair_endpoints_verified",
    }
    path = root / "research/certificates/compact_linked_source_correction_audit.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
