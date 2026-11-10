"""Detect injected Fox8 profile joins from raw records before correction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from audit_fox8_shared_source_repair import raw_counts
from verifiable_bot.fox8_profile_integrity import (
    detect_statuses_discrepancies, read_first_raw_statuses)
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "output/v3_fox8_targeted_twenty_ab_group/shared_source_repair.json"
SWEEP = ROOT / "output/v3_fox8_targeted_twenty_ab_group/top_source_donor_sweep.json"
OUT = ROOT / "research/certificates/fox8_wrong_join_discovery.json"


def main() -> None:
    shared = json.loads(SHARED.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    reports = []
    for run, donor_run in zip(shared["runs"], sweep["runs"], strict=True):
        seed = run["seed"]
        if seed != donor_run["seed"]:
            raise ValueError("seed mismatch")
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        snapshot = read_first_raw_statuses(runtime.source.raw_path,
                                           runtime.ids)
        independent, ambiguous = raw_counts(runtime.source.raw_path,
                                            runtime.ids)
        if (not np.array_equal(snapshot.values, independent)
                or list(snapshot.ambiguous_account_ids) != sorted(ambiguous)):
            raise ValueError("independent raw snapshot parsing differs")
        status_col = runtime.reference.names.index("statuses_count")
        clean = np.asarray(runtime.source.facts[:, status_col], dtype=float)
        baseline = detect_statuses_discrepancies(runtime.ids, clean,
                                                 snapshot)
        if baseline:
            raise ValueError("clean materialized source has discrepancies")
        chosen = [r["source_account_id"] for r in run["selected_sources"]]
        scenarios = []
        for scenario in run["scenarios"]:
            corrupted = clean.copy()
            for source, donor in scenario["wrong_join_donors"].items():
                corrupted[runtime.index[source]] = snapshot.values[
                    runtime.index[donor]]
            detected = detect_statuses_discrepancies(runtime.ids,
                                                     corrupted, snapshot)
            if (set(detected) != set(chosen) or len(detected) != len(chosen)
                    or any(uid in snapshot.ambiguous_account_ids
                           for uid in detected)):
                raise ValueError("source discrepancy detection differs")
            prioritized = sorted(detected, key=lambda uid: (
                -len(runtime.source.neighbors[runtime.index[uid]]), uid))
            if prioritized != chosen:
                raise ValueError("label-blind graph priority differs")
            corrupted[runtime.index[prioritized[0]]] = snapshot.values[
                runtime.index[prioritized[0]]]
            after_one = detect_statuses_discrepancies(runtime.ids,
                                                      corrupted, snapshot)
            if set(after_one) != set(chosen[1:]):
                raise ValueError("first repair did not clear one source")
            scenarios.append({
                "strategy": scenario["wrong_join_strategy"],
                "injected_records": len(chosen),
                "detected_records": len(detected),
                "false_positive_records": 0,
                "top_prioritized_source_account_id": prioritized[0],
                "remaining_detected_after_one_repair": len(after_one),
            })
        top = donor_run["top_source_account_id"]
        if top != chosen[0]:
            raise ValueError("single-source sweep uses another source")
        for donor in donor_run["donors"]:
            corrupted = clean.copy()
            corrupted[runtime.index[top]] = snapshot.values[
                runtime.index[donor["donor_account_id"]]]
            detected = detect_statuses_discrepancies(runtime.ids,
                                                     corrupted, snapshot)
            if detected != [top]:
                raise ValueError("single-source donor discrepancy missed")
        reports.append({
            "seed": seed,
            "clean_discrepancies": len(baseline),
            "ambiguous_raw_snapshot_accounts_excluded":
                list(snapshot.ambiguous_account_ids),
            "shared_scenarios": scenarios,
            "single_source_donors_detected": len(donor_run["donors"]),
        })
        print({"seed": seed, "scenarios_detected": len(scenarios),
               "donors_detected": len(donor_run["donors"])}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "all_injected_profile_joins_detected_from_raw_without_class_labels",
        "shared_result_sha256": sha256_file(SHARED),
        "sweep_result_sha256": sha256_file(SWEEP),
        "raw_profile_field": "statuses_count",
        "uses_bot_human_labels": False,
        "natural_error_detection_evaluated": False,
        "runs": reports,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
