"""Independently replay every Fox8 donor's raw-source decision response."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from audit_fox8_shared_source_repair import independent_score, raw_counts
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "output/v3_fox8_targeted_twenty_ab_group/top_source_donor_sweep.json"
OUT = ROOT / "research/certificates/fox8_top_source_donor_sweep_audit.json"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result["scope"] != "fox8_single_top_graph_source_wrong_join_donor_sensitivity":
        raise ValueError("Donor sweep scope differs")
    audits = []
    for run in result["runs"]:
        seed = run["seed"]
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        truth, ambiguous = raw_counts(runtime.source.raw_path, runtime.ids)
        eligible = [uid for uid in runtime.ids if uid not in ambiguous]
        top = sorted(eligible, key=lambda uid: (
            -len(runtime.source.neighbors[runtime.index[uid]]), uid))[0]
        if top != run["top_source_account_id"]:
            raise ValueError("Source ranking differs")
        candidates = [uid for uid in eligible if uid != top and
                      truth[runtime.index[uid]] != truth[runtime.index[top]]]
        expected_donors = sorted(candidates, key=lambda uid: hashlib.sha256(
            f"Fox8-high-fanout-donor-sweep-v1:{uid}".encode()).digest())[:32]
        if expected_donors != [d["donor_account_id"] for d in run["donors"]]:
            raise ValueError("Label-blind donor ordering differs")
        saved_path = runtime.run_dir / "test_predictions.json"
        if sha256_file(saved_path) != run["test_predictions_sha256"]:
            raise ValueError("Frozen test predictions differ")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        rows = np.asarray([runtime.index[d["account_id"]] for d in saved],
                          dtype=int)
        labels = np.asarray([d["label"] for d in saved], dtype=int)
        clean = independent_score(runtime, truth, rows)
        clean_errors = int(np.count_nonzero(
            (clean >= runtime.threshold) != labels))
        if clean_errors != run["clean_errors"] or run["donor_count"] != 32:
            raise ValueError("Clean source or donor count differs")
        if run["donors_with_multi_error_repair_advantage"] != sum(
                d["one_source_repair_beats_oracle_direct_label"]
                for d in run["donors"]):
            raise ValueError("Donor advantage count differs")
        for donor in run["donors"]:
            expected = clean_errors < max(0, donor["corrupted_errors"] - 1)
            if (donor["one_source_repair_beats_oracle_direct_label"] != expected
                    or donor["one_source_repair_remaining_errors"] !=
                    clean_errors):
                raise ValueError("One-action oracle bookkeeping differs")
        checked = []
        clean_error_mask = (clean >= runtime.threshold) != labels
        for position in range(len(run["donors"])):
            donor = run["donors"][position]
            altered = truth.copy()
            altered[runtime.index[top]] = truth[
                runtime.index[donor["donor_account_id"]]]
            probability = independent_score(runtime, altered, rows)
            error_mask = (probability >= runtime.threshold) != labels
            errors = int(np.count_nonzero(error_mask))
            new_errors = int(np.count_nonzero(error_mask & ~clean_error_mask))
            removed_errors = int(np.count_nonzero(clean_error_mask & ~error_mask))
            changed = int(np.count_nonzero(
                (probability >= runtime.threshold) !=
                (clean >= runtime.threshold)))
            if (errors != donor["corrupted_errors"]
                    or new_errors != donor["new_errors_from_corruption"]
                    or removed_errors != donor["removed_errors_from_corruption"]
                    or changed != donor["changed_class_predictions"]
                    or donor["oracle_one_direct_label_remaining_errors"] !=
                    max(0, errors - 1)
                    or donor["donor_raw_statuses_count"] !=
                    truth[runtime.index[donor["donor_account_id"]]]
                    or donor["source_raw_statuses_count"] !=
                    truth[runtime.index[top]]):
                raise ValueError("Independent donor response differs")
            checked.append({"donor_rank": position,
                            "corrupted_errors": errors,
                            "new_errors": new_errors,
                            "removed_errors": removed_errors,
                            "changed_classes": changed,
                            "repair_beats_oracle": donor[
                                "one_source_repair_beats_oracle_direct_label"]})
        audits.append({"seed": seed, "clean_errors": clean_errors,
                       "donors_with_advantage": run[
                           "donors_with_multi_error_repair_advantage"],
                       "independently_replayed_donors": checked})
        print({"seed": seed, "all_donors_independently_replayed":
               len(checked), "donors_with_advantage":
               audits[-1]["donors_with_advantage"]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "all_label_blind_donors_independently_replayed_from_raw",
        "result_sha256": sha256_file(RESULT),
        "all_32_donors_independently_rescored": True,
        "runs": audits,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
