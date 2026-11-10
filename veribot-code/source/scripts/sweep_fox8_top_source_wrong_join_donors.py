"""Label-blind donor sweep for one graph-central Fox8 source-record error."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from stress_fox8_shared_native_source_repair import (
    ambiguous_accounts, raw_statuses, score)
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_fox8_targeted_twenty_ab_group/top_source_donor_sweep.json"
AUDIT = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"
DONOR_COUNT = 32


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    source_audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if source_audit["status"] != (
            "all_model_used_fox8_twenty_facts_independently_recomputed_from_raw"):
        raise ValueError("Raw source truth audit differs")
    runs = []
    donor_order: list[str] | None = None
    top_source: str | None = None
    for seed in (0, 1, 2):
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        truth = raw_statuses(runtime.source.raw_path, runtime.ids)
        if sha256_file(runtime.source.raw_path) != source_audit["raw_source_sha256"]:
            raise ValueError("Fox8 raw source differs")
        ambiguous = ambiguous_accounts(runtime.source.raw_path, runtime.ids, truth)
        eligible = [uid for uid in runtime.ids if uid not in ambiguous]
        selected = sorted(eligible, key=lambda uid: (
            -len(runtime.source.neighbors[runtime.index[uid]]), uid))[0]
        if top_source is not None and selected != top_source:
            raise ValueError("Top native-graph source differs across seeds")
        top_source = selected
        candidates = [uid for uid in eligible if uid != selected and
                      truth[runtime.index[uid]] != truth[runtime.index[selected]]]
        ranked_donors = sorted(candidates, key=lambda uid: hashlib.sha256(
            f"Fox8-high-fanout-donor-sweep-v1:{uid}".encode()).digest())
        chosen = ranked_donors[:DONOR_COUNT]
        if donor_order is not None and chosen != donor_order:
            raise ValueError("Label-blind donor list differs across seeds")
        donor_order = chosen
        saved_path = runtime.run_dir / "test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        rows = np.asarray([runtime.index[r["account_id"]] for r in saved], dtype=int)
        labels = np.asarray([r["label"] for r in saved], dtype=int)
        clean = score(runtime, truth, rows)
        if np.max(np.abs(clean - np.asarray([r["probability"]
                                            for r in saved]))) > 1e-12:
            raise ValueError("Clean source cannot replay frozen prediction")
        clean_error = (clean >= runtime.threshold) != labels
        donor_results = []
        for donor in chosen:
            bad_source = truth.copy()
            bad_source[runtime.index[selected]] = truth[runtime.index[donor]]
            bad_probability = score(runtime, bad_source, rows)
            bad_error = (bad_probability >= runtime.threshold) != labels
            remaining_after_one_oracle_label = max(0, int(bad_error.sum()) - 1)
            row = {
                "donor_account_id": donor,
                "donor_raw_statuses_count": float(truth[runtime.index[donor]]),
                "source_raw_statuses_count": float(truth[runtime.index[selected]]),
                "absolute_log1p_count_difference": float(abs(
                    np.log1p(truth[runtime.index[donor]]) -
                    np.log1p(truth[runtime.index[selected]]))),
                "corrupted_errors": int(bad_error.sum()),
                "new_errors_from_corruption": int(np.count_nonzero(
                    bad_error & ~clean_error)),
                "removed_errors_from_corruption": int(np.count_nonzero(
                    clean_error & ~bad_error)),
                "changed_class_predictions": int(np.count_nonzero(
                    (bad_probability >= runtime.threshold) !=
                    (clean >= runtime.threshold))),
                "oracle_one_direct_label_remaining_errors":
                    remaining_after_one_oracle_label,
                "one_source_repair_remaining_errors": int(clean_error.sum()),
                "one_source_repair_beats_oracle_direct_label": bool(
                    clean_error.sum() < remaining_after_one_oracle_label),
            }
            donor_results.append(row)
        run = {
            "seed": seed,
            "test_accounts": len(rows),
            "clean_errors": int(clean_error.sum()),
            "top_source_account_id": selected,
            "top_source_native_neighbor_count": len(runtime.source.neighbors[
                runtime.index[selected]]),
            "top_source_test_recipients": sum(
                selected == r["account_id"] or
                selected in runtime.source.neighbors[runtime.index[r["account_id"]]]
                for r in saved),
            "donor_count": len(chosen),
            "donors_with_multi_error_repair_advantage": sum(
                r["one_source_repair_beats_oracle_direct_label"]
                for r in donor_results),
            "donors": donor_results,
            "test_predictions_sha256": sha256_file(saved_path),
        }
        runs.append(run)
        print({"seed": seed, "clean_errors": run["clean_errors"],
               "donors_with_advantage":
               run["donors_with_multi_error_repair_advantage"],
               "corrupted_error_counts": [r["corrupted_errors"]
                                          for r in donor_results]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "fox8_single_top_graph_source_wrong_join_donor_sensitivity",
        "donor_selection_uses_labels": False,
        "source_selection_uses_labels": False,
        "donors_are_distinct_real_raw_source_accounts": True,
        "natural_error_frequency_or_human_cost_measured": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
