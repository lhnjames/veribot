"""Independently replay saved test probabilities and paired counts for the locked booster."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_named_evidence_boosting"
    lock_path = folder / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads((folder / "selected/paired_summary.json")
                         .read_text(encoding="utf-8"))
    if summary["selection_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("frozen candidate selection lock differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    verified = []
    for seed in (0, 1, 2):
        run_dir = folder / "selected" / f"seed{seed}"
        path = run_dir / "test_predictions.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        run = summary["runs"][seed]
        model_path = run_dir / "model.joblib"
        reference_path = run_dir / "reference.json.gz"
        if (saved["manifest"]["selection_lock_sha256"] != sha256_file(lock_path)
                or saved["manifest"]["model_sha256"] != sha256_file(model_path)
                or saved["manifest"]["reference_sha256"] != sha256_file(reference_path)
                or run["predictions_sha256"] != sha256_file(path)):
            raise ValueError("selected model or prediction artifact differs")
        old = json.loads((args.selected_model_dir / f"seed{seed}" / "report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        reference = LabelBlindReference.load(reference_path)
        if (list(reference.names) != names[:86]
                or old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("source fact order or official split differs")
        test = split["test"]
        evidence = transform_with_support(
            reference, x[test, :86], split["supports"][test])
        matrix = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                  np.log1p(evidence.lower) * evidence.reliability))
        model = joblib.load(model_path)
        probability = model.predict_proba(matrix)[:, 1]
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        threshold = saved["manifest"]["threshold"]
        h_threshold = baseline["manifest"]["threshold"]
        counts = {key: 0 for key in ("new_correct", "hypbot_correct",
                                      "new_only_correct", "hypbot_only_correct")}
        max_difference = 0.0
        for position, index in enumerate(test):
            row = saved["predictions"][position]
            other = baseline["predictions"][position]
            if (row["account_id"] != str(int(index))
                    or other["account_id"] != row["account_id"]
                    or row["label"] != int(y[index])
                    or other["label"] != row["label"]):
                raise ValueError("paired source population differs")
            max_difference = max(max_difference,
                                 abs(float(probability[position]) - row["probability"]))
            new_ok = int(probability[position] >= threshold) == row["label"]
            h_ok = int(other["probability"] >= h_threshold) == row["label"]
            counts["new_correct"] += new_ok
            counts["hypbot_correct"] += h_ok
            counts["new_only_correct"] += new_ok and not h_ok
            counts["hypbot_only_correct"] += h_ok and not new_ok
        if (max_difference > 1e-12 or len(test) != run["n"]
                or counts["new_correct"] != run["correct"]
                or counts["hypbot_correct"] != run["hypbot_correct"]
                or counts["new_only_correct"] != run["new_correct_baseline_wrong"]
                or counts["hypbot_only_correct"] != run["new_wrong_baseline_correct"]):
            raise ValueError("replayed probabilities or paired counts differ")
        verified.append({"seed": seed, "n": len(test), **counts,
                         "max_saved_probability_difference": max_difference,
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(reference_path)})
        print(seed, counts, flush=True)
    output = {"scope": "independent_replay_of_3060_named_evidence_booster_test_rows",
              "selection_lock_sha256": sha256_file(lock_path),
              "paired_summary_sha256": sha256_file(
                  folder / "selected/paired_summary.json"),
              "runs": verified}
    destination = root / "research/mgtab_locked_named_evidence_booster_replay_2026-11-03.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
