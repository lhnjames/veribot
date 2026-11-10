"""Evaluate the dev-locked 86-fact evidence booster on fixed MGTAB test splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def evidence_matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_named_evidence_boosting"
    lock_path = folder / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev_path = folder / "dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    prior = json.loads((root / "output/v3_mgtab_named_raw_capacity/dev_only.json")
                       .read_text(encoding="utf-8"))
    if (lock["dev_result_sha256"] != sha256_file(dev_path)
            or lock["selected_config"] != [15, 300, 30, 10.0]
            or lock["selected_config_index"] != 1
            or lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("test configuration is not dev locked")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    if (lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(structural)
            or lock["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("source inputs differ from validation lock")
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor, selected_model_dir=full_dir)
    rows = []
    leaves, iterations, min_leaf, l2 = lock["selected_config"]
    for seed in (0, 1, 2):
        old_report = json.loads((full_dir / f"seed{seed}" / "report.json")
                                .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old_report)
        if (names[:86] != prior["fact_names"] or
                lock["split_sha256_by_seed"][str(seed)] !=
                old_report["manifest"]["split_sha256"]):
            raise ValueError("named source facts or official split differ")
        # The saved dev run specifies the fact count; source identity is bound above.
        if len(names) != 150 or dev["runs"][seed]["named_fact_count"] != 86:
            raise ValueError("named evidence vocabulary differs")
        train, val, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(
            x[train, :86], names[:86])
        if reference.version != dev["runs"][seed]["reference_version"]:
            raise ValueError("label-blind evidence reference differs")
        x_train = evidence_matrix(transform_with_support(
            reference, x[train, :86], split["supports"][train]))
        x_val = evidence_matrix(transform_with_support(
            reference, x[val, :86], split["supports"][val]))
        x_test = evidence_matrix(transform_with_support(
            reference, x[test, :86], split["supports"][test]))
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=leaves, max_iter=iterations,
            min_samples_leaf=min_leaf, l2_regularization=l2,
            learning_rate=0.05, random_state=0)
        model.fit(x_train, y[train])
        dev_probability = model.predict_proba(x_val)[:, 1]
        threshold = best_validation_threshold(dev_probability, y[val])
        dev_metrics = evaluate(y[val], dev_probability, threshold)
        expected = dev["runs"][seed]["alternatives"][1]
        if (abs(threshold - lock["threshold_by_seed"][str(seed)]) > 1e-12
                or abs(threshold - expected["threshold"]) > 1e-12
                or any(abs(dev_metrics[key] - expected["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("validation result failed exact replay before test")
        probability = model.predict_proba(x_test)[:, 1]
        metrics = evaluate(y[test], probability, threshold)
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (baseline["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(baseline["predictions"]) != len(test)):
            raise ValueError("paired baseline population differs")
        saved = []
        correct = baseline_correct = only_new = only_baseline = 0
        baseline_threshold = baseline["manifest"]["threshold"]
        for position, index in enumerate(test):
            other = baseline["predictions"][position]
            if other["account_id"] != str(int(index)) or other["label"] != int(y[index]):
                raise ValueError("paired account or label differs")
            prediction = int(probability[position] >= threshold)
            baseline_prediction = int(other["probability"] >= baseline_threshold)
            new_ok = prediction == int(y[index])
            baseline_ok = baseline_prediction == int(y[index])
            correct += new_ok
            baseline_correct += baseline_ok
            only_new += new_ok and not baseline_ok
            only_baseline += baseline_ok and not new_ok
            saved.append({"account_id": other["account_id"],
                          "label": int(y[index]),
                          "probability": float(probability[position]),
                          "prediction": prediction})
        run_dir = folder / "selected" / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        reference_path = run_dir / "reference.json.gz"
        joblib.dump(model, model_path)
        reference.save(reference_path)
        predictions_path = run_dir / "test_predictions.json"
        predictions_path.write_text(json.dumps({
            "manifest": {"split_sha256": lock["split_sha256_by_seed"][str(seed)],
                         "threshold": threshold,
                         "selection_lock_sha256": sha256_file(lock_path),
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(reference_path)},
            "predictions": saved}, indent=2) + "\n", encoding="utf-8")
        row = {"seed": seed, "n": len(test), "validation": dev_metrics,
               "test": metrics, "correct": correct,
               "hypbot_correct": baseline_correct,
               "new_correct_baseline_wrong": only_new,
               "new_wrong_baseline_correct": only_baseline,
               "net_correct": only_new - only_baseline,
               "model_sha256": sha256_file(model_path),
               "reference_sha256": sha256_file(reference_path),
               "predictions_sha256": sha256_file(predictions_path),
               "hypbot_predictions_sha256": sha256_file(baseline_path)}
        rows.append(row)
        print(seed, round(metrics["acc"] * 100, 2),
              correct, baseline_correct, flush=True)
    summary = {"dataset": "mgtab", "scope": "dev_locked_named_source_evidence_booster_fixed_test",
               "selected_config": lock["selected_config"],
               "selection_lock_sha256": sha256_file(lock_path),
               "prior_test_population_previously_explored": True,
               "runs": rows}
    destination = folder / "selected/paired_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
