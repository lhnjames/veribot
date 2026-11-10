"""Evaluate locked MGTAB twenty-fact source-quota models on fixed tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_A_weighted_top20_facts_dev import (
    CONFIG, FULL, ROOT)
from screen_mgtab_A_weighted_quota20_facts_dev import (
    OUT, QUOTAS, select_quota)
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_monotone_evidence_tree_dev import evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


LOCK = OUT.parent / "selection_lock.json"
REPORT = OUT.parent / "fixed_test_report.json"


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    screen = json.loads(OUT.read_text(encoding="utf-8"))
    if (lock["screen_sha256"] != sha256_file(OUT) or
            lock["test_metric_or_prediction_used_to_create_lock"] is not False or
            lock["chosen_quota_index"] != 2 or
            lock["chosen_quota_profile_relation_content"] != list(QUOTAS[2]) or
            lock["model_capacity"] != list(CONFIG)):
        raise ValueError("twenty-fact dev selection is not frozen")
    source = json.loads((ROOT / "data/raw/mgtab_source.json").read_text(
        encoding="utf-8"))
    data_dir = Path(source["path"])
    if sha256_file(data_dir / "features.pt") != lock["source_features_sha256"]:
        raise ValueError("MGTAB published source changed")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=FULL)
    runs = []
    for seed in (0, 1, 2):
        artifact = lock["runs"][seed]
        if artifact["seed"] != seed or any(sha256_file(Path(artifact[path_key])) !=
                artifact[hash_key] for path_key, hash_key in (
                    ("model_path", "model_sha256"),
                    ("reference_path", "reference_sha256"),
                    ("selector_path", "selector_sha256"))):
            raise ValueError("frozen twenty-fact model artifact changed")
        old = json.loads((FULL / f"seed{seed}/report.json").read_text(
            encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if (x.shape != (10199, 150) or
                old["manifest"]["split_sha256"] != artifact["split_sha256"]):
            raise ValueError("MGTAB source population or split differs")
        reference = LabelBlindReference.load(Path(artifact["reference_path"]))
        if reference.version != artifact["reference_version"]:
            raise ValueError("train-only reference version differs")
        with np.load(artifact["selector_path"], allow_pickle=False) as selector:
            importance = selector["importance"].astype(float)
            selected_names = selector["names"].astype(str).tolist()
            a = selector["a_indices"].astype(int)
            b = selector["b_indices"].astype(int)
            monotone = selector["monotone_constraints"].astype(int)
            group_names = selector["group_names"].astype(str).tolist()
            quota = tuple(selector["quota_profile_relation_content"].astype(int))
        if (selected_names != names or
                group_names != list(old["reasoner"]["groups"]) or
                quota != QUOTAS[2] or
                len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(split["train"]))):
            raise ValueError("locked selector or A/B split differs")
        groups = source_groups(names, old["reasoner"]["groups"])
        design = evidence_matrix(transform_with_support(
            reference, x, split["supports"]))
        sparse, keep = select_quota(design, importance, groups, quota)
        model = joblib.load(artifact["model_path"])
        if (model.monotone_constraints != monotone.tolist() or
                not all_trees_one_group(model, groups)[0] or
                np.any(keep.sum(axis=1) != 20)):
            raise ValueError("model constraints or selection budget differ")
        dev, test = split["val"], split["test"]
        threshold = float(artifact["threshold"])
        dev_metrics = evaluate(y[dev], model.predict_proba(sparse[dev])[:, 1],
                               threshold)
        if any(abs(dev_metrics[key] - artifact["dev_validation"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("locked model fails official dev replay")
        probability = model.predict_proba(sparse[test])[:, 1]
        metrics = evaluate(y[test], probability, threshold)
        baseline_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_rows = baseline["predictions"]
        if (baseline["manifest"]["split_sha256"] != artifact["split_sha256"]
                or len(baseline_rows) != len(test) or any(
                    row["account_id"] != str(int(i)) or row["label"] != int(y[i])
                    for row, i in zip(baseline_rows, test))):
            raise ValueError("paired HypBot population or labels differ")
        baseline_correct = sum(int((row["probability"] >=
            baseline["manifest"]["threshold"]) == row["label"])
            for row in baseline_rows)
        rows = [{"account_id": str(int(i)), "label": int(y[i]),
                 "probability": float(probability[j]),
                 "prediction": int(probability[j] >= threshold),
                 "selected_fact_names": [names[k] for k in np.flatnonzero(keep[i])]}
                for j, i in enumerate(test)]
        path = OUT.parent / f"seed{seed}/test_predictions.json"
        path.write_text(json.dumps({
            "seed": seed, "threshold": threshold,
            "model_sha256": artifact["model_sha256"],
            "selector_sha256": artifact["selector_sha256"],
            "predictions": rows,
        }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        runs.append({
            "seed": seed, "test_accounts": len(test),
            "test_metrics": metrics,
            "test_correct": int(round(metrics["acc"] * len(test))),
            "hypbot_correct_same_accounts": baseline_correct,
            "strict_numeric_goal_passes": bool(metrics["acc"] > .912 and
                metrics["acc"] > baseline_correct / len(test)),
            "selected_fact_budget_per_account": 20,
            "distinct_selected_fact_names_on_test": int(np.any(keep[test], axis=0).sum()),
            "test_predictions_path": str(path),
            "test_predictions_sha256": sha256_file(path),
            "hypbot_predictions_sha256": sha256_file(baseline_path),
        })
        print({"seed": seed, "acc": metrics["acc"],
               "correct": runs[-1]["test_correct"],
               "hypbot_correct": baseline_correct,
               "strict_goal": runs[-1]["strict_numeric_goal_passes"]}, flush=True)
    REPORT.write_text(json.dumps({
        "scope": "locked_MGTAB_A_weighted_group_quota_top20_fixed_test",
        "selection_lock_sha256": sha256_file(LOCK),
        "prior_candidate_exploration_on_same_test_population": True,
        "all_three_strict_numeric_goal_pass": all(
            row["strict_numeric_goal_passes"] for row in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
