"""Freeze A-weighted twenty-fact selector and B-fitted models before test access."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_A_weighted_top20_facts_dev import (
    A_importance, BASE, CONFIG, FULL, OUT, ROOT)
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_label_blind_top20_facts_dev import select_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


LOCK = OUT.parent / "selection_lock.json"


def main() -> None:
    screen = json.loads(OUT.read_text(encoding="utf-8"))
    means = [float(np.mean([run["candidates"][j]["validation"]["macro_f1"]
                            for run in screen["runs"]])) for j in (0, 1)]
    chosen = int(np.argmax(means))
    if (len(screen["runs"]) != 3 or
            screen["test_predictions_or_metrics_accessed"] is not False or
            chosen != 1 or CONFIG != (31, 400, 50, 10.0)):
        raise ValueError("A-weighted development selection differs")
    source = json.loads((ROOT / "data/raw/mgtab_source.json").read_text(
        encoding="utf-8"))
    data_dir = Path(source["path"])
    if sha256_file(data_dir / "features.pt") != screen["source_features_sha256"]:
        raise ValueError("published MGTAB source changed")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=FULL)
    artifacts = []
    for seed in (0, 1, 2):
        old = json.loads((FULL / f"seed{seed}/report.json").read_text(
            encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        expected = screen["runs"][seed]
        if (reference.version != expected["reference_version"] or
                old["manifest"]["split_sha256"] != expected["split_sha256"]):
            raise ValueError("reference or official split differs from dev screen")
        design = evidence_matrix(transform_with_support(
            reference, x, split["supports"]))
        groups = source_groups(names, old["reasoner"]["groups"])
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=seed)
        importance = A_importance(design[a], y[a])
        expected_importance = [item["abs_univariate_AUC_minus_half"]
                               for item in expected["A_only_fact_importance"]]
        if (not np.allclose(importance, expected_importance, atol=1e-14, rtol=0)
                or [item["fact"] for item in expected[
                    "A_only_fact_importance"]] != names):
            raise ValueError("A-only selection weights differ")
        sparse, keep = select_facts(design, groups, anchored=True,
                                    importance=importance)
        monotone = directions(sparse[a], y[a])
        model = LGBMClassifier(
            objective="binary", n_estimators=CONFIG[1],
            num_leaves=CONFIG[0], min_child_samples=CONFIG[2],
            reg_lambda=CONFIG[3], learning_rate=.05, n_jobs=4,
            random_state=0, deterministic=True, force_col_wise=True,
            monotone_constraints=monotone.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups, verbosity=-1)
        model.fit(sparse[b], y[b])
        one_group, _ = all_trees_one_group(model, groups)
        if not one_group:
            raise ValueError("fitted tree crosses source groups")
        threshold = expected["candidates"][chosen]["threshold"]
        metrics = evaluate(y[dev], model.predict_proba(sparse[dev])[:, 1],
                           threshold)
        if any(abs(metrics[key] - expected["candidates"][chosen][
                "validation"][key]) > 1e-12 for key in
                ("acc", "macro_f1", "auroc")):
            raise ValueError("selected model does not replay dev screen")
        folder = OUT.parent / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / "model.joblib"
        reference_path = folder / "label_blind_reference.json.gz"
        selector_path = folder / "selector.npz"
        joblib.dump(model, model_path)
        reference.save(reference_path)
        np.savez_compressed(selector_path,
                            importance=importance, names=np.asarray(names, dtype=str),
                            a_indices=a, b_indices=b,
                            monotone_constraints=monotone,
                            group_names=np.asarray(list(old["reasoner"]["groups"]),
                                                   dtype=str))
        artifacts.append({
            "seed": seed, "split_sha256": expected["split_sha256"],
            "reference_version": reference.version,
            "threshold": threshold,
            "dev_validation": metrics,
            "model_path": str(model_path), "model_sha256": sha256_file(model_path),
            "reference_path": str(reference_path),
            "reference_sha256": sha256_file(reference_path),
            "selector_path": str(selector_path),
            "selector_sha256": sha256_file(selector_path),
            "selected_source_facts_per_account": int(keep[dev].sum(axis=1).min()),
        })
        print({"seed": seed, "dev_f1": metrics["macro_f1"],
               "model_sha256": artifacts[-1]["model_sha256"][:12]}, flush=True)
    LOCK.write_text(json.dumps({
        "scope": "locked_mgtab_A_weighted_group_anchored_top20_source_fact_model",
        "selection_rule": "largest mean official-dev Macro-F1 across the two preregistered A-weighted top20 policies",
        "chosen_policy_index": chosen,
        "chosen_policy": "A_weighted_group_anchored_top20",
        "mean_dev_macro_f1_by_policy": means,
        "screen_sha256": sha256_file(OUT),
        "source_features_sha256": screen["source_features_sha256"],
        "model_capacity": list(CONFIG),
        "test_metric_or_prediction_used_to_create_lock": False,
        "prior_candidates_on_same_test_population_already_explored": True,
        "runs": artifacts,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(LOCK)


if __name__ == "__main__":
    main()
