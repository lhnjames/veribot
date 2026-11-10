"""Dev-only A-trained top-20 source-fact selector with B-only classifier fitting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_label_blind_top20_facts_dev import select_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "output/v3_mgtab_ab_spline_content/selected"
BASE = ROOT / "output/v3_mgtab_ab_group_additive_tree"
OUT = ROOT / "output/v3_mgtab_A_weighted_top20_facts/dev_only.json"
CONFIG = (31, 400, 50, 10.0)


def A_importance(design: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if design.shape[1] != 300 or len(design) != len(labels):
        raise ValueError("A-only source evidence matrix differs")
    signal = np.zeros(300, dtype=float)
    for j in range(300):
        column = design[:, j]
        if np.min(column) < np.max(column):
            signal[j] = abs(roc_auc_score(labels, column) - .5)
    return np.maximum(signal[:150], signal[150:])


def main() -> None:
    source = json.loads((ROOT / "data/raw/mgtab_source.json").read_text(
        encoding="utf-8"))
    data_dir = Path(source["path"])
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=FULL)
    base = json.loads((BASE / "dev_only.json").read_text(encoding="utf-8"))
    label_blind = json.loads((ROOT /
        "output/v3_mgtab_label_blind_top20_facts/dev_only.json").read_text(
            encoding="utf-8"))
    if base["configs"][1] != list(CONFIG) or len(base["runs"]) != 3 or any(
            row["candidates"][0]["policy"] != "global_top20"
            for row in label_blind["runs"]):
        raise ValueError("baseline or label-blind selector screen differs")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((FULL / f"seed{seed}/report.json").read_text(
            encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if (reference.version != old["manifest"]["reference_version"] or
                old["manifest"]["split_sha256"] !=
                base["runs"][seed]["split_sha256"]):
            raise ValueError("MGTAB source, reference or split differs")
        evidence = transform_with_support(reference, x, split["supports"])
        design = evidence_matrix(evidence)
        groups = source_groups(names, old["reasoner"]["groups"])
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=seed)
        importance = A_importance(design[a], y[a])
        if not np.any(importance):
            raise ValueError("A-only source-fact weights all vanish")
        candidates = []
        for policy in ("A_weighted_global_top20", "A_weighted_group_anchored_top20"):
            anchored = "anchored" in policy
            sparse, keep = select_facts(design, groups, anchored=anchored,
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
            one_group, tree_counts = all_trees_one_group(model, groups)
            if not one_group:
                raise ValueError("tree crossed source groups")
            probability = model.predict_proba(sparse[dev])[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            metric = evaluate(y[dev], probability, threshold)
            candidates.append({
                "policy": policy, "fact_budget_per_account": 20,
                "distinct_selected_fact_names_on_dev": int(np.any(keep[dev], axis=0).sum()),
                "source_groups_present_per_account_min": int(min(sum(
                    np.any(keep[i, [j for j in members if j < len(names)]])
                    for members in groups) for i in dev)),
                "source_groups_present_per_account_max": int(max(sum(
                    np.any(keep[i, [j for j in members if j < len(names)]])
                    for members in groups) for i in dev)),
                "A_monotone_directions": {
                    "increasing": int(np.sum(monotone == 1)),
                    "decreasing": int(np.sum(monotone == -1)),
                    "zero": int(np.sum(monotone == 0)),
                },
                "B_fitted_tree_count_by_source_group": tree_counts,
                "threshold": float(threshold), "validation": metric,
            })
        baseline = base["runs"][seed]["candidates"][1]["validation_selected"]
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "train_A": len(a), "train_B": len(b), "dev": len(dev),
            "baseline_150_fact_validation": baseline,
            "A_only_fact_importance": [
                {"fact": name, "abs_univariate_AUC_minus_half": float(importance[j])}
                for j, name in enumerate(names)],
            "candidates": candidates,
        })
        print({"seed": seed, "baseline_f1": round(baseline["macro_f1"], 4),
               "weighted_top20_f1": [round(row["validation"]["macro_f1"], 4)
                                     for row in candidates]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_official_train_dev_A_only_univariate_fact_weighted_per_account_top20",
        "test_predictions_or_metrics_accessed": False,
        "selector_label_use": "only disjoint training half A for fact-wise absolute univariate ROC-AUC; individual account labels absent at inference",
        "tree_fit_labels": "training half B only",
        "model_capacity": list(CONFIG),
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "lightgbm_version": lightgbm.__version__,
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
