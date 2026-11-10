"""Dev-only screen of twenty source facts selected without account labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "output/v3_mgtab_ab_spline_content/selected"
BASE = ROOT / "output/v3_mgtab_ab_group_additive_tree"
OUT = ROOT / "output/v3_mgtab_label_blind_top20_facts/dev_only.json"
CONFIG = (31, 400, 50, 10.0)
K = 20


def select_facts(design: np.ndarray, groups: list[list[int]],
                 *, anchored: bool,
                 importance: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    n, width = design.shape
    if width != 300:
        raise ValueError("expected 150 facts and their two tails")
    count = width // 2
    priority = np.maximum(design[:, :count], design[:, count:])
    if importance is not None:
        if (importance.shape != (count,) or not np.isfinite(importance).all()
                or np.any(importance < 0)):
            raise ValueError("A-learned fact importance differs")
        priority *= importance[None, :]
    if np.any(priority < 0) or not np.isfinite(priority).all():
        raise ValueError("evidence priority is invalid")
    ordering = np.argsort(-priority, axis=1, kind="stable")
    keep = np.zeros((n, count), dtype=bool)
    if anchored:
        group_facts = [[j for j in members if j < count] for members in groups]
        if len(group_facts) > K or any(not members for members in group_facts):
            raise ValueError("source groups do not fit the fact budget")
        for members in group_facts:
            choices = np.asarray(members, dtype=int)
            winner = choices[np.argmax(priority[:, choices], axis=1)]
            keep[np.arange(n), winner] = True
    for rank in range(count):
        if np.all(keep.sum(axis=1) >= K):
            break
        candidate = ordering[:, rank]
        available = keep.sum(axis=1) < K
        keep[np.arange(n)[available], candidate[available]] = True
    if np.any(keep.sum(axis=1) != K):
        raise ValueError("label-blind selection did not fill the fact budget")
    sparse = design.copy()
    sparse[:, :count] *= keep
    sparse[:, count:] *= keep
    return sparse, keep


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
    if base["configs"][1] != list(CONFIG) or len(base["runs"]) != 3:
        raise ValueError("frozen baseline capacity differs")
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
        candidates = []
        for policy in ("global_top20", "one_per_group_then_top20"):
            sparse, keep = select_facts(
                design, groups, anchored=policy.startswith("one_per_group"))
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
                "policy": policy, "fact_budget_per_account": K,
                "source_groups_present_per_account_min": int(min(sum(
                    np.any(keep[i, [j for j in members if j < len(names)]])
                    for members in groups) for i in dev)),
                "source_groups_present_per_account_max": int(max(sum(
                    np.any(keep[i, [j for j in members if j < len(names)]])
                    for members in groups) for i in dev)),
                "distinct_selected_fact_names_on_dev": int(np.any(keep[dev], axis=0).sum()),
                "A_monotone_directions": {
                    "increasing": int(np.sum(monotone == 1)),
                    "decreasing": int(np.sum(monotone == -1)),
                    "zero": int(np.sum(monotone == 0)),
                },
                "B_fitted_tree_count_by_source_group": tree_counts,
                "threshold": float(threshold), "validation": metric,
                "reference_version": reference.version,
            })
        baseline = base["runs"][seed]["candidates"][1]["validation_selected"]
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "train_A": len(a), "train_B": len(b), "dev": len(dev),
            "baseline_150_fact_validation": baseline,
            "candidates": candidates,
        })
        print({"seed": seed, "baseline_f1": round(baseline["macro_f1"], 4),
               "top20_f1": [round(row["validation"]["macro_f1"], 4)
                            for row in candidates]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_official_train_dev_label_blind_per_account_top20_source_facts",
        "test_predictions_or_metrics_accessed": False,
        "label_blind_selector": "max of reliable upper/lower source-fact tail values; stable fact-index tie break; optionally reserve one fact per predefined source group",
        "model": "same A/B split, monotone direction procedure, single-source-group LightGBM capacity as 150-fact benchmark",
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "lightgbm_version": lightgbm.__version__,
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
