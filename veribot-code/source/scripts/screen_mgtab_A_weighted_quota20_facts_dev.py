"""Dev-only twenty-fact selection with fixed profile/relation/content budgets."""

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
from screen_mgtab_A_weighted_top20_facts_dev import (
    A_importance, BASE, CONFIG, FULL, OUT as PREVIOUS, ROOT)
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


OUT = ROOT / "output/v3_mgtab_A_weighted_quota20_facts/dev_only.json"
QUOTAS = ((4, 8, 8), (3, 9, 8), (4, 10, 6))


def select_quota(design: np.ndarray, importance: np.ndarray,
                 groups: list[list[int]], quota: tuple[int, int, int]
                 ) -> tuple[np.ndarray, np.ndarray]:
    if (design.shape[1] != 300 or importance.shape != (150,) or
            len(groups) != 9 or sum(quota) != 20 or quota[1] < 7):
        raise ValueError("source evidence or quota contract differs")
    n = len(design)
    priority = np.maximum(design[:, :150], design[:, 150:]) * importance[None, :]
    keep = np.zeros((n, 150), dtype=bool)
    detailed = [[j for j in members if j < 150] for members in groups]
    for members in detailed:
        columns = np.asarray(members, dtype=int)
        winner = columns[np.argmax(priority[:, columns], axis=1)]
        keep[np.arange(n), winner] = True
    broad = (np.asarray(detailed[0], dtype=int),
             np.asarray([j for members in detailed[1:8] for j in members], dtype=int),
             np.asarray(detailed[8], dtype=int))
    if (len(broad[0]) != 20 or len(broad[1]) != 66 or len(broad[2]) != 64 or
            sorted(j for group in broad for j in group) != list(range(150))):
        raise ValueError("source group fact partition differs")
    for columns, target in zip(broad, quota):
        ordering = columns[np.argsort(-priority[:, columns], axis=1,
                                      kind="stable")]
        for rank in range(len(columns)):
            count = keep[:, columns].sum(axis=1)
            if np.all(count >= target):
                break
            missing = count < target
            keep[np.arange(n)[missing], ordering[missing, rank]] = True
        if np.any(keep[:, columns].sum(axis=1) != target):
            raise ValueError("quota failed to fill all twenty fact slots")
    sparse = design.copy()
    sparse[:, :150] *= keep
    sparse[:, 150:] *= keep
    return sparse, keep


def main() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    base = json.loads((BASE / "dev_only.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "data/raw/mgtab_source.json").read_text(
        encoding="utf-8"))
    data_dir = Path(source["path"])
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=FULL)
    if (len(previous["runs"]) != 3 or
            previous["test_predictions_or_metrics_accessed"] is not False):
        raise ValueError("A-weighted development screen differs")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((FULL / f"seed{seed}/report.json").read_text(
            encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if (reference.version != previous["runs"][seed]["reference_version"] or
                old["manifest"]["split_sha256"] !=
                base["runs"][seed]["split_sha256"]):
            raise ValueError("source, split or reference differs")
        design = evidence_matrix(transform_with_support(
            reference, x, split["supports"]))
        groups = source_groups(names, old["reasoner"]["groups"])
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=seed)
        importance = A_importance(design[a], y[a])
        expected = [item["abs_univariate_AUC_minus_half"] for item in
                    previous["runs"][seed]["A_only_fact_importance"]]
        if not np.allclose(importance, expected, atol=1e-14, rtol=0):
            raise ValueError("A-only source relevance differs")
        candidates = []
        for quota in QUOTAS:
            sparse, keep = select_quota(design, importance, groups, quota)
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
                raise ValueError("fitted tree crossed source group")
            probability = model.predict_proba(sparse[dev])[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            candidates.append({
                "quota_profile_relation_content": list(quota),
                "validation": evaluate(y[dev], probability, threshold),
                "selected_threshold": float(threshold),
                "distinct_source_facts_used_on_dev": int(np.any(keep[dev], axis=0).sum()),
                "tree_count_by_group": tree_counts,
            })
        baseline = base["runs"][seed]["candidates"][1]["validation_selected"]
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "baseline_150_fact_validation": baseline,
            "previous_top20_validation": previous["runs"][seed]["candidates"][1][
                "validation"],
            "candidates": candidates,
        })
        print({"seed": seed, "baseline_f1": round(baseline["macro_f1"], 4),
               "quota_f1": [round(row["validation"]["macro_f1"], 4)
                            for row in candidates]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_official_train_dev_A_weighted_twenty_source_facts_with_group_quotas",
        "test_predictions_or_metrics_accessed": False,
        "quotas_fixed_before_this_dev_screen": [list(row) for row in QUOTAS],
        "previous_A_weighted_screen_sha256": sha256_file(PREVIOUS),
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "label_use": "A only for fact relevance and directions; B only for group-additive tree fitting; no account label at inference",
        "lightgbm_version": lightgbm.__version__,
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
