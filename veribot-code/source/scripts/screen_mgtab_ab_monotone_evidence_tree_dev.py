"""Dev-only LightGBM with A-derived evidence directions and B-only fitting."""

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
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIGS = ((15, 200, 30, 10.0), (31, 400, 50, 10.0))


def evidence_matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def directions(x_a: np.ndarray, y_a: np.ndarray) -> np.ndarray:
    bot, human = x_a[y_a == 1], x_a[y_a == 0]
    if len(bot) == 0 or len(human) == 0:
        raise ValueError("direction population needs both classes")
    delta = bot.mean(axis=0) - human.mean(axis=0)
    sign = np.sign(delta).astype(np.int8)
    return sign


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((full_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if reference.version != old["manifest"]["reference_version"]:
            raise ValueError("MGTAB source/reference differs")
        evidence = transform_with_support(reference, x, split["supports"])
        design = evidence_matrix(evidence)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        constraint = directions(design[a], y[a])
        if len(constraint) != 300 or not set(constraint) <= {-1, 0, 1}:
            raise ValueError("A-derived monotone directions differ")
        candidates = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations,
                num_leaves=leaves, min_child_samples=min_leaf,
                reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                random_state=0, deterministic=True, force_col_wise=True,
                monotone_constraints=constraint.tolist(),
                monotone_constraints_method="advanced", verbosity=-1)
            model.fit(design[b], y[b])
            probability = model.predict_proba(design[dev])[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            raw = np.asarray(model.booster_.predict(design[dev][:32], raw_score=True))
            contribution = np.asarray(model.booster_.predict(
                design[dev][:32], pred_contrib=True))
            additivity = float(np.max(np.abs(contribution.sum(axis=1) - raw)))
            if contribution.shape != (32, 301) or additivity > 1e-9:
                raise ValueError("evidence tree does not replay score")
            candidates.append({
                "config": [leaves, iterations, min_leaf, l2],
                "validation_at_half": evaluate(y[dev], probability, 0.5),
                "selected_threshold": threshold,
                "validation_selected": evaluate(y[dev], probability, threshold),
                "max_probe_logit_additivity_error": additivity,
                "tree_count": model.booster_.num_trees(),
            })
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "direction_half_count": len(a), "fit_half_count": len(b),
            "increasing_input_count": int(np.count_nonzero(constraint == 1)),
            "decreasing_input_count": int(np.count_nonzero(constraint == -1)),
            "unconstrained_input_count": int(np.count_nonzero(constraint == 0)),
            "original_ab_spline_validation": old["validation"],
            "candidates": candidates,
        })
        print(json.dumps({
            "seed": seed,
            "old_acc": old["validation"]["acc"],
            "new_acc": [round(item["validation_selected"]["acc"], 5)
                        for item in candidates],
            "new_f1": [round(item["validation_selected"]["macro_f1"], 5)
                       for item in candidates]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_ab_disjoint_monotone_evidence_tree",
        "test_predictions_or_metrics_computed": False,
        "reference_uses_train_facts_without_labels": True,
        "monotone_direction_labels": "A only",
        "tree_fit_labels": "B only",
        "group_additivity_enforced": False,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "lightgbm_version": lightgbm.__version__,
        "configs": CONFIGS,
        "runs": runs,
    }
    path = root / "output/v3_mgtab_ab_monotone_evidence_tree/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
