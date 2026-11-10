"""Train-selected top-20 source evidence with cross-type reciprocity, dev only."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_cross_reciprocal_evidence_dev import evidence_matrix
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def fit_model(x: np.ndarray, y: np.ndarray, leaves: int) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=leaves,
        min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1)
    model.fit(x, y)
    return model


def score(model: LGBMClassifier, x: np.ndarray,
          y: np.ndarray) -> dict:
    probability = model.predict_proba(x)[:, 1]
    threshold = best_validation_threshold(probability, y)
    return {"threshold": threshold,
            "validation": evaluate(y, probability, threshold)}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    prior_path = root / "output/v3_mgtab_named_top20_lightgbm/dev_only.json"
    prior = json.loads(prior_path.read_text())
    lock = json.loads((root / "output/v3_mgtab_named_top20_lightgbm/selection_lock.json")
                      .read_text())
    cross_path = root / "data/processed/mgtab_v3/cross_reciprocal_facts.npz"
    cross_audit = root / "research/certificates/mgtab_cross_reciprocal_sample_source_audit.json"
    if json.loads(cross_audit.read_text())["fact_artifact_sha256"] != sha256_file(cross_path):
        raise ValueError("Cross-reciprocal source audit differs")
    with np.load(cross_path, allow_pickle=False) as cache:
        cross_names = cache["names"].astype(str).tolist()
        cross_values = cache["values"].astype(float)
        cross_support = cache["supports"].astype(float)
    full = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((full / f"seed{seed}/report.json").read_text())
        x, y, names, split = selected_facts(args, seed, old)
        if len(names) != 150 or len(cross_names) != 84:
            raise ValueError("Named source population differs")
        values = np.column_stack((x[:, :86], cross_values))
        reference = LabelBlindReference(min_context_n=30).fit(
            values[split["train"]], names[:86] + cross_names)
        train, dev = split["train"], split["val"]
        fit_x = evidence_matrix(reference, values[train],
                                split["supports"][train], cross_support[train])
        dev_x = evidence_matrix(reference, values[dev],
                                split["supports"][dev], cross_support[dev])
        old_selected = np.asarray(lock["selected_fact_indices_by_seed"][str(seed)],
                                  dtype=int)
        old_cols = np.r_[old_selected, old_selected + 170]
        old_result = score(fit_model(fit_x[:, old_cols], y[train], 31),
                           dev_x[:, old_cols], y[dev])
        old_expected = prior["runs"][seed]["alternatives"][3]
        if (abs(old_result["threshold"] - old_expected["threshold"]) > 1e-12
                or any(abs(old_result["validation"][k] -
                           old_expected["validation"][k]) > 1e-12
                       for k in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Original 20-fact validation failed exact replay")
        alternatives = []
        for selector_leaves in (15, 31):
            selector = fit_model(fit_x, y[train], selector_leaves)
            importance = selector.booster_.feature_importance(importance_type="gain")
            selected = np.sort(np.argsort(
                -(importance[:170] + importance[170:]), kind="stable")[:20])
            if len(set(map(int, selected))) != 20:
                raise ValueError("Top-20 selector did not return 20 unique facts")
            cols = np.r_[selected, selected + 170]
            candidate = score(fit_model(fit_x[:, cols], y[train], 31),
                              dev_x[:, cols], y[dev])
            alternatives.append({
                "selector_leaves": selector_leaves,
                "fitter_leaves": 31,
                "selected_fact_indices": selected.tolist(),
                "selected_fact_names": [(names[:86] + cross_names)[int(j)]
                                        for j in selected],
                "cross_reciprocal_count": int(np.count_nonzero(selected >= 86)),
                **candidate,
            })
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "original_20_replayed": old_result,
            "alternatives": alternatives,
        })
        print(json.dumps({"seed": seed,
                          "old_macro_f1": old_result["validation"]["macro_f1"],
                          "new": [(a["cross_reciprocal_count"],
                                   a["validation"]["macro_f1"])
                                  for a in alternatives]}), flush=True)
    result = {
        "scope": "MGTAB_train_selected_20_named_plus_cross_reciprocal_evidence_official_dev_only",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "prior_report_with_test_metrics_loaded_for_manifest": True,
        "prior_test_metrics_used_for_fit_or_selection": False,
        "new_test_predictions_or_metrics_computed": False,
        "content_projection_used": False,
        "cross_reciprocal_fact_sha256": sha256_file(cross_path),
        "cross_reciprocal_source_audit_sha256": sha256_file(cross_audit),
        "original_20_dev_sha256": sha256_file(prior_path),
        "runs": runs,
    }
    path = root / "output/v3_mgtab_cross_reciprocal_top20/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
