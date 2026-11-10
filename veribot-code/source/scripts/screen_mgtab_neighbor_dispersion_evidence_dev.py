"""Official train/dev capacity check for 12 audited neighbor dispersion facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "output/v3_mgtab_named_evidence_lightgbm/dev_only.json"
AUDIT = ROOT / "research/certificates/mgtab_neighbor_dispersion_full_source_audit.json"
DISPERSION = ROOT / "data/processed/mgtab_v3/neighbor_dispersion_facts.npz"
OUTPUT = ROOT / "output/v3_mgtab_neighbor_dispersion_evidence/dev_only.json"
CONFIG = (31, 400, 50, 10.0)


def design(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def fit_predict(x_train: np.ndarray, labels: np.ndarray,
                x_dev: np.ndarray) -> tuple[LGBMClassifier, np.ndarray]:
    leaves, rounds, min_leaf, l2 = CONFIG
    model = LGBMClassifier(
        objective="binary", n_estimators=rounds, num_leaves=leaves,
        min_child_samples=min_leaf, reg_lambda=l2,
        learning_rate=0.05, n_jobs=4, random_state=0,
        deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(x_train, labels)
    return model, model.predict_proba(x_dev)[:, 1]


def main() -> None:
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    data = resolve_mgtab_source_dir(ROOT)
    structural = ROOT / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = ROOT / "data/processed/mgtab_v3/neighbor_facts.npz"
    if (prior["source_features_sha256"] != sha256_file(data / "features.pt") or
            prior["structural_facts_sha256"] != sha256_file(structural) or
            prior["neighbor_facts_sha256"] != sha256_file(neighbor) or
            prior["configs"][3] != list(CONFIG) or
            audit["fact_cells_checked"] != 10199*12 or
            audit["facts_sha256"] != sha256_file(DISPERSION)):
        raise ValueError("source, dispersion audit, or prior control changed")
    with np.load(DISPERSION, allow_pickle=False) as cache:
        new_values = cache["values"].astype(float)
        new_supports = cache["supports"].astype(float)
        new_names = cache["names"].astype(str).tolist()
        new_ids = cache["ids"].astype(int)
    if (new_values.shape != (10199, 12) or
            not np.array_equal(new_ids, np.arange(10199))):
        raise ValueError("dispersion population differs")
    runs = []
    for seed in (0, 1, 2):
        facts, labels, ids, names, split, source = load(
            "mgtab", data, None, seed)
        facts, names, source = attach_mgtab_structural(
            facts, names, ids, source, data, structural)
        facts, names, supports, source = attach_mgtab_neighbor(
            facts, names, ids, source, data, neighbor)
        if (source["split_sha256"] != prior["runs"][seed]["split_sha256"] or
                len(names) != 86 or len(ids) != len(new_ids)):
            raise ValueError("official split or account population differs: "
                             f"hash={source['split_sha256']}, "
                             f"names={len(names)}, ids={len(ids)}, "
                             f"new_ids={len(new_ids)}")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train, :86], names[:86])
        if reference.version != prior["runs"][seed]["reference_version"]:
            raise ValueError("original named reference differs")
        old_train = design(transform_with_support(
            reference, facts[train, :86], supports[train]))
        old_dev = design(transform_with_support(
            reference, facts[dev, :86], supports[dev]))
        new_reference = LabelBlindReference(min_context_n=30).fit(
            new_values[train], new_names)
        new_train_batch = new_reference.transform(new_values[train])
        new_dev_batch = new_reference.transform(new_values[dev])
        new_train = design(new_train_batch)
        new_dev = design(new_dev_batch)
        train_reliability = new_supports[train] / (new_supports[train] + 5.0)
        dev_reliability = new_supports[dev] / (new_supports[dev] + 5.0)
        new_train *= np.column_stack((train_reliability, train_reliability))
        new_dev *= np.column_stack((dev_reliability, dev_reliability))
        full_train = np.column_stack((old_train, new_train))
        full_dev = np.column_stack((old_dev, new_dev))
        if (full_train.shape[1] != 196 or not np.isfinite(full_train).all() or
                not np.isfinite(full_dev).all()):
            raise ValueError("named dispersion evidence matrix differs")
        _, old_p = fit_predict(old_train, labels[train], old_dev)
        old_threshold = best_validation_threshold(old_p, labels[dev])
        old_eval = evaluate(labels[dev], old_p, old_threshold)
        previous_row = prior["runs"][seed]["alternatives"][3]
        if (abs(old_threshold-previous_row["threshold"]) > 1e-9 or
                any(abs(old_eval[name]-previous_row["validation"][name]) > 1e-9
                    for name in ("acc", "macro_f1", "auroc"))):
            raise ValueError("86-fact capacity control did not replay")
        model, new_p = fit_predict(full_train, labels[train], full_dev)
        threshold = best_validation_threshold(new_p, labels[dev])
        new_eval = evaluate(labels[dev], new_p, threshold)
        gains = model.booster_.feature_importance(importance_type="gain")
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "named_reference_version": reference.version,
            "dispersion_reference_version": new_reference.version,
            "baseline": {"threshold": old_threshold,
                         "validation": old_eval},
            "with_dispersion": {
                "threshold": threshold, "validation": new_eval,
                "new_tail_columns_with_positive_gain": int(np.count_nonzero(
                    gains[172:] > 0)),
                "new_group_gain_fraction": float(
                    np.sum(gains[172:]) / np.sum(gains)),
            },
        })
        print({"seed": seed,
               "baseline_macro_f1": old_eval["macro_f1"],
               "new_macro_f1": new_eval["macro_f1"],
               "new_gain_fraction": runs[-1]["with_dispersion"][
                   "new_group_gain_fraction"]}, flush=True)
    old_mean = float(np.mean([r["baseline"]["validation"]["macro_f1"]
                              for r in runs]))
    new_mean = float(np.mean([
        r["with_dispersion"]["validation"]["macro_f1"] for r in runs]))
    gate = (new_mean >= old_mean + 0.005 and all(
        r["with_dispersion"]["validation"]["macro_f1"] >=
        r["baseline"]["validation"]["macro_f1"] - 0.002
        for r in runs))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "scope": "mgtab_train_dev_only_neighbor_dispersion_named_evidence_capacity",
        "test_predictions_or_metrics_accessed": False,
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "source_features_sha256": sha256_file(data / "features.pt"),
        "dispersion_source_audit_sha256": sha256_file(AUDIT),
        "dispersion_facts_sha256": sha256_file(DISPERSION),
        "config": CONFIG,
        "selection_gate": "mean dev Macro-F1 at least 0.5 points above same-capacity baseline with no seed worse by over 0.2 points",
        "baseline_mean_dev_macro_f1": old_mean,
        "with_dispersion_mean_dev_macro_f1": new_mean,
        "gate_passed": gate,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
