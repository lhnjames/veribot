"""Dev-only incremental test of source-audited reciprocal MGTAB evidence."""

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
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIGS = ((31, 400, 50, 10.0), (15, 400, 50, 10.0))


def evidence_matrix(reference: LabelBlindReference, values: np.ndarray,
                    original_support: np.ndarray, reciprocal_support: np.ndarray
                    ) -> np.ndarray:
    evidence = reference.transform(values)
    reliability = evidence.reliability.copy()
    supports = np.column_stack((original_support, reciprocal_support))
    reliability[:, 62:170] *= supports / (supports + 5.0)
    return np.column_stack((np.log1p(evidence.upper) * reliability,
                            np.log1p(evidence.lower) * reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original = root / "output/v3_mgtab_ab_spline_content/selected"
    prior_path = root / "output/v3_mgtab_named_evidence_lightgbm/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    source = root / "data/processed/mgtab_v3/cross_reciprocal_facts.npz"
    manifest = json.loads(source.with_suffix(".json").read_text())
    audit_path = root / "research/certificates/mgtab_cross_reciprocal_sample_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (manifest["facts_sha256"] != sha256_file(source)
            or audit["fact_artifact_sha256"] != sha256_file(source)
            or audit["status"] != "sampled_source_facts_verified"):
        raise ValueError("Cross-type reciprocal source artifact is not audited")
    with np.load(source, allow_pickle=False) as cache:
        ids = cache["ids"]
        new_names = cache["names"].astype(str).tolist()
        new_values = cache["values"].astype(float)
        new_support = cache["supports"].astype(float)
    if (not np.array_equal(ids, np.arange(10199))
            or len(new_names) != 84 or new_values.shape != (10199, 84)):
        raise ValueError("Cross-type reciprocal account order differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=original)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((original / f"seed{seed}/report.json").read_text())
        x, y, names, split = selected_facts(args, seed, old)
        if len(names) != 150 or len(set(names[:86] + new_names)) != 170:
            raise ValueError("Named fact vocabulary differs")
        train, dev = split["train"], split["val"]
        full = np.column_stack((x[:, :86], new_values))
        reference = LabelBlindReference(min_context_n=30).fit(
            full[train], names[:86] + new_names)
        fit_x = evidence_matrix(reference, full[train],
                                split["supports"][train], new_support[train])
        dev_x = evidence_matrix(reference, full[dev],
                                split["supports"][dev], new_support[dev])
        if (fit_x.shape[1] != 340 or dev_x.shape[1] != 340
                or not np.isfinite(fit_x).all() or not np.isfinite(dev_x).all()):
            raise ValueError("Evidence-only matrix differs")
        all_columns = np.arange(340)
        base_columns = np.r_[np.arange(86), np.arange(170, 256)]
        outcomes = {}
        for config in CONFIGS:
            leaves, rounds, child, l2 = config
            for branch, columns in (("named_86_control", base_columns),
                                    ("named_86_plus_reciprocal_84", all_columns)):
                model = LGBMClassifier(
                    objective="binary", n_estimators=rounds,
                    num_leaves=leaves, min_child_samples=child,
                    reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                    random_state=0, deterministic=True, force_col_wise=True,
                    verbosity=-1)
                model.fit(fit_x[:, columns], y[train])
                probability = model.predict_proba(dev_x[:, columns])[:, 1]
                threshold = best_validation_threshold(probability, y[dev])
                outcomes[f"{branch}_leaves{leaves}"] = {
                    "config": config,
                    "threshold": threshold,
                    "validation": evaluate(y[dev], probability, threshold),
                    "reciprocal_gain_share": (
                        float(model.booster_.feature_importance(
                            importance_type="gain")[np.r_[86:170, 256:340]].sum() /
                              max(model.booster_.feature_importance(
                                  importance_type="gain").sum(), 1e-12))
                        if branch != "named_86_control" else 0.0),
                }
        expected = prior["runs"][seed]["alternatives"][3]
        control = outcomes["named_86_control_leaves31"]
        if (abs(control["threshold"] - expected["threshold"]) > 1e-12
                or any(abs(control["validation"][key] -
                           expected["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("86-fact validation control failed exact replay")
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "outcomes": outcomes,
        })
        print(json.dumps({"seed": seed,
                          "control": control["validation"]["macro_f1"],
                          "extra": outcomes["named_86_plus_reciprocal_84_leaves31"]["validation"]["macro_f1"]}),
              flush=True)
    result = {
        "scope": "MGTAB_official_train_dev_only_source_audited_cross_type_reciprocal_evidence",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "prior_report_with_test_metrics_loaded_for_manifest": True,
        "prior_test_metrics_used_for_fit_or_selection": False,
        "new_test_predictions_or_metrics_computed": False,
        "published_content_projection_used": False,
        "cross_reciprocal_fact_sha256": sha256_file(source),
        "cross_reciprocal_source_audit_sha256": sha256_file(audit_path),
        "prior_named_evidence_lightgbm_sha256": sha256_file(prior_path),
        "configs": CONFIGS,
        "runs": runs,
    }
    path = root / "output/v3_mgtab_cross_reciprocal_evidence/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
