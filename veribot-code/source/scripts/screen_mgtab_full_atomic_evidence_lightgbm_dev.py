"""Dev-only MGTAB capacity screen over all source-bound atomic evidence tails."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIGS = ((15, 200, 30, 10.0), (15, 400, 30, 10.0),
           (31, 200, 50, 10.0), (31, 400, 50, 10.0))


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    source_audit = root / "research/certificates/mgtab_full_source_audit.json"
    source_status = json.loads(source_audit.read_text(encoding="utf-8"))
    if source_status["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("MGTAB source facts are not independently audited")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((full_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if x.shape != (10199, 150) or len(names) != 150:
            raise ValueError("MGTAB source evidence vocabulary differs")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            x[train], names)
        if reference.version != old["manifest"]["reference_version"]:
            raise ValueError("Label-blind training reference differs from frozen model")
        x_train = matrix(transform_with_support(reference, x[train],
                                                split["supports"][train]))
        x_dev = matrix(transform_with_support(reference, x[dev],
                                              split["supports"][dev]))
        if (x_train.shape[1] != 300 or not np.isfinite(x_train).all()
                or not np.isfinite(x_dev).all()):
            raise ValueError("Encoded source evidence matrix differs")
        alternatives = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations,
                num_leaves=leaves, min_child_samples=min_leaf,
                reg_lambda=l2, learning_rate=0.05, n_jobs=4,
                random_state=0, deterministic=True,
                force_col_wise=True, verbosity=-1)
            model.fit(x_train, y[train])
            probability = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            probe = x_dev[:32]
            raw = np.asarray(model.booster_.predict(probe, raw_score=True))
            shap = np.asarray(model.booster_.predict(probe, pred_contrib=True))
            additivity = float(np.max(np.abs(shap.sum(axis=1) - raw)))
            if shap.shape != (len(probe), 301) or additivity > 1e-9:
                raise ValueError("Evidence contribution does not reconstruct score")
            importances = model.booster_.feature_importance(importance_type="gain")
            fact_gain = importances[:150] + importances[150:]
            group_gain = {
                "published_profile": float(fact_gain[:20].sum()),
                "published_native_relation": float(fact_gain[20:86].sum()),
                "published_content_projection": float(fact_gain[86:].sum()),
            }
            total_gain = max(sum(group_gain.values()), 1e-12)
            group_gain = {key: value / total_gain for key, value in
                          group_gain.items()}
            group_masks = {}
            for group, indices in {
                    "published_profile": list(range(0, 20)),
                    "published_native_relation": list(range(20, 86)),
                    "published_content_projection": list(range(86, 150)),
            }.items():
                masked = x_dev.copy()
                masked[:, indices] = 0.0
                masked[:, np.asarray(indices) + 150] = 0.0
                masked_probability = model.predict_proba(masked)[:, 1]
                group_masks[group] = {
                    "changed_classes_at_fixed_half": int(np.count_nonzero(
                        (masked_probability >= 0.5) != (probability >= 0.5))),
                    "changed_classes_at_dev_threshold": int(np.count_nonzero(
                        (masked_probability >= threshold) !=
                        (probability >= threshold))),
                }
            alternatives.append({
                "config": [leaves, iterations, min_leaf, l2],
                "validation_at_fixed_half": evaluate(y[dev], probability, 0.5),
                "validation_selected_threshold": threshold,
                "validation_at_selected_threshold": evaluate(
                    y[dev], probability, threshold),
                "group_gain_share": group_gain,
                "group_masks": group_masks,
                "max_probe_logit_additivity_error": additivity,
            })
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "projection_fingerprint": old["manifest"]["projection_fingerprint"],
            "prior_ab_spline_validation": old["validation"],
            "alternatives": alternatives,
        })
        print(json.dumps({"seed": seed,
                          "acc_half": [round(item[
                              "validation_at_fixed_half"]["acc"], 5)
                                       for item in alternatives],
                          "macro_f1_half": [round(item[
                              "validation_at_fixed_half"]["macro_f1"], 5)
                                            for item in alternatives]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_all_150_source_evidence_lightgbm",
        "test_predictions_or_metrics_computed": False,
        "source_level": "published_profile_tensor_typed_edges_and_averaged_tweet_tensor",
        "projection_semantics": "numeric coordinates only; no original tweet text or externally verified behavior claim",
        "model_input": "300 upper/lower tail evidence values only; no raw feature bypass",
        "lightgbm_version": lightgbm.__version__,
        "source_audit_sha256": sha256_file(source_audit),
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "structural_facts_sha256": sha256_file(args.structural_facts),
        "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
        "configs": CONFIGS,
        "runs": runs,
    }
    path = root / "output/v3_mgtab_full_atomic_evidence_lightgbm/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
