"""Fixed-test evaluation of dev-locked MGTAB 150-fact evidence-only classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


GROUPS = {
    "published_profile": list(range(20)),
    "published_native_relations": list(range(20, 86)),
    "published_tweet_projection": list(range(86, 150)),
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_full_atomic_evidence_lightgbm"
    lock_path, dev_path = folder / "selection_lock.json", folder / "dev_only.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    if (lock["dev_result_sha256"] != sha256_file(dev_path)
            or lock["selected_config"] != [31, 400, 50, 10.0]
            or lock["selected_config_index"] != 3
            or lock["threshold"] != 0.5
            or lock["test_metric_or_prediction_used_to_create_lock"] is not False
            or lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(args.structural_facts)
            or lock["neighbor_facts_sha256"] != sha256_file(args.neighbor_facts)):
        raise ValueError("MGTAB model, sources or selection not locked")
    summary = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if (x.shape != (10199, 150)
                or old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or old["manifest"]["projection_fingerprint"] !=
                lock["projection_fingerprint_by_seed"][str(seed)]):
            raise ValueError("MGTAB source population or projection differs")
        train, val, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Train-only reference differs from dev lock")
        train_batch = transform_with_support(reference, x[train],
                                             split["supports"][train])
        val_batch = transform_with_support(reference, x[val], split["supports"][val])
        test_batch = transform_with_support(reference, x[test], split["supports"][test])
        train_matrix, val_matrix, test_matrix = (
            matrix(batch) for batch in (train_batch, val_batch, test_batch))
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1)
        model.fit(train_matrix, y[train])
        val_probability = model.predict_proba(val_matrix)[:, 1]
        expected = dev["runs"][seed]["alternatives"][3]["validation_at_fixed_half"]
        if any(abs(evaluate(y[val], val_probability, 0.5)[metric] - expected[metric]) > 1e-12
               for metric in ("acc", "macro_f1", "auroc")):
            raise ValueError("Official validation result failed exact replay")
        probability = model.predict_proba(test_matrix)[:, 1]
        raw = np.asarray(model.booster_.predict(test_matrix, raw_score=True))
        shap = np.asarray(model.booster_.predict(test_matrix, pred_contrib=True))
        if (shap.shape != (len(test), 301)
                or np.max(np.abs(shap.sum(axis=1) - raw)) > 1e-9
                or np.max(np.abs(expit(raw) - probability)) > 1e-12):
            raise ValueError("Evidence contributions fail exact logit replay")
        pred = probability >= 0.5
        masks = {}
        for group, indices in GROUPS.items():
            masked = test_matrix.copy()
            masked[:, indices] = 0.0
            masked[:, np.asarray(indices) + 150] = 0.0
            masked_probability = model.predict_proba(masked)[:, 1]
            masked_pred = masked_probability >= 0.5
            masks[group] = {
                "changed_test_classes": int(np.count_nonzero(masked_pred != pred)),
                "masked_correct": int(np.count_nonzero(masked_pred == y[test])),
                "full_correct": int(np.count_nonzero(pred == y[test])),
            }
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_rows = baseline["predictions"]
        if (baseline["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(baseline_rows) != len(test)
                or any(item["account_id"] != str(int(i)) or
                       item["label"] != int(y[i])
                       for item, i in zip(baseline_rows, test))):
            raise ValueError("Paired HypBot test IDs, labels or split differ")
        baseline_pred = np.asarray([
            item["probability"] >= baseline["manifest"]["threshold"]
            for item in baseline_rows])
        only_new = int(np.count_nonzero((pred == y[test]) &
                                        (baseline_pred != y[test])))
        only_baseline = int(np.count_nonzero((pred != y[test]) &
                                             (baseline_pred == y[test])))
        disagreement = only_new + only_baseline
        paired_p = (float(binomtest(only_new, disagreement, 0.5).pvalue)
                    if disagreement else 1.0)
        run_dir = folder / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        reference_path = run_dir / "label_blind_reference.json.gz"
        evidence_path = run_dir / "test_evidence.npz"
        joblib.dump(model, model_path)
        reference.save(reference_path)
        np.savez_compressed(
            evidence_path, ids=np.asarray([str(int(i)) for i in test], dtype=str),
            names=np.asarray(names, dtype=str),
            raw_values=x[test].astype(np.float64),
            upper=test_batch.upper, lower=test_batch.lower,
            reliability=test_batch.reliability,
            available=test_batch.available,
            supports=split["supports"][test],
            evidence_matrix=test_matrix,
            shap=shap, logit=raw, probability=probability,
            label=y[test].astype(np.int8))
        cards_path = run_dir / "test_evidence_cards.jsonl"
        with cards_path.open("w", encoding="utf-8") as stream:
            for j, original_index in enumerate(test):
                terms = shap[j, :150] + shap[j, 150:300]
                top = []
                for index in np.argsort(np.abs(terms))[::-1][:12]:
                    k = int(index)
                    group = next(group for group, members in GROUPS.items() if k in members)
                    top.append({
                        "fact": names[k], "source_group": group,
                        "source_level": ("published averaged-tweet tensor plus frozen train-only PCA"
                                         if k >= 86 else
                                         "published profile or typed edge tensors"),
                        "raw_value": (float(x[original_index, k])
                                      if np.isfinite(x[original_index, k]) else None),
                        "upper_tail": float(test_batch.upper[j, k]),
                        "lower_tail": float(test_batch.lower[j, k]),
                        "reliability": float(test_batch.reliability[j, k]),
                        "logit_contribution": float(terms[k]),
                    })
                card = {
                    "account_id": str(int(original_index)),
                    "label": int(y[original_index]),
                    "prediction": int(pred[j]),
                    "probability": float(probability[j]),
                    "logit": float(raw[j]),
                    "intercept_logit": float(shap[j, -1]),
                    "all_fact_logit_contributions": {
                        name: float(value) for name, value in zip(names, terms)},
                    "top_absolute_contributors": top,
                    "published_source": {
                        "features_tensor": str(data_dir / "features.pt"),
                        "edge_index_tensor": str(data_dir / "edge_index.pt"),
                        "edge_type_tensor": str(data_dir / "edge_type.pt"),
                        "account_tensor_row": int(original_index),
                        "projection_artifact": str(args.selected_model_dir /
                                                   f"seed{seed}/projection.npz"),
                    },
                }
                stream.write(json.dumps(card) + "\n")
        predictions_path = run_dir / "test_predictions.json"
        predictions_path.write_text(json.dumps({
            "manifest": {"dataset": "mgtab", "seed": seed, "threshold": 0.5,
                         "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                         "selection_lock_sha256": sha256_file(lock_path),
                         "reference_version": reference.version,
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(reference_path),
                         "evidence_sha256": sha256_file(evidence_path),
                         "cards_sha256": sha256_file(cards_path)},
            "predictions": [{"account_id": str(int(i)), "label": int(y[i]),
                             "probability": float(p)}
                            for i, p in zip(test, probability)]
        }, indent=2) + "\n", encoding="utf-8")
        item = {
            "seed": seed, "test_accounts": len(test),
            "test": evaluate(y[test], probability, 0.5),
            "dev": evaluate(y[val], val_probability, 0.5),
            "hypbot_test_acc": float(np.mean(baseline_pred == y[test])),
            "only_new_correct": only_new,
            "only_hypbot_correct": only_baseline,
            "paired_discordance_exact_two_sided_p": paired_p,
            "group_masks": masks,
            "max_logit_additivity_error": float(np.max(np.abs(shap.sum(axis=1) - raw))),
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(reference_path),
            "evidence_sha256": sha256_file(evidence_path),
            "cards_sha256": sha256_file(cards_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
        summary.append(item)
        print(json.dumps({"seed": seed, "test_acc": item["test"]["acc"],
                          "hypbot_acc": item["hypbot_test_acc"],
                          "only_new_correct": only_new,
                          "only_hypbot_correct": only_baseline}), flush=True)
    report = {
        "scope": "mgtab_exploratory_fixed_test_150_atomic_source_evidence_lightgbm",
        "prior_results_on_same_test_population_already_observed": True,
        "selection_lock_sha256": sha256_file(lock_path),
        "runs": summary,
        "mean_acc": float(np.mean([row["test"]["acc"] for row in summary])),
        "mean_hypbot_acc": float(np.mean([row["hypbot_test_acc"] for row in summary])),
    }
    (folder / "fixed_test_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
