"""Evaluate the dev-locked Fox8 evidence-only classifier on fixed test splits."""

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
from screen_fox8_neighbor_atomic_evidence_capacity_dev import (
    evidence_matrix, load_cache, weighted_atomic)
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def native_neighbors(path: Path, ids: list[str]) -> list[list[str]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    neighbors = {uid: set() for uid in ids}
    for item in source["events"]:
        u, v = str(item["source_id"]), str(item["target_id"])
        if u in neighbors and v in neighbors and u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return [sorted(neighbors[uid]) for uid in ids]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_neighbor_atomic_evidence_capacity"
    lock_path, dev_path = folder / "selection_lock.json", folder / "dev_only.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    raw_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    audit_path = root / "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json"
    if (lock["dev_result_sha256"] != sha256_file(dev_path)
            or lock["raw_neighbor_sha256"] != sha256_file(raw_path)
            or lock["raw_neighbor_audit_sha256"] != sha256_file(audit_path)
            or lock["selected"]["variant"] != "raw_neighbor_74"
            or lock["selected"]["config_index"] != 0
            or lock["config"] != [15, 200, 30, 10.0]
            or lock["threshold"] != 0.5
            or lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("Fox8 candidate was not validation locked")
    ids, new_names, new_values, supports = load_cache(raw_path)
    if len(new_names) != 31:
        raise ValueError("Native neighbor fact count differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
    import pandas as pd
    source_ids = pd.read_csv(data_dir / "features.csv")["user_id"].astype(str).tolist()
    if ids != source_ids:
        raise ValueError("Native neighbor account order differs")
    event_index_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    neighbor_ids = native_neighbors(event_index_path, ids)
    args = SimpleNamespace(
        dataset="fox8", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    summary = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if (len(names) != 43 or old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("Fox8 official split or evidence vocabulary differs")
        full = np.column_stack((x, new_values))
        fact_names = names + new_names
        train, dev_index, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(
            full[train], fact_names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Label-blind reference differs from dev lock")
        train_batch = weighted_atomic(reference, full[train], supports[train], 43)
        dev_batch = weighted_atomic(reference, full[dev_index], supports[dev_index], 43)
        test_batch = weighted_atomic(reference, full[test], supports[test], 43)
        x_train, x_dev, x_test = (evidence_matrix(batch)
                                  for batch in (train_batch, dev_batch, test_batch))
        model = LGBMClassifier(
            objective="binary", n_estimators=200, num_leaves=15,
            min_child_samples=30, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1)
        model.fit(x_train, y[train])
        dev_probability = model.predict_proba(x_dev)[:, 1]
        expected = dev["runs"][seed]["variants"]["raw_neighbor_74"]["model_results"][0]
        if any(abs(evaluate(y[dev_index], dev_probability, 0.5)[key] -
                   expected["validation_at_half"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("Official dev metrics failed exact replay before test")
        probability = model.predict_proba(x_test)[:, 1]
        raw_score = np.asarray(model.booster_.predict(x_test, raw_score=True))
        shap = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
        if (shap.shape != (len(test), 149)
                or np.max(np.abs(shap.sum(axis=1) - raw_score)) > 1e-9
                or np.max(np.abs(expit(raw_score) - probability)) > 1e-12):
            raise ValueError("Evidence contributions fail exact decision replay")
        threshold = lock["threshold"]
        pred = probability >= threshold
        masked = x_test.copy()
        masked[:, 43:74] = 0.0
        masked[:, 117:148] = 0.0
        masked_probability = model.predict_proba(masked)[:, 1]
        masked_pred = masked_probability >= threshold
        baseline_path = root / f"output/hypbot_fox8_native_graph_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline["manifest"]["split_sha256"] != lock["split_sha256_by_seed"][str(seed)]:
            raise ValueError("Native HypBot paired split differs")
        baseline_rows = baseline["predictions"]
        if len(baseline_rows) != len(test) or any(
                row["account_id"] != ids[int(i)] or row["label"] != int(y[i])
                for row, i in zip(baseline_rows, test)):
            raise ValueError("Native HypBot test account order differs")
        baseline_pred = np.asarray([
            row["probability"] >= baseline["manifest"]["threshold"]
            for row in baseline_rows])
        veribot_only = int(np.count_nonzero((pred == y[test]) &
                                            (baseline_pred != y[test])))
        hypbot_only = int(np.count_nonzero((pred != y[test]) &
                                           (baseline_pred == y[test])))
        discordant = veribot_only + hypbot_only
        p_value = (float(binomtest(veribot_only, discordant, 0.5).pvalue)
                   if discordant else 1.0)
        run_dir = folder / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        ref_path = run_dir / "label_blind_reference.json.gz"
        evidence_path = run_dir / "test_evidence.npz"
        joblib.dump(model, model_path)
        reference.save(ref_path)
        np.savez_compressed(
            evidence_path, ids=np.asarray([ids[int(i)] for i in test], dtype=str),
            names=np.asarray(fact_names, dtype=str),
            raw_values=full[test].astype(np.float64),
            upper=test_batch.upper, lower=test_batch.lower,
            reliability=test_batch.reliability,
            available=test_batch.available, supports=supports[test],
            evidence_matrix=x_test, shap=shap,
            logit=raw_score, probability=probability,
            neighbor_mask_probability=masked_probability,
            label=y[test].astype(np.int8))
        card_path = run_dir / "test_evidence_cards.jsonl"
        with card_path.open("w", encoding="utf-8") as stream:
            for j, original_index in enumerate(test):
                fact_contribution = shap[j, :74] + shap[j, 74:148]
                ordered = np.argsort(np.abs(fact_contribution))[::-1]
                top = []
                for rank in ordered[:12]:
                    k = int(rank)
                    top.append({
                        "fact": fact_names[k],
                        "raw_value": (float(full[original_index, k])
                                      if np.isfinite(full[original_index, k]) else None),
                        "upper_tail": float(test_batch.upper[j, k]),
                        "lower_tail": float(test_batch.lower[j, k]),
                        "reliability": float(test_batch.reliability[j, k]),
                        "logit_contribution": float(fact_contribution[k]),
                        "source": ("native reply-or-retweet neighbor IDs + named "
                                   "features.csv values" if k >= 43 else
                                   "target account named profile, sampled tweet, "
                                   "or native interaction fact"),
                    })
                card = {
                    "account_id": ids[int(original_index)],
                    "label": int(y[original_index]),
                    "prediction": int(pred[j]),
                    "probability": float(probability[j]),
                    "intercept_logit": float(shap[j, -1]),
                    "logit": float(raw_score[j]),
                    "all_fact_logit_contributions": {
                        name: float(value) for name, value in
                        zip(fact_names, fact_contribution)},
                    "top_absolute_contributors": top,
                    "native_neighbor_account_ids": neighbor_ids[int(original_index)],
                    "native_source_locator": {
                        "feature_table": str(data_dir / "features.csv"),
                        "native_relation_events": str(event_index_path),
                    },
                }
                stream.write(json.dumps(card, ensure_ascii=False) + "\n")
        predictions_path = run_dir / "test_predictions.json"
        predictions = {
            "manifest": {"dataset": "fox8", "seed": seed,
                         "threshold": threshold,
                         "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                         "reference_version": reference.version,
                         "selection_lock_sha256": sha256_file(lock_path),
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(ref_path),
                         "evidence_sha256": sha256_file(evidence_path),
                         "cards_sha256": sha256_file(card_path)},
            "predictions": [{"account_id": ids[int(i)], "label": int(y[i]),
                             "probability": float(p)}
                            for i, p in zip(test, probability)],
        }
        predictions_path.write_text(json.dumps(predictions, indent=2) + "\n",
                                    encoding="utf-8")
        row = {
            "seed": seed, "test_accounts": len(test),
            "test": evaluate(y[test], probability, threshold),
            "dev": evaluate(y[dev_index], dev_probability, threshold),
            "native_hypbot_test_acc": float(np.mean(baseline_pred == y[test])),
            "paired_veribot_only_correct": veribot_only,
            "paired_native_hypbot_only_correct": hypbot_only,
            "paired_discordance_exact_two_sided_p": p_value,
            "native_neighbor_evidence_mask": {
                "changed_test_classes": int(np.count_nonzero(pred != masked_pred)),
                "full_test_correct": int(np.count_nonzero(pred == y[test])),
                "masked_test_correct": int(np.count_nonzero(masked_pred == y[test]))},
            "max_logit_additivity_error": float(np.max(np.abs(
                shap.sum(axis=1) - raw_score))),
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(ref_path),
            "evidence_sha256": sha256_file(evidence_path),
            "cards_sha256": sha256_file(card_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
        summary.append(row)
        print(json.dumps({"seed": seed, "test_acc": row["test"]["acc"],
                          "native_hypbot_acc": row["native_hypbot_test_acc"],
                          "neighbor_mask_changed": row["native_neighbor_evidence_mask"][
                              "changed_test_classes"]}), flush=True)
    output = {
        "scope": "Fox8_exploratory_fixed_test_after_official_dev_lock",
        "prior_test_results_on_same_account_population_already_observed": True,
        "selection_lock_sha256": sha256_file(lock_path),
        "source_feature_sha256": sha256_file(data_dir / "features.csv"),
        "native_event_index_sha256": sha256_file(event_index_path),
        "raw_neighbor_sha256": sha256_file(raw_path),
        "runs": summary,
        "mean_acc": float(np.mean([row["test"]["acc"] for row in summary])),
        "mean_native_hypbot_acc": float(np.mean([
            row["native_hypbot_test_acc"] for row in summary])),
    }
    (folder / "fixed_test_report.json").write_text(json.dumps(output, indent=2) + "\n",
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
