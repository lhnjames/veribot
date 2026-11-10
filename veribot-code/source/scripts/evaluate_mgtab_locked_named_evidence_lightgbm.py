"""Evaluate the dev-locked MGTAB named-evidence LightGBM against paired HypBot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.mgtab_profile_schema import PROFILE_SCHEMA
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_named_evidence_lightgbm"
    lock_path = folder / "selection_lock.json"
    dev_path = folder / "dev_only.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    prior = json.loads((root / "output/v3_mgtab_named_raw_capacity/dev_only.json")
                       .read_text(encoding="utf-8"))
    if (lock["dev_result_sha256"] != sha256_file(dev_path)
            or lock["selected_config_index"] != 3
            or lock["selected_config"] != [31, 400, 50, 10.0]
            or lock["test_metric_or_prediction_used_to_create_lock"] is not False
            or dev["published_content_projection_used"] is not False):
        raise ValueError("named-evidence capacity is not validation locked")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    if (lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(structural)
            or lock["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("named published sources differ from validation lock")
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(dataset="mgtab", data_dir=data_dir, account_file=None,
                           native_graph_facts=None, structural_facts=structural,
                           neighbor_facts=neighbor, selected_model_dir=full_dir)
    leaves, iterations, min_leaf, l2 = lock["selected_config"]
    rows = []
    for seed in (0, 1, 2):
        old_report = json.loads((full_dir / f"seed{seed}/report.json")
                                .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old_report)
        if (names[:86] != prior["fact_names"]
                or lock["split_sha256_by_seed"][str(seed)] !=
                old_report["manifest"]["split_sha256"]
                or len(names) != 150):
            raise ValueError("selected named fact vocabulary or split differs")
        train, val, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(
            x[train, :86], names[:86])
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("label-blind evidence reference differs")
        x_train = matrix(transform_with_support(
            reference, x[train, :86], split["supports"][train]))
        x_val = matrix(transform_with_support(
            reference, x[val, :86], split["supports"][val]))
        test_batch = transform_with_support(
            reference, x[test, :86], split["supports"][test])
        x_test = matrix(test_batch)
        if (x_train.shape[1] != 172 or x_val.shape[1] != 172
                or x_test.shape[1] != 172 or not np.isfinite(x_test).all()):
            raise ValueError("named evidence-only matrix differs")
        model = LGBMClassifier(
            objective="binary", n_estimators=iterations, num_leaves=leaves,
            min_child_samples=min_leaf, reg_lambda=l2, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
            verbosity=-1)
        model.fit(x_train, y[train])
        dev_probability = model.predict_proba(x_val)[:, 1]
        threshold = best_validation_threshold(dev_probability, y[val])
        dev_metrics = evaluate(y[val], dev_probability, threshold)
        expected = dev["runs"][seed]["alternatives"][lock["selected_config_index"]]
        if (abs(threshold - lock["threshold_by_seed"][str(seed)]) > 1e-12
                or abs(threshold - expected["threshold"]) > 1e-12
                or any(abs(dev_metrics[key] - expected["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("validation result failed exact replay before test")
        probability = model.predict_proba(x_test)[:, 1]
        raw_score = np.asarray(model.booster_.predict(x_test, raw_score=True))
        contribution = np.asarray(model.booster_.predict(x_test,
                                                         pred_contrib=True))
        if contribution.shape != (len(test), 173):
            raise ValueError("evidence contribution dimension differs")
        max_logit_error = float(np.max(np.abs(contribution.sum(axis=1) - raw_score)))
        if max_logit_error > 1e-9:
            raise ValueError("evidence contributions do not reconstruct logit")
        predictions = probability >= threshold
        group_masks = {}
        for group, indices in {
            "published_profile": list(range(20)) + list(range(86, 106)),
            "published_native_relations": list(range(20, 86)) +
                                          list(range(106, 172)),
        }.items():
            masked = x_test.copy()
            masked[:, indices] = 0.0
            masked_probability = model.predict_proba(masked)[:, 1]
            masked_class = masked_probability >= threshold
            group_masks[group] = {
                "input_columns": len(indices),
                "changed_class_accounts": int(np.count_nonzero(
                    predictions != masked_class)),
                "full_correct": int(np.count_nonzero(predictions == y[test])),
                "masked_correct": int(np.count_nonzero(masked_class == y[test])),
            }
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (baseline["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(baseline["predictions"]) != len(test)):
            raise ValueError("paired HypBot test population differs")
        baseline_threshold = baseline["manifest"]["threshold"]
        saved_predictions, cards = [], []
        baseline_correct = []
        for position, index in enumerate(test):
            other = baseline["predictions"][position]
            if other["account_id"] != str(int(index)) or other["label"] != int(y[index]):
                raise ValueError("paired MGTAB account or label differs")
            baseline_correct.append(int(other["probability"] >= baseline_threshold)
                                    == int(y[index]))
            saved_predictions.append({
                "account_id": other["account_id"], "label": int(y[index]),
                "probability": float(probability[position]),
                "prediction": int(predictions[position]),
                "raw_logit": float(raw_score[position]),
                "contribution_sum": float(contribution[position].sum()),
            })
            top = np.argsort(np.abs(contribution[position, :-1]))[-8:][::-1]
            cards.append({
                "account_id": other["account_id"],
                "prediction": int(predictions[position]),
                "probability": float(probability[position]),
                "raw_logit": float(raw_score[position]),
                "contribution_bias": float(contribution[position, -1]),
                "contribution_sum": float(contribution[position].sum()),
                "source_level": "published MGTAB tensors and typed edges",
                "top_absolute_contributions": [
                    {"fact_name": names[int(j % 86)],
                     "tail": "upper" if j < 86 else "lower",
                     "raw_fact_value": (float(x[index, j % 86])
                                        if np.isfinite(x[index, j % 86]) else None),
                     "evidence_input": float(x_test[position, j]),
                     "logit_contribution": float(contribution[position, j]),
                     "published_source": ("features.pt profile column" if j % 86 < 20
                                          else "typed edge derived fact cache"),
                     "paper_table_a5_profile_alias": (PROFILE_SCHEMA[j % 86][0]
                                                      if j % 86 < 20 else None),
                     "published_node_row": int(index)}
                    for j in top],
                "attributions_are_not_individual_causal_proofs": True,
            })
        baseline_correct = np.asarray(baseline_correct, dtype=bool)
        new_correct = predictions == y[test]
        only_new = int(np.count_nonzero(new_correct & ~baseline_correct))
        only_old = int(np.count_nonzero(~new_correct & baseline_correct))
        run_dir = folder / "selected" / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.txt"
        reference_path = run_dir / "reference.json.gz"
        prediction_path = run_dir / "test_predictions.json"
        cards_path = run_dir / "evidence_cards.jsonl"
        model.booster_.save_model(str(model_path))
        reference.save(reference_path)
        prediction_path.write_text(json.dumps({
            "manifest": {"split_sha256": lock["split_sha256_by_seed"][str(seed)],
                         "threshold": threshold,
                         "selection_lock_sha256": sha256_file(lock_path),
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(reference_path)},
            "predictions": saved_predictions}, indent=2) + "\n", encoding="utf-8")
        with cards_path.open("w", encoding="utf-8") as stream:
            for card in cards:
                stream.write(json.dumps(card, ensure_ascii=False) + "\n")
        metrics = evaluate(y[test], probability, threshold)
        rows.append({
            "seed": seed, "n": len(test), "validation": dev_metrics,
            "test": metrics,
            "hypbot_correct": int(np.count_nonzero(baseline_correct)),
            "candidate_correct": int(np.count_nonzero(new_correct)),
            "candidate_only_correct": only_new,
            "hypbot_only_correct": only_old,
            "paired_net_correct": only_new - only_old,
            "discordant_exact_two_sided_p": float(binomtest(
                only_new, only_new + only_old, p=0.5).pvalue),
            "max_evidence_logit_reconstruction_error": max_logit_error,
            "group_masks": group_masks,
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(reference_path),
            "predictions_sha256": sha256_file(prediction_path),
            "evidence_cards_sha256": sha256_file(cards_path),
            "hypbot_predictions_sha256": sha256_file(baseline_path),
        })
        print(seed, round(metrics["acc"] * 100, 2),
              rows[-1]["candidate_correct"], rows[-1]["hypbot_correct"],
              flush=True)
    summary = {
        "dataset": "mgtab",
        "scope": "dev_locked_named_release_source_evidence_lightgbm_fixed_test",
        "selected_config": lock["selected_config"],
        "selection_lock_sha256": sha256_file(lock_path),
        "prior_test_population_previously_explored": True,
        "published_content_projection_used": False,
        "runs": rows,
    }
    destination = folder / "selected/paired_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
