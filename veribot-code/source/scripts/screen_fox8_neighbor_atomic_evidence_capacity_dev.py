"""Dev-only test whether 31 source-audited neighbor facts survive evidence encoding."""

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
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


CONFIGS = ((15, 200, 30, 10.0), (31, 400, 50, 10.0))


def evidence_matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def weighted_atomic(reference: LabelBlindReference, values: np.ndarray,
                    supports: np.ndarray, old_count: int) -> EvidenceBatch:
    batch = reference.transform(values)
    reliability = batch.reliability.copy()
    reliability[:, old_count:] *= supports / (supports + 5.0)
    return EvidenceBatch(batch.upper, batch.lower, reliability,
                         batch.available, batch.reference_version, batch.names)


def load_cache(path: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        return (cache["ids"].astype(str).tolist(),
                cache["names"].astype(str).tolist(),
                cache["values"].astype(float),
                cache["supports"].astype(float))


def run_model(x_train: np.ndarray, x_dev: np.ndarray, y_train: np.ndarray,
              y_dev: np.ndarray, config: tuple[int, int, int, float],
              old_fact_count: int, names: list[str]) -> dict:
    leaves, iterations, min_leaf, l2 = config
    model = LGBMClassifier(
        objective="binary", n_estimators=iterations, num_leaves=leaves,
        min_child_samples=min_leaf, reg_lambda=l2, learning_rate=0.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_dev)[:, 1]
    threshold = best_validation_threshold(probability, y_dev)
    probe = x_dev[:min(32, len(x_dev))]
    raw = np.asarray(model.booster_.predict(probe, raw_score=True))
    contribution = np.asarray(model.booster_.predict(probe, pred_contrib=True))
    additivity = float(np.max(np.abs(contribution.sum(axis=1) - raw)))
    if additivity > 1e-9 or contribution.shape != (len(probe), x_dev.shape[1] + 1):
        raise ValueError("Decision contributions do not reconstruct score")
    result = {
        "config": {"num_leaves": leaves, "n_estimators": iterations,
                   "min_child_samples": min_leaf, "reg_lambda": l2},
        "evidence_input_count": int(x_dev.shape[1]),
        "validation_at_half": evaluate(y_dev, probability, 0.5),
        "validation_selected_threshold": threshold,
        "validation_at_selected_threshold": evaluate(y_dev, probability, threshold),
        "max_probe_logit_additivity_error": additivity,
    }
    if x_dev.shape[1] > 2 * old_fact_count:
        masked = x_dev.copy()
        masked[:, old_fact_count: x_dev.shape[1] // 2] = 0.0
        masked[:, x_dev.shape[1] // 2 + old_fact_count:] = 0.0
        p_mask = model.predict_proba(masked)[:, 1]
        result["native_neighbor_evidence_mask_changed_classes_at_half"] = int(
            np.count_nonzero((p_mask >= 0.5) != (probability >= 0.5)))
        result["native_neighbor_evidence_mask_changed_classes_at_selected"] = int(
            np.count_nonzero((p_mask >= threshold) != (probability >= threshold)))
        importance = model.booster_.feature_importance(importance_type="gain")
        result["native_neighbor_gain_share"] = float((
            importance[old_fact_count: x_dev.shape[1] // 2].sum() +
            importance[x_dev.shape[1] // 2 + old_fact_count:].sum()) /
            max(importance.sum(), 1e-12))
        fact_gain = importance[:len(names)] + importance[len(names):]
        top = np.argsort(fact_gain)[::-1][:10]
        result["top_fact_gain"] = [
            {"fact": names[int(j)], "gain_share": float(fact_gain[j] /
                                                        max(fact_gain.sum(), 1e-12))}
            for j in top]
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
    raw_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    raw_audit_path = root / "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json"
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    if (raw_audit["status"] != "sampled_raw_source_neighbor_means_verified"
            or raw_audit["fact_artifact_sha256"] != sha256_file(raw_path)):
        raise ValueError("Raw neighbor source audit differs")
    raw_ids, raw_names, raw_values, raw_supports = load_cache(raw_path)
    scaled_audit_path = root / "research/certificates/fox8_scaled_native_neighbor_aggregate_audit.json"
    scaled_audit = json.loads(scaled_audit_path.read_text(encoding="utf-8"))
    if scaled_audit["scope"] != "train_only_scaled_clipped_named_native_neighbor_aggregate":
        raise ValueError("Scaled neighbor source audit differs")
    args = SimpleNamespace(
        dataset="fox8", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    import pandas as pd
    ids = pd.read_csv(data_dir / "features.csv")["user_id"].astype(str).tolist()
    if ids != raw_ids:
        raise ValueError("Raw neighbor order differs")
    runs = []
    for seed in (0, 1, 2):
        report = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        if len(names) != 43 or len(raw_names) != 31:
            raise ValueError("Evidence vocabulary differs")
        scaled_path = root / f"data/processed/fox8_v3/scaled_native_neighbor_seed{seed}.npz"
        scaled_run = next(item for item in scaled_audit["runs"] if item["seed"] == seed)
        if scaled_run["artifact_sha256"] != sha256_file(scaled_path):
            raise ValueError("Scaled fact hash differs")
        scaled_ids, scaled_names, scaled_values, scaled_supports = load_cache(scaled_path)
        if ids != scaled_ids or not np.array_equal(raw_supports, scaled_supports):
            raise ValueError("Scaled neighbor support or order differs")
        status_name = "native_interaction_neighbor_mean_statuses_count"
        status_index = raw_names.index(status_name)
        keep_neighbor = [j for j in range(len(raw_names)) if j != status_index]
        train, dev = split["train"], split["val"]
        variants = {}
        for variant, new_names, new_values, supports in (
                ("control_43", [], None, None),
                ("raw_neighbor_74", raw_names, raw_values, raw_supports),
                ("raw_neighbor_73_without_neighbor_statuses_count",
                 [raw_names[j] for j in keep_neighbor],
                 raw_values[:, keep_neighbor], raw_supports[:, keep_neighbor]),
                ("hypbot_scaled_neighbor_74", scaled_names, scaled_values, scaled_supports)):
            full = x if new_values is None else np.column_stack((x, new_values))
            reference = LabelBlindReference(min_context_n=30).fit(
                full[train], names + new_names)
            if new_values is None:
                batch_train = reference.transform(full[train])
                batch_dev = reference.transform(full[dev])
            else:
                batch_train = weighted_atomic(reference, full[train],
                                              supports[train], len(names))
                batch_dev = weighted_atomic(reference, full[dev],
                                            supports[dev], len(names))
            matrix_train = evidence_matrix(batch_train)
            matrix_dev = evidence_matrix(batch_dev)
            if not np.isfinite(matrix_train).all() or not np.isfinite(matrix_dev).all():
                raise ValueError("Evidence input is nonfinite")
            variants[variant] = {
                "reference_version": reference.version,
                "source_names": names + new_names,
                "model_results": [run_model(matrix_train, matrix_dev, y[train],
                                            y[dev], config, len(names), names + new_names)
                                  for config in CONFIGS],
            }
        runs.append({"seed": seed, "split_sha256": report["manifest"]["split_sha256"],
                     "variants": variants})
        print(json.dumps({"seed": seed, "fixed_half_errors": {
            key: [item["validation_at_half"]["errors"]
                  if "errors" in item["validation_at_half"] else
                  int(round((1.0 - item["validation_at_half"]["acc"]) * len(dev)))
                  for item in value["model_results"]]
            for key, value in variants.items()}}), flush=True)
    output = {
        "scope": "fox8_train_dev_only_atomic_source_evidence_neighbor_capacity",
        "test_labels_used_for_fit_or_selection": False,
        "new_test_predictions_or_metrics_computed": False,
        "lightgbm_version": lightgbm.__version__,
        "raw_neighbor_sha256": sha256_file(raw_path),
        "raw_neighbor_audit_sha256": sha256_file(raw_audit_path),
        "scaled_neighbor_audit_sha256": sha256_file(scaled_audit_path),
        "configs": CONFIGS, "runs": runs,
    }
    path = root / "output/v3_fox8_neighbor_atomic_evidence_capacity/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
