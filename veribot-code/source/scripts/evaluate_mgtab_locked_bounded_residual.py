"""Evaluate the dev-locked 86-fact evidence plus explicit bounded residual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from screen_mgtab_bounded_content_residual import fit_residual
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for, rows
from verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    destination = args.output_dir.resolve()
    data = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    lock_path = root / "output/v3_mgtab_stronger_bounded_residual/selection_lock.json"
    dev_path = root / "output/v3_mgtab_stronger_bounded_residual/dev_only.json"
    lock = read(lock_path)
    dev = read(dev_path)
    cfg = lock["candidate"]
    if (lock["dev_result_sha256"] != sha256_file(dev_path) or
            cfg["named_evidence_facts"] != 86 or
            cfg["residual_dimensions"] != 16 or
            cfg["absolute_coefficient_bound"] != 0.5 or
            cfg["theoretical_absolute_residual_logit_cap"] != 8.0 or
            lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("dev selection lock differs")
    published = torch.load(data / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    runs = []
    for seed in (0, 1, 2):
        original_path = root / f"output/v3_mgtab_ab_spline_content/selected/seed{seed}/report.json"
        if sha256_file(original_path) != lock["full_reference_report_sha256_by_seed"][str(seed)]:
            raise ValueError("full 86+64 reference report differs")
        original = read(original_path)
        reference_dev = original["validation"]["macro_f1"]
        previous = dev["runs"][seed]
        selected = previous["candidates"]["dimensions_16_bound_0.5"]
        if reference_dev - selected["validation"]["macro_f1"] > 0.03 + 1e-12:
            raise ValueError("locked candidate misses prespecified dev tolerance")
        x, y, ids, names, split, source = load("mgtab", data, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, data, structural)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, data, neighbor)
        train, val = split["train"], split["val"]
        if source["split_sha256"] != previous["split_sha256"]:
            raise ValueError("fixed split differs")
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if reference.version != previous["reference_version"]:
            raise ValueError("reference differs from locked dev run")
        evidence = evidence_with_neighbor_support(reference, x, supports)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        model = MonotoneSplineReasoner(groups_for("mgtab", names), l2=1e-5)
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        base_b = model.logits(rows(evidence, b))
        base_val = model.logits(rows(evidence, val))
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=16, svd_solver="randomized", random_state=seed).fit(
            normalized[train])
        projected = pca.transform(normalized)
        projected /= np.maximum(np.sqrt(pca.explained_variance_), 1e-6)
        z = np.tanh(projected)
        weights = fit_residual(base_b, z[b], y[b], 0.5)
        p_val = expit(base_val + z[val] @ weights)
        threshold = float(lock["threshold_by_seed"][str(seed)])
        measured_dev = evaluate(y[val], p_val, threshold)
        if (abs(threshold - selected["threshold"]) > 1e-12 or
                np.max(np.abs(weights - selected["coefficients"])) > 1e-8 or
                any(abs(measured_dev[name] - selected["validation"][name]) > 1e-9
                    for name in ("acc", "macro_f1", "auroc"))):
            raise ValueError("locked dev model did not replay")

        # Test labels and predictions are first indexed after the locked dev
        # model, its threshold and the three dev metrics have been replayed.
        test = split["test"]
        saved_path = root / f"output/v3_mgtab_ab_spline_content/direct_selected_20/seed{seed}/test_predictions.json"
        hypbot_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        saved = read(saved_path)
        hypbot = read(hypbot_path)
        if (saved["manifest"]["split_sha256"] != source["split_sha256"] or
                hypbot["manifest"]["split_sha256"] != source["split_sha256"] or
                len(test) != len(saved["predictions"]) or
                len(test) != len(hypbot["predictions"])):
            raise ValueError("fixed test comparison population differs")
        for i, prior, other in zip(test, saved["predictions"],
                                   hypbot["predictions"], strict=True):
            if (str(ids[i]) != prior["account_id"] or
                    str(ids[i]) != other["account_id"] or
                    int(y[i]) != prior["label"] or
                    int(y[i]) != other["label"]):
                raise ValueError("paired test account or label differs")
        base_test = model.logits(rows(evidence, test))
        residual_test = z[test] @ weights
        probability = expit(base_test + residual_test)
        candidate = probability >= threshold
        baseline_threshold = hypbot["manifest"]["threshold"]
        baseline = np.asarray([row["probability"] >= baseline_threshold
                               for row in hypbot["predictions"]], dtype=bool)
        truth = y[test].astype(bool)
        candidate_metric = evaluate(y[test], probability, threshold)
        baseline_acc = float(np.mean(baseline == truth))
        saved_threshold = saved["manifest"]["threshold"]
        selected_prediction = np.asarray([
            row["probability"] >= saved_threshold for row in saved["predictions"]],
            dtype=bool)
        run_dir = destination / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        reference_path = run_dir / "reference.json.gz"
        reference.save(reference_path)
        reasoner_path = run_dir / "reasoner.json"
        reasoner_path.write_text(json.dumps(model.manifest(), indent=2) + "\n",
                                 encoding="utf-8")
        residual_path = run_dir / "residual_transform.npz"
        np.savez_compressed(residual_path, scaler_mean=scaler.mean_,
                            scaler_scale=scaler.scale_, pca_mean=pca.mean_,
                            pca_components=pca.components_,
                            pca_explained_variance=pca.explained_variance_,
                            residual_coefficients=weights)
        prediction_path = run_dir / "test_predictions.json"
        prediction_path.write_text(json.dumps({
            "manifest": {"dataset": "mgtab", "seed": seed,
                         "split_sha256": source["split_sha256"],
                         "selection_lock_sha256": sha256_file(lock_path),
                         "threshold": threshold,
                         "reference_sha256": sha256_file(reference_path),
                         "reasoner_sha256": sha256_file(reasoner_path),
                         "residual_transform_sha256": sha256_file(residual_path)},
            "predictions": [{"account_id": str(ids[i]), "label": int(y[i]),
                             "probability": float(probability[j]),
                             "evidence_logit": float(base_test[j]),
                             "residual_logit": float(residual_test[j])}
                            for j, i in enumerate(test)]}, indent=2) + "\n",
            encoding="utf-8")
        runs.append({
            "seed": seed, "test_n": len(test),
            "candidate": candidate_metric,
            "hypbot_acc": baseline_acc,
            "selected_20_evidence_acc": float(np.mean(selected_prediction == truth)),
            "candidate_minus_hypbot_acc": candidate_metric["acc"] - baseline_acc,
            "candidate_correct_hypbot_wrong": int(np.sum((candidate == truth) &
                                                           (baseline != truth))),
            "candidate_wrong_hypbot_correct": int(np.sum((candidate != truth) &
                                                           (baseline == truth))),
            "residual_changes_evidence_only_class_at_locked_threshold": int(
                np.count_nonzero((base_test + residual_test >=
                                  np.log(threshold / (1 - threshold))) !=
                                 (base_test >= np.log(threshold / (1 - threshold))))),
            "observed_max_absolute_residual_logit_test": float(
                np.max(np.abs(residual_test))),
            "predictions_sha256": sha256_file(prediction_path),
            "hypbot_predictions_sha256": sha256_file(hypbot_path),
            "selected_predictions_sha256": sha256_file(saved_path),
        })
        print(seed, runs[-1]["candidate"]["acc"], baseline_acc, flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "dev_locked_exploratory_fixed_test_bounded_content_residual",
               "selection_lock_sha256": sha256_file(lock_path),
               "prior_test_results_on_same_population_already_observed": True,
               "runs": runs}
    (destination / "paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
