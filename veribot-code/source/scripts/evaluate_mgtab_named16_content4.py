"""Evaluate the validation-locked MGTAB 16 named + 4 content evidence budget."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_mgtab_sparse_evidence_cards import locator
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_atomic, rank_content
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def mask_all(evidence):
    high = np.zeros_like(evidence.upper)
    low = np.zeros_like(evidence.lower)
    return type(evidence)(high, low, evidence.reliability, evidence.available,
                          evidence.reference_version, evidence.names)


def run(args: argparse.Namespace, seed: int, screen: dict, lock: dict) -> dict:
    original_dir = args.selected_model_dir / f"seed{seed}"
    original = json.loads((original_dir / "report.json").read_text(encoding="utf-8"))
    x, y, names, split = selected_facts(args, seed, original)
    reference_path = original_dir / "reference.json.gz"
    reference = LabelBlindReference.load(reference_path)
    if (reference.version != original["manifest"]["reference_version"]
            or sha256_file(reference_path) !=
            original["manifest"]["reference_artifact_sha256"]):
        raise ValueError("label-blind reference differs")
    train, dev, test = split["train"], split["val"], split["test"]
    atomic_train = transform_with_support(reference, x[train], split["supports"][train])
    atomic_dev = transform_with_support(reference, x[dev], split["supports"][dev])
    atomic_test = transform_with_support(reference, x[test], split["supports"][test])
    a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                            stratify=y[train], random_state=seed)
    structural = rank_atomic(atomic_train, y[train], a, list(range(86)), 16)
    content = rank_content(atomic_train, y[train], a, 4)
    selected = structural + content
    frozen = next(row for row in screen["runs"] if row["seed"] == seed)
    chosen = next(row for row in frozen["budgets"]
                  if row["paper_named_profile_or_relation_count"] == 16)
    if ([names[j] for j in structural] != chosen["selected_named_facts"]
            or [names[j] for j in content] != chosen["selected_content_facts"]
            or original["manifest"]["split_sha256"] != frozen["split_sha256"]):
        raise ValueError("training-only feature selection differs from lock")
    evidence_train = columns(atomic_train, selected)
    evidence_dev = columns(atomic_dev, selected)
    evidence_test = columns(atomic_test, selected)
    original_groups = original["reasoner"]["groups"]
    groups = {group: [name for name in members if name in evidence_train.names]
              for group, members in original_groups.items()}
    groups = {group: members for group, members in groups.items() if members}
    model = MonotoneSplineReasoner(groups, l2=lock["selected_l2"])
    model.fit(rows(evidence_train, a), y[train][a],
              rows(evidence_train, b), y[train][b])
    dev_probability = model.predict_proba(evidence_dev)
    threshold = best_validation_threshold(dev_probability, y[dev])
    validation = evaluate(y[dev], dev_probability, threshold)
    expected = chosen
    if (any(abs(validation[key] - expected["validation"][key]) > 1e-12
            for key in ("acc", "macro_f1", "auroc"))
            or abs(threshold - expected["threshold"]) > 1e-12):
        raise ValueError("validation result differs from selection lock")
    probability = model.predict_proba(evidence_test)
    basis = model._design(evidence_test)
    contributions = np.column_stack([
        basis[:, model._slices[name]] @ model.coef[model._slices[name]]
        for name in evidence_test.names])
    logit = model.intercept + contributions.sum(axis=1)
    if np.max(np.abs(expit(logit) - probability)) > 1e-12:
        raise ValueError("twenty facts do not reconstruct test probabilities")
    if np.max(np.abs(model.predict_proba(mask_all(evidence_test)) -
                     expit(model.intercept))) > 1e-12:
        raise ValueError("hidden bypass of twenty-fact bottleneck")
    grouped = model.contributions(evidence_test)
    group_mask_changes = {}
    for group, members in groups.items():
        removed = set(members)
        high = evidence_test.upper.copy()
        low = evidence_test.lower.copy()
        for j, name in enumerate(evidence_test.names):
            if name in removed:
                high[:, j] = 0.0
                low[:, j] = 0.0
        masked = type(evidence_test)(high, low, evidence_test.reliability,
                                     evidence_test.available,
                                     evidence_test.reference_version,
                                     evidence_test.names)
        group_mask_changes[group] = int(np.count_nonzero(
            (model.predict_proba(masked) >= threshold) != (probability >= threshold)))
    out = args.output_dir / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cards_path = out / "evidence_cards.jsonl.gz"
    predictions = []
    with cards_path.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                for offset, index in enumerate(test):
                    account_id = str(int(index))
                    terms = []
                    for j, fact_index in enumerate(selected):
                        name = names[fact_index]
                        terms.append({
                            "name": name,
                            "source_locator": locator(fact_index, name, int(index),
                                                      original["manifest"]["projection_fingerprint"]),
                            "source_value": float(x[int(index), fact_index]),
                            "available": bool(evidence_test.available[offset, j]),
                            "upper_tail_score": float(evidence_test.upper[offset, j]),
                            "lower_tail_score": float(evidence_test.lower[offset, j]),
                            "reliability": float(evidence_test.reliability[offset, j]),
                            "selected_tail": model.tail[j],
                            "fixed_direction": int(model.sign[j]),
                            "knots": list(model.knots[name]),
                            "nonnegative_slopes": model.coef[model._slices[name]].tolist(),
                            "logit_contribution": float(contributions[offset, j]),
                        })
                    card = {
                        "dataset": "mgtab", "seed": seed, "account_id": account_id,
                        "source_level": "published_profile_edge_and_mean_tweet_tensors",
                        "source_features_sha256": original["manifest"]["source_features_sha256"],
                        "projection_artifact_sha256": original["manifest"]["projection_artifact_sha256"],
                        "reference_version": reference.version,
                        "selection_lock_sha256": sha256_file(args.lock),
                        "intercept": float(model.intercept), "terms": terms,
                        "group_contributions": {group: float(value[offset])
                                                for group, value in grouped.items()},
                        "logit": float(logit[offset]),
                        "probability": float(probability[offset]),
                        "threshold": float(threshold),
                        "prediction": int(probability[offset] >= threshold),
                    }
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
                    predictions.append({"account_id": account_id,
                                        "label": int(y[index]), "split": "test",
                                        "probability": float(probability[offset]),
                                        "group_contributions": card["group_contributions"]})
    manifest = {"dataset": "mgtab", "seed": seed,
                "split_sha256": original["manifest"]["split_sha256"],
                "source_files_sha256": original["manifest"]["files_sha256"],
                "source_features_sha256": original["manifest"]["source_features_sha256"],
                "projection_artifact_sha256": original["manifest"]["projection_artifact_sha256"],
                "projection_fingerprint": original["manifest"]["projection_fingerprint"],
                "reference_version": reference.version,
                "concept_count": 20,
                "selected_fact_indices": selected,
                "selected_fact_names": [names[j] for j in selected],
                "selection_lock_sha256": sha256_file(args.lock),
                "validation_screen_sha256": sha256_file(args.screen),
                "threshold": float(threshold),
                "prior_test_results_on_same_population_already_observed": True,
                "content_source_level": "published_averaged_tweet_tensor",
                "semantic_content_ground_truth": False}
    report = {"manifest": manifest, "reasoner": model.manifest(),
              "validation": validation,
              "test": evaluate(y[test], probability, threshold),
              "group_mask_prediction_changes": group_mask_changes,
              "all_evidence_mask_probability": float(expit(model.intercept)),
              "cards_sha256": sha256_file(cards_path),
              "test_accounts": len(test)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": manifest, "predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "validation": validation, "test": report["test"],
            "group_mask_prediction_changes": group_mask_changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--structural-facts", required=True, type=Path)
    parser.add_argument("--neighbor-facts", required=True, type=Path)
    parser.add_argument("--selected-model-dir", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if (screen["dataset"] != "mgtab" or lock["dataset"] != "mgtab"
            or sha256_file(args.screen) != lock["validation_screen_sha256"]
            or lock["selected_l2"] != 1e-5
            or lock["named_count"] != 16 or lock["content_count"] != 4
            or screen["test_predictions_accessed"]):
        raise ValueError("MGTAB twenty-fact validation selection differs")
    print(json.dumps([run(args, seed, screen, lock) for seed in (0, 1, 2)], indent=2))


if __name__ == "__main__":
    main()
