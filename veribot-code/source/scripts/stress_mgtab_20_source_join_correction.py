"""Test repair of a known MGTAB published-content row join error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_content_projection_evidence import transform_with_support
from stress_botsim_12_source_join_correction import hash_order
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def run(args: argparse.Namespace, seed: int, published: np.ndarray) -> dict:
    full_dir = args.full_model_dir / f"seed{seed}"
    compact_dir = args.compact_model_dir / f"seed{seed}"
    full = json.loads((full_dir / "report.json").read_text(encoding="utf-8"))
    report = json.loads((compact_dir / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((compact_dir / "test_predictions.json").read_text(encoding="utf-8"))
    x, y, names, split = selected_facts(args, seed, full)
    manifest = report["manifest"]
    if (manifest != saved["manifest"]
            or manifest["split_sha256"] != full["manifest"]["split_sha256"]
            or manifest["source_features_sha256"] !=
            sha256_file(args.data_dir / "features.pt")):
        raise ValueError("published source or selected compact model differs")
    with np.load(full_dir / "projection.npz", allow_pickle=False) as projection:
        mean = projection["scaler_mean"]
        scale = projection["scaler_scale"]
        center = projection["pca_mean"]
        components = projection["pca_components"]
    if sha256_file(full_dir / "projection.npz") != manifest["projection_artifact_sha256"]:
        raise ValueError("training-only content projection changed")
    reference_path = full_dir / "reference.json.gz"
    if sha256_file(reference_path) != full["manifest"]["reference_artifact_sha256"]:
        raise ValueError("label-blind reference changed")
    reference = LabelBlindReference.load(reference_path)
    model = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    selected_indices = manifest["selected_fact_indices"]
    test = split["test"]
    test_ids = [str(int(i)) for i in test]
    original = x[test].copy()
    normalized = published[test, 20:].copy()
    normalized -= mean
    normalized /= scale
    projected = normalized @ components.T
    projected -= center.reshape(1, -1) @ components.T
    if np.max(np.abs(projected - original[:, 86:])) > 1e-10:
        raise ValueError("content projection does not derive from published tensor")
    clean_atomic = transform_with_support(reference, original,
                                          split["supports"][test])
    clean_probability = model.predict_proba(columns(clean_atomic, selected_indices))
    recorded = saved["predictions"]
    if ([row["account_id"] for row in recorded] != test_ids
            or np.max(np.abs(clean_probability - np.asarray(
                [row["probability"] for row in recorded]))) > 1e-12):
        raise ValueError("clean source does not reproduce frozen decisions")
    recipient_order = hash_order(test_ids, "veribot-mgtab-content-join-recipient-v1")
    donor_order = hash_order(test_ids, "veribot-mgtab-content-join-donor-v1")
    selected = recipient_order[:round(len(test) * 0.25)]
    corrupted = original.copy()
    donors = {}
    for rank, recipient in enumerate(selected):
        donor = next((candidate for candidate in donor_order[rank:] + donor_order[:rank]
                      if candidate != recipient and not np.array_equal(
                          published[int(test[candidate]), 20:],
                          published[int(test[recipient]), 20:])), None)
        if donor is None:
            raise ValueError("no different published content tensor row")
        corrupted[recipient, 86:] = projected[donor]
        donors[recipient] = donor
    bad_atomic = transform_with_support(reference, corrupted,
                                        split["supports"][test])
    bad_probability = model.predict_proba(columns(bad_atomic, selected_indices))
    repaired = corrupted.copy()
    repaired[np.ix_(selected, list(range(86, 150)))] = original[
        np.ix_(selected, list(range(86, 150)))]
    if not np.array_equal(repaired, original, equal_nan=True):
        raise ValueError("source repair did not restore published content tensor")
    repaired_probability = model.predict_proba(columns(transform_with_support(
        reference, repaired, split["supports"][test]), selected_indices))
    if np.max(np.abs(repaired_probability - clean_probability)) > 1e-12:
        raise ValueError("source repair did not restore fixed model probability")
    threshold = float(manifest["threshold"])
    labels = y[test]
    clean_error = (clean_probability >= threshold) != labels
    bad_error = (bad_probability >= threshold) != labels
    repaired_error = (repaired_probability >= threshold) != labels
    selected = np.asarray(selected, dtype=int)
    strategies = {
        "hash_random": sorted(selected, key=lambda i: hashlib.sha256(
            f"veribot-mgtab-correction-order-v1:{test_ids[i]}".encode()).digest()),
        "model_uncertainty": sorted(selected,
                                    key=lambda i: (abs(bad_probability[i] - threshold),
                                                   test_ids[i])),
    }
    budgets = sorted(set((0, 20, 50, 100, 150, len(selected))))
    curves = {}
    for strategy, ordering in strategies.items():
        points = []
        for budget in budgets:
            checked = np.asarray(ordering[:budget], dtype=int)
            source_corrected = bad_error.copy()
            source_corrected[checked] = clean_error[checked]
            direct_labeled = bad_error.copy()
            direct_labeled[checked] = False
            points.append({
                "account_actions": budget,
                "source_correction_total_errors": int(source_corrected.sum()),
                "direct_account_label_total_errors": int(direct_labeled.sum()),
                "source_correction_net_errors_removed": int(
                    bad_error.sum() - source_corrected.sum()),
                "direct_account_label_net_errors_removed": int(
                    bad_error.sum() - direct_labeled.sum()),
            })
        if points[-1]["source_correction_total_errors"] != int(clean_error.sum()):
            raise ValueError("full-budget source repair does not restore clean errors")
        curves[strategy] = points
    rows = []
    for recipient in selected:
        donor = donors[int(recipient)]
        rows.append({
            "account_id": test_ids[recipient],
            "donor_account_id": test_ids[donor],
            "source_locator": {"file": "features.pt", "row": int(test[recipient]),
                               "content_columns_inclusive": [20, 787],
                               "projection_fingerprint": manifest["projection_fingerprint"]},
            "wrong_join_locator": {"file": "features.pt", "row": int(test[donor]),
                                   "content_columns_inclusive": [20, 787]},
            "selected_content_values": {
                names[j]: float(original[recipient, j])
                for j in selected_indices if j >= 86},
            "wrong_join_selected_content_values": {
                names[j]: float(corrupted[recipient, j])
                for j in selected_indices if j >= 86},
            "clean_probability": float(clean_probability[recipient]),
            "corrupted_probability": float(bad_probability[recipient]),
            "repaired_probability": float(repaired_probability[recipient]),
            "label_for_evaluation": int(labels[recipient]),
        })
    return {
        "seed": seed, "split_sha256": manifest["split_sha256"],
        "test_accounts": len(test), "selected_accounts": len(selected),
        "clean_errors": int(clean_error.sum()),
        "corrupted_errors": int(bad_error.sum()),
        "source_corrected_errors": int(repaired_error.sum()),
        "corruption_changed_predictions": int(np.count_nonzero(
            (bad_probability >= threshold) != (clean_probability >= threshold))),
        "corruption_new_errors": int(np.sum(bad_error & ~clean_error)),
        "corruption_removed_errors": int(np.sum(clean_error & ~bad_error)),
        "max_repair_probability_difference": float(np.max(
            np.abs(repaired_probability - clean_probability))),
        "threshold": threshold, "curves": curves, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--structural-facts", required=True, type=Path)
    parser.add_argument("--neighbor-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--compact-model-dir", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.selected_model_dir = args.full_model_dir
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    original_seed0 = json.loads((args.full_model_dir / "seed0" / "report.json")
                                .read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or sha256_file(args.source_audit) !=
            original_seed0["manifest"]["source_audit_sha256"]):
        raise ValueError("published tensor source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("published source tensor shape differs")
    runs = [run(args, seed, published) for seed in (0, 1, 2)]
    result = {
        "dataset": "mgtab", "model": "source_backed_twenty_direct_numeric_concepts",
        "scope": "known_published_content_tensor_row_join_error_injection_and_repair",
        "injection_uses_bot_human_labels": False,
        "account_selection_uses_bot_human_labels": False,
        "source_correction_uses_bot_human_labels": False,
        "comparison_unit": "account actions; reviewer time and natural error frequency not measured",
        "source_truth": "independently verified published feature tensor",
        "content_semantic_truth": False,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{key: row[key] for key in (
        "seed", "test_accounts", "selected_accounts", "clean_errors",
        "corrupted_errors", "source_corrected_errors",
        "corruption_changed_predictions", "corruption_new_errors",
        "corruption_removed_errors")}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
