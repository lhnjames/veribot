"""Rebuild saved computable-evidence predictions from released model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import (EvidenceBatch, LabelBlindReference,
                                                SignConstrainedReasoner)
from verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner
from screen_mgtab_content_projection_evidence import fingerprint
from gate0_shallow_tree import load, sha256_file
from train_v3_computable import attach_mgtab_structural, attach_mgtab_neighbor
from verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support


def verify(args: argparse.Namespace, seed: int) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    predictions = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    x, y, ids, names, split, current = load(args.dataset, args.data_dir,
                                               args.account_file, seed,
                                               args.native_graph_facts)
    if args.fox8_native_facts is not None:
        if args.dataset != "fox8" or args.fox8_native_source_audit is None or \
                args.fox8_raw is None:
            raise ValueError("Fox8 native facts require raw source and full audit")
        manifest_native = json.loads(args.fox8_native_facts.with_suffix(".json").read_text(
            encoding="utf-8"))
        audit_native = json.loads(args.fox8_native_source_audit.read_text(
            encoding="utf-8"))
        event_index = args.fox8_native_facts.with_name("interaction_event_index.json")
        if (manifest_native["facts_sha256"] != sha256_file(args.fox8_native_facts)
                or manifest_native["raw_source_sha256"] != sha256_file(args.fox8_raw)
                or manifest_native["event_index_sha256"] != sha256_file(event_index)
                or audit_native["status"] !=
                "verified_all_native_interaction_fact_cells_and_event_locators"
                or audit_native["fact_cache_sha256"] != manifest_native["facts_sha256"]):
            raise ValueError("Fox8 native source facts or audit differ")
        with np.load(args.fox8_native_facts, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("Fox8 native fact account order differs")
            x = np.column_stack((x, cache["values"].astype(float)))
            names += cache["names"].astype(str).tolist()
    if args.mgtab_structural_facts is not None:
        if args.dataset != "mgtab":
            raise ValueError("MGTAB structural fact cache applies only to MGTAB")
        x, names, current = attach_mgtab_structural(
            x, names, ids, current, args.data_dir, args.mgtab_structural_facts)
    if args.mgtab_neighbor_facts is not None and args.dataset != "mgtab":
        raise ValueError("MGTAB neighbor facts apply only to MGTAB")
    x, names, neighbor_supports, current = attach_mgtab_neighbor(
        x, names, ids, current, args.data_dir, args.mgtab_neighbor_facts)
    saved = report["manifest"]
    if args.fox8_native_facts is not None:
        if (saved["native_facts_sha256"] != sha256_file(args.fox8_native_facts)
                or saved["native_source_audit_sha256"] !=
                sha256_file(args.fox8_native_source_audit)
                or saved["native_event_index_sha256"] != sha256_file(event_index)):
            raise ValueError("Fox8 saved native evidence differs")
    if args.mgtab_content_projection:
        if args.dataset != "mgtab" or len(names) != 86:
            raise ValueError("content projection requires MGTAB's 86 source facts")
        projection_path = directory / "projection.npz"
        if sha256_file(projection_path) != saved["projection_artifact_sha256"] or \
                sha256_file(args.data_dir / "features.pt") != saved["source_features_sha256"]:
            raise ValueError("content projection or published tensor differs")
        with np.load(projection_path, allow_pickle=False) as artifact:
            mean = artifact["scaler_mean"]
            scale = artifact["scaler_scale"]
            pca_mean = artifact["pca_mean"]
            components = artifact["pca_components"]
            train_indices = artifact["train_indices"]
            added_names = artifact["feature_names"].astype(str).tolist()
        if not np.array_equal(train_indices, split["train"]) or \
                fingerprint(mean, scale, pca_mean, components) != saved[
                    "projection_fingerprint"]:
            raise ValueError("projection train split or parameters differ")
        import torch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                               weights_only=True).numpy().astype(np.float64)
        content = published[:, 20:]
        refit_scaler = StandardScaler().fit(content[train_indices])
        normalized = refit_scaler.transform(content)
        refit_pca = PCA(n_components=len(added_names), svd_solver="randomized",
                        random_state=seed).fit(normalized[train_indices])
        if fingerprint(refit_scaler.mean_, refit_scaler.scale_, refit_pca.mean_,
                       refit_pca.components_) != saved["projection_fingerprint"]:
            raise ValueError("projection parameters do not refit from training source")
        projected = refit_pca.transform(normalized)
        x = np.column_stack((x, projected))
        names += added_names
    if saved["files_sha256"] != current["files_sha256"] or \
            saved["split_sha256"] != current["split_sha256"]:
        raise ValueError("source files or split differ from training")
    reference_path = directory / "reference.json.gz"
    if sha256_file(reference_path) != saved["reference_artifact_sha256"]:
        raise ValueError("reference artifact file checksum differs from report")
    reference = LabelBlindReference.load(reference_path)
    reasoner_payload = report["reasoner"]
    reasoner = (MonotoneSplineReasoner.from_manifest(reasoner_payload)
                if reasoner_payload.get("type") == "monotone_interval_spline"
                else PairwiseEvidenceReasoner.from_manifest(reasoner_payload)
                if reasoner_payload.get("type") == "pairwise_same_direction_minimum"
                else DualTailEvidenceReasoner.from_manifest(reasoner_payload)
                if reasoner_payload.get("type") == "dual_tail_sign_constrained_additive"
                else SignConstrainedReasoner.from_manifest(reasoner_payload))
    if reference.version != saved["reference_version"] or \
            reasoner.reference_version != reference.version or tuple(names) != reference.names:
        raise ValueError("reference, reasoner and vocabulary disagree")
    test = split["test"]
    test_evidence = evidence_with_neighbor_support(
        reference, x[test], None if neighbor_supports is None else neighbor_supports[test])
    reconstructed = reasoner.predict_proba(test_evidence)
    recorded = predictions["predictions"]
    if len(recorded) != len(test) or [row["account_id"] for row in recorded] != \
            [ids[int(i)] for i in test] or \
            any(row["label"] != int(y[i]) for row, i in zip(recorded, test)):
        raise ValueError("account order or labels differ from saved predictions")
    difference = float(np.max(np.abs(reconstructed - np.array(
        [row["probability"] for row in recorded]))))
    if difference > 1e-12:
        raise ValueError(f"saved probabilities do not reproduce: max diff {difference}")
    actual_groups = reasoner.contributions(test_evidence)
    if any(set(row.get("group_contributions", {})) != set(actual_groups)
           for row in recorded):
        raise ValueError("saved group contribution vocabulary differs")
    group_difference = float(max(
        np.max(np.abs(values - np.asarray([
            row["group_contributions"][name] for row in recorded], dtype=float)))
        for name, values in actual_groups.items()))
    if group_difference > 1e-12:
        raise ValueError(f"saved group contributions do not reproduce: {group_difference}")
    saved_logit = float(reasoner.intercept) + sum(
        np.asarray([row["group_contributions"][name] for row in recorded])
        for name in actual_groups)
    logit_difference = float(np.max(np.abs(saved_logit - reasoner.logits(test_evidence))))
    if logit_difference > 1e-12:
        raise ValueError(f"saved evidence terms do not reconstruct logits: {logit_difference}")
    masked = EvidenceBatch(np.zeros_like(test_evidence.upper),
                           np.zeros_like(test_evidence.lower),
                           np.zeros_like(test_evidence.reliability),
                           np.zeros_like(test_evidence.available, dtype=bool),
                           test_evidence.reference_version, test_evidence.names)
    masked_logit_difference = float(np.max(np.abs(
        reasoner.logits(masked) - float(reasoner.intercept))))
    if masked_logit_difference > 1e-12:
        raise ValueError("classifier retains a path when all evidence is masked")
    return {"dataset": args.dataset, "seed": seed, "n": len(test),
            "max_probability_difference": difference,
            "max_group_contribution_difference": group_difference,
            "max_logit_reconstruction_difference": logit_difference,
            "max_all_evidence_masked_logit_difference": masked_logit_difference,
            "reference_sha256": reference.version, "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fox8", "botsim", "mgtab"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-graph-facts", type=Path)
    parser.add_argument("--mgtab-structural-facts", type=Path)
    parser.add_argument("--mgtab-neighbor-facts", type=Path)
    parser.add_argument("--mgtab-content-projection", action="store_true")
    parser.add_argument("--fox8-native-facts", type=Path)
    parser.add_argument("--fox8-native-source-audit", type=Path)
    parser.add_argument("--fox8-raw", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    result = [verify(args, seed) for seed in args.seeds]
    (args.model_dir / "artifact_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
