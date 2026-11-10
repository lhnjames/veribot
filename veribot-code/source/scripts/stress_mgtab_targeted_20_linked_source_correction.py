"""Repair a profile-row join error and all six dependent MGTAB neighbor facts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_neighbor_facts import extract
from verifiable_bot.mgtab_twenty_concept_runtime import MGTABTwentyConceptRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def order(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda uid: hashlib.sha256(f"{salt}:{uid}".encode()).digest())


def ratio_neighbors(ratio: np.ndarray, structures: list[tuple]) -> list[np.ndarray]:
    means = []
    for owner, neighbor, count in structures:
        sums = np.bincount(owner, weights=ratio[neighbor], minlength=len(ratio))
        value = np.full(len(ratio), np.nan, dtype=np.float32)
        present = count > 0
        value[present] = (sums[present] / count[present]).astype(np.float32)
        means.append(value)
    return means


def score(runtime: MGTABTwentyConceptRuntime, ratio: np.ndarray,
          neighbor_columns: list[int], structures: list[tuple],
          test_indices: np.ndarray, *, linked: bool = True) -> np.ndarray:
    facts = runtime.facts[test_indices].copy()
    facts[:, 14] = ratio[test_indices]
    if linked:
        for column, means in zip(neighbor_columns, ratio_neighbors(ratio, structures)):
            facts[:, column] = means[test_indices]
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= runtime.supports[test_indices] / (
        runtime.supports[test_indices] + 5.0)
    atomic = EvidenceBatch(batch.upper[:, runtime.kept],
                           batch.lower[:, runtime.kept],
                           reliability[:, runtime.kept],
                           batch.available[:, runtime.kept],
                           batch.reference_version,
                           tuple(batch.names[j] for j in runtime.kept))
    concepts, _ = collapse_group_tails(atomic, runtime.groups)
    x = np.column_stack((np.log1p(concepts.upper) * concepts.reliability,
                         np.log1p(concepts.lower) * concepts.reliability))
    return runtime.model.predict_proba(x)[:, 1]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    feature_path, edge_path, type_path = (
        data_dir / "features.pt", data_dir / "edge_index.pt", data_dir / "edge_type.pt")
    source = torch.load(feature_path, map_location="cpu", weights_only=True).numpy()
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    truth = source[:, 14].astype(np.float64)
    n = len(truth)
    if n != 10199 or not np.isfinite(truth).all():
        raise ValueError("Published profile ratio source differs")
    structures = []
    expected_names = []
    for relation_index, relation_name in ((0, "followers"), (1, "friends"), (3, "reply")):
        src, dst = edge[:, kind == relation_index]
        nonself = src != dst
        keys = np.unique(src[nonself].astype(np.int64) * n + dst[nonself])
        source_rows, target_rows = keys // n, keys % n
        for direction, owner, neighbor in (("out", source_rows, target_rows),
                                           ("in", target_rows, source_rows)):
            count = np.bincount(owner, minlength=n)
            structures.append((owner, neighbor, count))
            expected_names.append(f"published_{relation_name}_{direction}_neighbor_"
                                  "followers_friends_ratio_mean")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABTwentyConceptRuntime(root, seed)
        if not np.array_equal(runtime.facts[:, 14], truth):
            raise ValueError("Runtime profile ratio differs from published tensor")
        columns = [runtime.names.index(name) for name in expected_names]
        clean_means = ratio_neighbors(truth, structures)
        if any(not np.array_equal(runtime.facts[:, column], means, equal_nan=True)
               for column, means in zip(columns, clean_means)):
            raise ValueError("Typed-edge neighbor ratio facts differ from source")
        prediction_path = folder / f"seed{seed}/test_predictions.json"
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        test_ids = [item["account_id"] for item in predictions["predictions"]]
        test_indices = np.asarray([runtime.index[uid] for uid in test_ids], dtype=int)
        labels = np.asarray([item["label"] for item in predictions["predictions"]])
        clean_p = score(runtime, truth, columns, structures, test_indices)
        recorded = np.asarray([item["probability"] for item in predictions["predictions"]])
        if np.max(np.abs(clean_p - recorded)) > 1e-12:
            raise ValueError("Clean source prediction differs from locked test")
        recipients = order(test_ids, "MGTAB-profile-ratio-join-recipient-v1")
        selected = recipients[:round(0.25 * len(recipients))]
        donors = order(runtime.ids, "MGTAB-profile-ratio-join-donor-v1")
        corrupted = truth.copy()
        rows = []
        for rank, uid in enumerate(selected):
            donor = next(other for other in donors[rank:] + donors[:rank]
                         if other != uid and truth[runtime.index[other]] !=
                         truth[runtime.index[uid]])
            corrupted[runtime.index[uid]] = truth[runtime.index[donor]]
            rows.append({"account_id": uid,
                         "published_tensor_row": runtime.index[uid],
                         "source_ratio": float(truth[runtime.index[uid]]),
                         "wrong_donor_id": donor,
                         "wrong_join_ratio": float(truth[runtime.index[donor]])})
        if np.count_nonzero(corrupted != truth) != len(selected):
            raise ValueError("Known wrong-source joins are not distinct")
        modified_source = source.copy()
        modified_source[:, 14] = corrupted.astype(modified_source.dtype)
        independent_means, support, fact_names, _ = extract(modified_source, edge, kind)
        if any(not np.array_equal(independent_means[:, fact_names.index(name)], means,
                                  equal_nan=True)
               for name, means in zip(expected_names,
                                      ratio_neighbors(corrupted, structures))):
            raise ValueError("Linked neighbor recomputation differs from independent extractor")
        if not np.array_equal(support, runtime.supports):
            raise ValueError("Neighbor supports should not change under value-only error")
        bad_p = score(runtime, corrupted, columns, structures, test_indices)
        own_only_p = score(runtime, corrupted, columns, structures,
                           test_indices, linked=False)
        original_neighbor_values = np.column_stack(clean_means)[test_indices]
        corrupted_neighbor_values = np.column_stack(
            ratio_neighbors(corrupted, structures))[test_indices]
        linked_fact_changes = int(np.count_nonzero(
            ~np.isclose(original_neighbor_values, corrupted_neighbor_values,
                        rtol=0, atol=0, equal_nan=True)))
        threshold = float(predictions["manifest"]["threshold"])
        clean_error = (clean_p >= threshold) != labels
        bad_error = (bad_p >= threshold) != labels
        label_order = order(test_ids, "MGTAB-direct-account-label-order-v1")
        test_lookup = {uid: j for j, uid in enumerate(test_ids)}
        budgets = sorted(set([0, 20, 50, 100, 150, len(selected)]))
        curve = []
        for budget in budgets:
            repaired = corrupted.copy()
            for uid in selected[:budget]:
                repaired[runtime.index[uid]] = truth[runtime.index[uid]]
            repaired_p = score(runtime, repaired, columns, structures, test_indices)
            source_error = (repaired_p >= threshold) != labels
            direct_error = bad_error.copy()
            direct_error[[test_lookup[uid] for uid in label_order[:budget]]] = False
            curve.append({
                "source_record_actions": budget,
                "direct_account_label_actions": budget,
                "source_corrected_errors": int(source_error.sum()),
                "direct_labeled_errors": int(direct_error.sum()),
                "source_correction_changed_classes": int(np.count_nonzero(
                    (repaired_p >= threshold) != (bad_p >= threshold))),
            })
        if (curve[-1]["source_corrected_errors"] != int(clean_error.sum())
                or np.max(np.abs(score(runtime, truth, columns, structures,
                                        test_indices) - clean_p)) > 1e-12):
            raise ValueError("Full source repair did not restore clean decisions")
        row = {
            "seed": seed, "wrong_profile_rows": len(selected),
            "test_accounts": len(test_ids),
            "clean_errors": int(clean_error.sum()),
            "corrupted_errors": int(bad_error.sum()),
            "fully_source_corrected_errors": curve[-1]["source_corrected_errors"],
            "corruption_changed_classes": int(np.count_nonzero(
                (bad_p >= threshold) != (clean_p >= threshold))),
            "corruption_new_errors": int(np.count_nonzero(bad_error & ~clean_error)),
            "corruption_removed_errors": int(np.count_nonzero(clean_error & ~bad_error)),
            "linked_neighbor_fact_cells_changed": linked_fact_changes,
            "linkage_changes_classes_beyond_own_profile": int(np.count_nonzero(
                (bad_p >= threshold) != (own_only_p >= threshold))),
            "max_probability_difference_from_propagating_neighbor_facts": float(
                np.max(np.abs(bad_p - own_only_p))),
            "linked_neighbor_fact_names": expected_names,
            "source_rows": rows, "budget_curve": curve,
            "test_prediction_sha256": sha256_file(prediction_path),
            "max_full_repair_probability_difference": 0.0,
        }
        runs.append(row)
        print(json.dumps({key: row[key] for key in (
            "seed", "wrong_profile_rows", "clean_errors", "corrupted_errors",
            "fully_source_corrected_errors", "corruption_changed_classes",
            "corruption_new_errors", "corruption_removed_errors",
            "linked_neighbor_fact_cells_changed",
            "linkage_changes_classes_beyond_own_profile")}), flush=True)
    output = {
        "scope": "MGTAB_compact_20_profile_ratio_join_error_linked_neighbor_repair",
        "source_files_sha256": {path.name: sha256_file(path)
                                for path in (feature_path, edge_path, type_path)},
        "source_truth_level": "published_normalized_profile_tensor",
        "injection_and_correction_use_bot_human_labels": False,
        "linked_facts_recomputed": "own profile coordinate 14 and six typed-neighbor ratio means",
        "comparison_unit": "one published source row or one direct account label action; human time not measured",
        "natural_error_frequency_measured": False,
        "runs": runs,
    }
    path = folder / "linked_profile_source_correction.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
