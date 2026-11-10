"""Compare a frozen MGTAB decision with a label-blind local content donor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "research/certificates/mgtab_content_source_baseline_stability.json"
BASELINE_READBACK = ROOT / "research/certificates/mgtab_content_source_baseline_stability_readback.json"
OUT_DIR = ROOT / "output/v3_mgtab_content_local_source_replacement"
OUT_CERT = ROOT / "research/certificates/mgtab_content_local_source_replacement.json"


def encoded(runtime: MGTABFullAtomicRuntime, facts: np.ndarray,
            supports: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + 5.0)
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    readback = json.loads(BASELINE_READBACK.read_text(encoding="utf-8"))
    if (baseline["status"] !=
            "two_label_blind_published_content_replacements_compared" or
            readback["status"] !=
            "all_published_content_replacement_stability_rows_read_back" or
            readback["source_certificate_sha256"] != sha256_file(BASELINE)):
        raise ValueError("prior source replacement comparison is unbound")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for previous in baseline["runs"]:
        seed = int(previous["seed"])
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        old_path = Path(previous["result_path"])
        if sha256_file(old_path) != previous["result_sha256"]:
            raise ValueError("prior source result changed")
        with np.load(old_path, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            centroid = z["centroid_probability"].astype(float)
            global_donor = z["train_exemplar_probability"].astype(float)
            selected = z["selected_only_projection"].astype(bool)
            threshold = float(z["threshold"])
        if (len(ids) != 1020 or threshold != runtime.threshold or
                sha256_file(runtime.run_dir / "model.joblib") != previous[
                    "model_sha256"] or
                sha256_file(runtime.run_dir / "label_blind_reference.json.gz")
                != previous["reference_sha256"]):
            raise ValueError("frozen model or test rows differ")
        with np.load(runtime.projection_path, allow_pickle=False) as z:
            train = z["train_indices"].astype(int)
            scaler_mean = z["scaler_mean"].astype(float)
            scaler_scale = z["scaler_scale"].astype(float)
            pca_mean = z["pca_mean"].astype(float)
            components = z["pca_components"].astype(float)
        published_path = runtime.data_dir / "features.pt"
        if (sha256_file(published_path) != previous[
                "published_features_sha256"] or
                sha256_file(runtime.projection_path) != previous[
                    "projection_artifact_sha256"] or
                len(train) != 7139 or len(np.unique(train)) != 7139):
            raise ValueError("published source or train projection differs")
        published = torch.load(published_path, map_location="cpu",
                               weights_only=True).numpy()
        if published.shape != (10199, 788):
            raise ValueError("published feature shape differs")
        test_rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        if np.any(np.isin(test_rows, train)):
            raise ValueError("test account appeared in training donor set")
        original_facts = runtime.facts[test_rows]
        original_encoded = encoded(runtime, original_facts,
                                   runtime.supports[test_rows])
        original_prob = runtime.model.predict_proba(original_encoded)[:, 1]
        original_error = float(np.max(np.abs(original_prob-full)))
        if original_error > 1e-10:
            raise ValueError("original frozen prediction differs")
        order = np.argsort(train)
        sorted_train = train[order]
        train_projected = ((published[sorted_train, 20:].astype(float) -
                            scaler_mean) / scaler_scale - pca_mean) @ components.T
        test_projected = ((published[test_rows, 20:].astype(float) -
                           scaler_mean) / scaler_scale - pca_mean) @ components.T
        centroid_projected = (-pca_mean) @ components.T
        centroid_distances = np.linalg.norm(
            test_projected-centroid_projected, axis=1)
        projection_error = float(np.max(np.abs(
            test_projected-original_facts[:, 86:150])))
        if projection_error > 1e-10:
            raise ValueError("published test projection differs")
        # Ascending row order makes the first equal-distance minimum deterministic.
        nearest_positions = np.empty(len(ids), dtype=int)
        nearest_distances = np.empty(len(ids), dtype=float)
        for start in range(0, len(ids), 64):
            stop = min(start + 64, len(ids))
            square = cdist(test_projected[start:stop], train_projected,
                           metric="sqeuclidean")
            positions = np.argmin(square, axis=1)
            nearest_positions[start:stop] = positions
            nearest_distances[start:stop] = np.sqrt(square[
                np.arange(stop-start), positions])
        local_rows = sorted_train[nearest_positions]
        if np.any(np.isin(local_rows, test_rows)):
            raise ValueError("local donor equals its test account")
        donor_content = published[local_rows, 20:].astype(float)
        donor_projected = ((donor_content-scaler_mean) /
                           scaler_scale-pca_mean) @ components.T
        donor_error = float(np.max(np.abs(
            donor_projected-train_projected[nearest_positions])))
        if donor_error > 1e-10:
            raise ValueError("local published donor projection differs")
        modified = original_facts.copy()
        modified[:, 86:150] = donor_projected
        changed_encoded = encoded(runtime, modified,
                                   runtime.supports[test_rows])
        if (not np.array_equal(changed_encoded[:, :86],
                               original_encoded[:, :86]) or
                not np.array_equal(changed_encoded[:, 150:236],
                                   original_encoded[:, 150:236])):
            raise ValueError("local content donor changed other input groups")
        local = runtime.model.predict_proba(changed_encoded)[:, 1]
        original_class = full >= threshold
        center_flip = (centroid >= threshold) != original_class
        global_flip = (global_donor >= threshold) != original_class
        local_flip = (local >= threshold) != original_class
        center_disagree = center_flip != local_flip
        global_disagree = global_flip != local_flip
        endpoint_margin = np.minimum.reduce((
            np.abs(full-threshold), np.abs(centroid-threshold),
            np.abs(local-threshold)))
        out_path = OUT_DIR / f"seed{seed}.npz"
        np.savez_compressed(out_path, ids=ids, full_probability=full,
                            centroid_probability=centroid,
                            global_donor_probability=global_donor,
                            local_donor_probability=local,
                            local_donor_row=local_rows,
                            local_donor_distance_in_64d_projection=
                                nearest_distances,
                            centroid_distance_in_64d_projection=
                                centroid_distances,
                            selected_only_projection=selected,
                            threshold=np.asarray(threshold))
        row = {
            "dataset": "mgtab", "seed": seed, "accounts": len(ids),
            "centroid_class_flips": int(center_flip.sum()),
            "global_donor_class_flips": int(global_flip.sum()),
            "local_donor_class_flips": int(local_flip.sum()),
            "centroid_local_class_flip_disagreement": int(
                center_disagree.sum()),
            "global_local_class_flip_disagreement": int(
                global_disagree.sum()),
            "centroid_only_vs_local": int(np.sum(center_flip & ~local_flip)),
            "local_only_vs_centroid": int(np.sum(local_flip & ~center_flip)),
            "both_centroid_and_local_flip": int(np.sum(
                center_flip & local_flip)),
            "centroid_local_disagreements_all_endpoint_margins_at_least_0p01":
                int(np.sum(center_disagree & (endpoint_margin >= 0.01))),
            "only_projection_displayed_accounts": int(selected.sum()),
            "only_projection_local_class_flips": int(np.sum(
                selected & local_flip)),
            "local_donor_distance_median_64d": float(np.median(
                nearest_distances)),
            "centroid_distance_median_64d": float(np.median(
                centroid_distances)),
            "local_donor_nearer_than_centroid_accounts": int(np.sum(
                nearest_distances < centroid_distances)),
            "local_donor_distance_max_64d": float(np.max(
                nearest_distances)),
            "distinct_local_donor_rows": int(len(np.unique(local_rows))),
            "max_original_probability_replay_error": original_error,
            "max_test_projection_replay_error": projection_error,
            "max_local_donor_projection_replay_error": donor_error,
            "prior_result_sha256": previous["result_sha256"],
            "published_features_sha256": previous["published_features_sha256"],
            "projection_artifact_sha256": previous[
                "projection_artifact_sha256"],
            "model_sha256": previous["model_sha256"],
            "reference_sha256": previous["reference_sha256"],
            "result_path": str(out_path),
            "result_sha256": sha256_file(out_path),
        }
        runs.append(row)
        print(json.dumps({key: row[key] for key in (
            "seed", "local_donor_class_flips",
            "centroid_local_class_flip_disagreement",
            "local_donor_distance_median_64d")}), flush=True)
    OUT_CERT.write_text(json.dumps({
        "status": "local_label_blind_published_content_donor_response",
        "scope": "all_3060_mgtab_fixed_test_decisions",
        "operation": "replace each test account published average-tweet tensor row with its nearest published training row in the frozen train-only 64d PCA space",
        "known_error_correction": False,
        "fixed_test_labels_read": False,
        "prior_baseline_certificate_sha256": sha256_file(BASELINE),
        "prior_baseline_readback_sha256": sha256_file(BASELINE_READBACK),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
