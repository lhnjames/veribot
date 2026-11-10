"""Test whether MGTAB source responses depend on the replacement row."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
CENTROID_CERT = ROOT / "research/certificates/mgtab_content_source_centroid_response.json"
CENTROID_READBACK = ROOT / "research/certificates/mgtab_content_source_centroid_response_readback.json"
OUT_DIR = ROOT / "output/v3_mgtab_content_source_baseline_stability"
OUT_CERT = ROOT / "research/certificates/mgtab_content_source_baseline_stability.json"


def encode(runtime: MGTABFullAtomicRuntime, facts: np.ndarray,
           supports: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + 5.0)
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def main() -> None:
    centroid_cert = json.loads(CENTROID_CERT.read_text(encoding="utf-8"))
    readback = json.loads(CENTROID_READBACK.read_text(encoding="utf-8"))
    if (centroid_cert["status"] !=
            "published_content_row_replacement_and_input_mask_compared" or
            readback["status"] !=
            "all_published_content_source_responses_read_back" or
            readback["source_certificate_sha256"] !=
            sha256_file(CENTROID_CERT)):
        raise ValueError("original source replacement lacks independent readback")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for prior in centroid_cert["runs"]:
        seed = prior["seed"]
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        prior_path = Path(prior["result_path"])
        if sha256_file(prior_path) != prior["result_sha256"]:
            raise ValueError("prior centroid source response changed")
        with np.load(prior_path, allow_pickle=False) as previous:
            ids = previous["ids"].astype(str)
            full = previous["full_probability"].astype(float)
            centroid = previous["centroid_source_probability"].astype(float)
            selected = previous["selected_only_projection"].astype(bool)
            threshold = float(previous["threshold"])
            centroid_projection = previous[
                "train_centroid_projection"].astype(float)
        if (len(ids) != 1020 or threshold != runtime.threshold or
                prior["model_sha256"] != sha256_file(
                    runtime.run_dir / "model.joblib") or
                prior["reference_sha256"] != sha256_file(
                    runtime.run_dir / "label_blind_reference.json.gz")):
            raise ValueError("frozen model or test population differs")
        with np.load(runtime.projection_path, allow_pickle=False) as z:
            train = z["train_indices"].astype(int)
            scaler_mean = z["scaler_mean"].astype(float)
            scaler_scale = z["scaler_scale"].astype(float)
            pca_mean = z["pca_mean"].astype(float)
            components = z["pca_components"].astype(float)
        if (len(train) != 7139 or len(np.unique(train)) != len(train) or
                np.any(train < 0) or np.any(train >= 10199) or
                sha256_file(runtime.projection_path) != prior[
                    "projection_artifact_sha256"]):
            raise ValueError("train-only projection indices differ")
        published = torch.load(runtime.data_dir / "features.pt",
                               map_location="cpu", weights_only=True).numpy()
        if (published.shape != (10199, 788) or
                sha256_file(runtime.data_dir / "features.pt") != prior[
                    "published_features_sha256"]):
            raise ValueError("published MGTAB tensor differs")
        train_projected = ((published[train, 20:].astype(float) -
                            scaler_mean) / scaler_scale - pca_mean)
        train_projected = train_projected @ components.T
        center = (-pca_mean) @ components.T
        if np.max(np.abs(center-centroid_projection)) > 1e-12:
            raise ValueError("train centroid projection differs")
        distances = np.linalg.norm(train_projected-center, axis=1)
        nearest_position = int(np.lexsort((train, distances))[0])
        exemplar_row = int(train[nearest_position])
        exemplar_content = published[exemplar_row, 20:].astype(float)
        test_indices = np.asarray([runtime.index[uid] for uid in ids],
                                  dtype=int)
        if np.any(np.isin(test_indices, train)):
            raise ValueError("fixed test row appears in train projection fit")
        original = runtime.facts[test_indices]
        original_encoded = encode(runtime, original,
                                  runtime.supports[test_indices])
        replay = runtime.model.predict_proba(original_encoded)[:, 1]
        full_error = float(np.max(np.abs(replay-full)))
        if full_error > 1e-10:
            raise ValueError("original frozen prediction differs")
        edited_content = published[test_indices, 20:].astype(float).copy()
        edited_content[:] = exemplar_content
        exemplar_projection = ((edited_content - scaler_mean) /
                               scaler_scale - pca_mean) @ components.T
        projected_error = float(np.max(np.abs(
            exemplar_projection-train_projected[nearest_position])))
        if projected_error > 1e-10:
            raise ValueError("published train exemplar projection differs")
        modified = original.copy()
        modified[:, 86:150] = exemplar_projection
        changed_encoded = encode(runtime, modified,
                                 runtime.supports[test_indices])
        if (not np.array_equal(changed_encoded[:, :86],
                               original_encoded[:, :86]) or
                not np.array_equal(changed_encoded[:, 150:236],
                                   original_encoded[:, 150:236])):
            raise ValueError("content replacement altered other evidence")
        exemplar = runtime.model.predict_proba(changed_encoded)[:, 1]
        old_class = full >= threshold
        centroid_flip = (centroid >= threshold) != old_class
        exemplar_flip = (exemplar >= threshold) != old_class
        disagreement = centroid_flip != exemplar_flip
        endpoint_margin = np.minimum.reduce((
            np.abs(full-threshold), np.abs(centroid-threshold),
            np.abs(exemplar-threshold)))
        out_path = OUT_DIR / f"seed{seed}.npz"
        np.savez_compressed(out_path, ids=ids, full_probability=full,
                            centroid_probability=centroid,
                            train_exemplar_probability=exemplar,
                            threshold=np.asarray(threshold),
                            nearest_train_row=np.asarray(exemplar_row),
                            selected_only_projection=selected)
        row = {
            "dataset": "mgtab", "seed": seed, "accounts": len(ids),
            "nearest_train_source_row": exemplar_row,
            "nearest_train_source_distance_in_64d_projection": float(
                distances[nearest_position]),
            "centroid_class_flips": int(centroid_flip.sum()),
            "nearest_train_exemplar_class_flips": int(exemplar_flip.sum()),
            "class_flip_disagreement": int(disagreement.sum()),
            "disagreements_with_all_endpoint_margins_at_least_0p001": int(
                np.sum(disagreement & (endpoint_margin >= 0.001))),
            "disagreements_with_all_endpoint_margins_at_least_0p01": int(
                np.sum(disagreement & (endpoint_margin >= 0.01))),
            "both_class_flip": int(np.sum(centroid_flip & exemplar_flip)),
            "centroid_only_class_flip": int(np.sum(centroid_flip & ~exemplar_flip)),
            "exemplar_only_class_flip": int(np.sum(exemplar_flip & ~centroid_flip)),
            "only_projection_displayed_accounts": int(selected.sum()),
            "only_projection_class_flip_disagreement": int(np.sum(
                selected & disagreement)),
            "minimum_centroid_flip_endpoint_threshold_margin": float(
                np.min(np.minimum(np.abs(full[centroid_flip]-threshold),
                                  np.abs(centroid[centroid_flip]-threshold)))),
            "minimum_exemplar_flip_endpoint_threshold_margin": float(
                np.min(np.minimum(np.abs(full[exemplar_flip]-threshold),
                                  np.abs(exemplar[exemplar_flip]-threshold)))),
            "max_original_probability_replay_difference": full_error,
            "max_exemplar_projection_replay_difference": projected_error,
            "prior_result_sha256": prior["result_sha256"],
            "published_features_sha256": prior["published_features_sha256"],
            "projection_artifact_sha256": prior[
                "projection_artifact_sha256"],
            "model_sha256": prior["model_sha256"],
            "reference_sha256": prior["reference_sha256"],
            "result_path": str(out_path),
            "result_sha256": sha256_file(out_path),
        }
        runs.append(row)
        print(json.dumps({key: row[key] for key in (
            "seed", "nearest_train_source_row", "centroid_class_flips",
            "nearest_train_exemplar_class_flips", "class_flip_disagreement")}),
            flush=True)
    OUT_CERT.write_text(json.dumps({
        "status": "two_label_blind_published_content_replacements_compared",
        "scope": "three_mgtab_fixed_test_runs_all_3060_decisions",
        "operation_1": "replace published average-tweet tensor row with train-only content centroid",
        "operation_2": "replace the same tensor row with a published training row nearest the centroid in the frozen 64d PCA space",
        "neither_operation_is_a_known_error_correction": True,
        "fixed_test_labels_read": False,
        "prior_centroid_certificate_sha256": sha256_file(CENTROID_CERT),
        "prior_centroid_readback_sha256": sha256_file(CENTROID_READBACK),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
