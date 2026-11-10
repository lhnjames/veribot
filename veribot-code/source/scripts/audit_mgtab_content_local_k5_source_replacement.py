"""Measure frozen MGTAB class response to five label-blind local content donors."""

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
PRIOR_CERT = ROOT / "research/certificates/mgtab_content_local_source_replacement.json"
PRIOR_READBACK = ROOT / "research/certificates/mgtab_content_local_source_replacement_readback.json"
OUT_DIR = ROOT / "output/v3_mgtab_content_local_k5_source_replacement"
OUT_CERT = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement.json"
K = 5


def score(runtime: MGTABFullAtomicRuntime, facts: np.ndarray,
          supports: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + 5.0)
    x = np.column_stack((np.log1p(batch.upper) * reliability,
                         np.log1p(batch.lower) * reliability))
    return runtime.model.predict_proba(x)[:, 1]


def main() -> None:
    prior = json.loads(PRIOR_CERT.read_text(encoding="utf-8"))
    readback = json.loads(PRIOR_READBACK.read_text(encoding="utf-8"))
    if (prior["status"] !=
            "local_label_blind_published_content_donor_response" or
            readback["status"] !=
            "local_published_content_donor_rows_and_probabilities_read_back" or
            readback["source_certificate_sha256"] != sha256_file(PRIOR_CERT)):
        raise ValueError("single nearest-donor run lacks independent readback")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for old in prior["runs"]:
        seed = int(old["seed"])
        old_path = Path(old["result_path"])
        if sha256_file(old_path) != old["result_sha256"]:
            raise ValueError("single nearest-donor result changed")
        with np.load(old_path, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            centroid = z["centroid_probability"].astype(float)
            first_row = z["local_donor_row"].astype(int)
            first_distance = z[
                "local_donor_distance_in_64d_projection"].astype(float)
            first_probability = z["local_donor_probability"].astype(float)
            selected = z["selected_only_projection"].astype(bool)
            threshold = float(z["threshold"])
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        source_path = runtime.data_dir / "features.pt"
        if (len(ids) != 1020 or threshold != runtime.threshold or
                sha256_file(source_path) != old["published_features_sha256"] or
                sha256_file(runtime.projection_path) != old[
                    "projection_artifact_sha256"] or
                sha256_file(runtime.run_dir / "model.joblib") != old[
                    "model_sha256"] or
                sha256_file(runtime.run_dir / "label_blind_reference.json.gz")
                != old["reference_sha256"]):
            raise ValueError("frozen source, projection, or model changed")
        published = torch.load(source_path, map_location="cpu",
                               weights_only=True).numpy()
        with np.load(runtime.projection_path, allow_pickle=False) as z:
            train = np.sort(z["train_indices"].astype(int))
            mean = z["scaler_mean"].astype(float)
            scale = z["scaler_scale"].astype(float)
            pca_mean = z["pca_mean"].astype(float)
            components = z["pca_components"].astype(float)
        if (published.shape != (10199, 788) or len(train) != 7139 or
                len(np.unique(train)) != len(train)):
            raise ValueError("published donor population changed")
        test_rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        if np.any(np.isin(test_rows, train)):
            raise ValueError("test account appears in donor set")
        train_projection = ((published[train, 20:].astype(float)-mean) /
                            scale-pca_mean) @ components.T
        test_projection = ((published[test_rows, 20:].astype(float)-mean) /
                           scale-pca_mean) @ components.T
        if np.max(np.abs(test_projection-runtime.facts[
                test_rows, 86:150])) > 1e-10:
            raise ValueError("published content projection differs")
        donor_rows = np.empty((len(ids), K), dtype=int)
        donor_distances = np.empty((len(ids), K), dtype=float)
        for start in range(0, len(ids), 32):
            stop = min(start+32, len(ids))
            square = cdist(test_projection[start:stop], train_projection,
                           metric="sqeuclidean")
            # Stable sort respects ascending source row for exact distance ties.
            positions = np.argsort(square, axis=1, kind="stable")[:, :K]
            donor_rows[start:stop] = train[positions]
            donor_distances[start:stop] = np.sqrt(np.take_along_axis(
                square, positions, axis=1))
        if (not np.array_equal(donor_rows[:, 0], first_row) or
                np.max(np.abs(donor_distances[:, 0]-first_distance))
                > 1e-10):
            raise ValueError("first nearest donor differs from prior run")
        probabilities = np.empty((len(ids), K), dtype=float)
        original_facts = runtime.facts[test_rows]
        for rank in range(K):
            donor_content = published[donor_rows[:, rank], 20:].astype(float)
            projected = ((donor_content-mean)/scale-pca_mean) @ components.T
            modified = original_facts.copy()
            modified[:, 86:150] = projected
            probabilities[:, rank] = score(runtime, modified,
                                           runtime.supports[test_rows])
        first_error = float(np.max(np.abs(
            probabilities[:, 0]-first_probability)))
        if first_error > 1e-10:
            raise ValueError("first local donor probability differs")
        original_class = full >= threshold
        flips = (probabilities >= threshold) != original_class[:, None]
        flip_count = flips.sum(axis=1)
        nearest_distance_gap = donor_distances[:, -1]-donor_distances[:, 0]
        out_path = OUT_DIR / f"seed{seed}.npz"
        np.savez_compressed(out_path, ids=ids, full_probability=full,
                            centroid_probability=centroid,
                            donor_rows=donor_rows,
                            donor_distances_in_64d_projection=donor_distances,
                            donor_probabilities=probabilities,
                            selected_only_projection=selected,
                            threshold=np.asarray(threshold))
        row = {
            "dataset": "mgtab", "seed": seed, "accounts": len(ids),
            "k": K,
            "at_least_one_of_k_class_flips": int(np.sum(flip_count > 0)),
            "all_k_class_flip": int(np.sum(flip_count == K)),
            "mixed_class_flip_across_k": int(np.sum(
                (flip_count > 0) & (flip_count < K))),
            "no_k_class_flip": int(np.sum(flip_count == 0)),
            "first_donor_class_flips": int(np.sum(flips[:, 0])),
            "first_no_flip_but_another_flips": int(np.sum(
                ~flips[:, 0] & (flip_count > 0))),
            "first_flips_but_another_does_not": int(np.sum(
                flips[:, 0] & (flip_count < K))),
            "only_projection_displayed_accounts": int(selected.sum()),
            "only_projection_any_k_class_flips": int(np.sum(
                selected & (flip_count > 0))),
            "median_fifth_minus_first_distance_64d": float(np.median(
                nearest_distance_gap)),
            "max_first_probability_replay_error": first_error,
            "prior_result_sha256": old["result_sha256"],
            "published_features_sha256": old["published_features_sha256"],
            "projection_artifact_sha256": old[
                "projection_artifact_sha256"],
            "model_sha256": old["model_sha256"],
            "reference_sha256": old["reference_sha256"],
            "result_path": str(out_path),
            "result_sha256": sha256_file(out_path),
        }
        runs.append(row)
        print(json.dumps({key: row[key] for key in (
            "seed", "at_least_one_of_k_class_flips",
            "mixed_class_flip_across_k", "all_k_class_flip")}),
            flush=True)
    OUT_CERT.write_text(json.dumps({
        "status": "five_nearest_published_content_donor_responses",
        "scope": "all_3060_mgtab_fixed_test_decisions",
        "neighbor_selection": "five nearest published training rows in frozen train-only 64d PCA, ascending source-row tie break, no class labels",
        "known_error_correction": False,
        "fixed_test_labels_read": False,
        "prior_local_certificate_sha256": sha256_file(PRIOR_CERT),
        "prior_local_readback_sha256": sha256_file(PRIOR_READBACK),
        "runs": runs,
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
