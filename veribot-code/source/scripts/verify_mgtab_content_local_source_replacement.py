"""Independently read back MGTAB local-donor source responses."""

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
CERT = ROOT / "research/certificates/mgtab_content_local_source_replacement.json"
PRIOR = ROOT / "research/certificates/mgtab_content_source_baseline_stability.json"
PRIOR_READBACK = ROOT / "research/certificates/mgtab_content_source_baseline_stability_readback.json"
OUT = ROOT / "research/certificates/mgtab_content_local_source_replacement_readback.json"


def main() -> None:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if (cert["status"] != "local_label_blind_published_content_donor_response" or
            cert["prior_baseline_certificate_sha256"] != sha256_file(PRIOR) or
            cert["prior_baseline_readback_sha256"] !=
            sha256_file(PRIOR_READBACK) or
            cert["known_error_correction"] is not False or
            cert["fixed_test_labels_read"] is not False):
        raise ValueError("local donor result lacks frozen prior binding")
    previous = {run["seed"]: run for run in prior["runs"]}
    runs = []
    for row in cert["runs"]:
        seed = int(row["seed"])
        old = previous[seed]
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        path = Path(row["result_path"])
        old_path = Path(old["result_path"])
        source_path = runtime.data_dir / "features.pt"
        if (row["dataset"] != "mgtab" or row["accounts"] != 1020 or
                sha256_file(path) != row["result_sha256"] or
                sha256_file(old_path) != row["prior_result_sha256"] or
                sha256_file(source_path) != row[
                    "published_features_sha256"] or
                sha256_file(runtime.projection_path) != row[
                    "projection_artifact_sha256"] or
                sha256_file(runtime.run_dir / "model.joblib") != row[
                    "model_sha256"] or
                sha256_file(runtime.run_dir / "label_blind_reference.json.gz")
                != row["reference_sha256"]):
            raise ValueError("local donor source/model hash differs")
        with np.load(path, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            centroid = z["centroid_probability"].astype(float)
            global_donor = z["global_donor_probability"].astype(float)
            local = z["local_donor_probability"].astype(float)
            local_rows = z["local_donor_row"].astype(int)
            local_distances = z[
                "local_donor_distance_in_64d_projection"].astype(float)
            center_distances = z[
                "centroid_distance_in_64d_projection"].astype(float)
            selected = z["selected_only_projection"].astype(bool)
            threshold = float(z["threshold"])
        with np.load(old_path, allow_pickle=False) as z:
            if (not np.array_equal(ids, z["ids"].astype(str)) or
                    not np.array_equal(selected, z[
                        "selected_only_projection"].astype(bool)) or
                    np.max(np.abs(full-z["full_probability"])) > 1e-12 or
                    np.max(np.abs(centroid-z["centroid_probability"]))
                    > 1e-12 or
                    np.max(np.abs(global_donor-z[
                        "train_exemplar_probability"])) > 1e-12 or
                    threshold != float(z["threshold"])):
                raise ValueError("local donor test rows differ from prior")
        if threshold != runtime.threshold:
            raise ValueError("frozen threshold differs")
        with np.load(runtime.projection_path, allow_pickle=False) as z:
            train_rows = z["train_indices"].astype(int)
            mean = z["scaler_mean"].astype(float)
            scale = z["scaler_scale"].astype(float)
            pca_mean = z["pca_mean"].astype(float)
            components = z["pca_components"].astype(float)
        published = torch.load(source_path, map_location="cpu",
                               weights_only=True).numpy()
        if (published.shape != (10199, 788) or len(train_rows) != 7139 or
                len(np.unique(train_rows)) != 7139):
            raise ValueError("published source or donor population differs")
        test_rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        if np.any(np.isin(test_rows, train_rows)):
            raise ValueError("train/test account overlap")
        ordered_train = np.sort(train_rows)
        train_projected = ((published[ordered_train, 20:].astype(float)-mean) /
                           scale-pca_mean) @ components.T
        test_projected = ((published[test_rows, 20:].astype(float)-mean) /
                          scale-pca_mean) @ components.T
        expected_center_distances = np.linalg.norm(
            test_projected-((-pca_mean) @ components.T), axis=1)
        positions = np.empty(len(ids), dtype=int)
        distances = np.empty(len(ids), dtype=float)
        for start in range(0, len(ids), 64):
            stop = min(start+64, len(ids))
            square = cdist(test_projected[start:stop], train_projected,
                           metric="sqeuclidean")
            nearest = np.argmin(square, axis=1)
            positions[start:stop] = nearest
            distances[start:stop] = np.sqrt(square[
                np.arange(stop-start), nearest])
        if (not np.array_equal(local_rows, ordered_train[positions]) or
                np.max(np.abs(local_distances-distances)) > 1e-10 or
                np.max(np.abs(center_distances-expected_center_distances))
                > 1e-10):
            raise ValueError("nearest train donor or distance differs")
        facts = runtime.facts[test_rows].copy()
        projected_donors = ((published[local_rows, 20:].astype(float)-mean) /
                            scale-pca_mean) @ components.T
        facts[:, 86:150] = projected_donors
        reference = runtime.reference.transform(facts)
        reliability = reference.reliability.copy()
        reliability[:, 62:86] *= runtime.supports[test_rows] / (
            runtime.supports[test_rows]+5.0)
        encoded = np.column_stack((np.log1p(reference.upper)*reliability,
                                   np.log1p(reference.lower)*reliability))
        recomputed = runtime.model.predict_proba(encoded)[:, 1]
        replay_error = float(np.max(np.abs(recomputed-local)))
        if replay_error > 1e-10:
            raise ValueError("saved local donor probability is not reproduced")
        original_class = full >= threshold
        center_flip = (centroid >= threshold) != original_class
        global_flip = (global_donor >= threshold) != original_class
        local_flip = (local >= threshold) != original_class
        difference = center_flip != local_flip
        endpoint_margin = np.minimum.reduce((
            np.abs(full-threshold), np.abs(centroid-threshold),
            np.abs(local-threshold)))
        counts = {
            "centroid_class_flips": int(center_flip.sum()),
            "global_donor_class_flips": int(global_flip.sum()),
            "local_donor_class_flips": int(local_flip.sum()),
            "centroid_local_class_flip_disagreement": int(difference.sum()),
            "global_local_class_flip_disagreement": int(np.sum(
                global_flip != local_flip)),
            "centroid_only_vs_local": int(np.sum(center_flip & ~local_flip)),
            "local_only_vs_centroid": int(np.sum(local_flip & ~center_flip)),
            "both_centroid_and_local_flip": int(np.sum(
                center_flip & local_flip)),
            "centroid_local_disagreements_all_endpoint_margins_at_least_0p01":
                int(np.sum(difference & (endpoint_margin >= 0.01))),
            "only_projection_displayed_accounts": int(selected.sum()),
            "only_projection_local_class_flips": int(np.sum(
                selected & local_flip)),
            "local_donor_distance_median_64d": float(np.median(distances)),
            "centroid_distance_median_64d": float(np.median(
                expected_center_distances)),
            "local_donor_nearer_than_centroid_accounts": int(np.sum(
                distances < expected_center_distances)),
            "local_donor_distance_max_64d": float(np.max(distances)),
            "distinct_local_donor_rows": int(len(np.unique(local_rows))),
        }
        for key, value in counts.items():
            if isinstance(value, float):
                if abs(value-row[key]) > 1e-10:
                    raise ValueError(f"saved {key} differs")
            elif value != row[key]:
                raise ValueError(f"saved {key} differs")
        runs.append({"seed": seed, "accounts": len(ids),
                     "max_probability_replay_error": replay_error,
                     **counts})
    if [r["seed"] for r in runs] != [0, 1, 2]:
        raise ValueError("three frozen MGTAB runs differ")
    OUT.write_text(json.dumps({
        "status": "local_published_content_donor_rows_and_probabilities_read_back",
        "source_certificate_sha256": sha256_file(CERT),
        "accounts": sum(r["accounts"] for r in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"accounts": 3060,
                      "local_flips": sum(r["local_donor_class_flips"]
                                         for r in runs),
                      "centroid_local_disagreements": sum(
                          r["centroid_local_class_flip_disagreement"]
                          for r in runs),
                      "max_probability_replay_error": max(
                          r["max_probability_replay_error"] for r in runs)}))


if __name__ == "__main__":
    main()
