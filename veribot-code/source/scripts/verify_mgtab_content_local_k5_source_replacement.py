"""Read back five nearest MGTAB donor rows and rerun frozen probabilities."""

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
CERT = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement.json"
PRIOR_CERT = ROOT / "research/certificates/mgtab_content_local_source_replacement.json"
PRIOR_READBACK = ROOT / "research/certificates/mgtab_content_local_source_replacement_readback.json"
OUT = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement_readback.json"


def main() -> None:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_CERT.read_text(encoding="utf-8"))
    if (cert["status"] != "five_nearest_published_content_donor_responses" or
            cert["scope"] != "all_3060_mgtab_fixed_test_decisions" or
            cert["known_error_correction"] is not False or
            cert["fixed_test_labels_read"] is not False or
            cert["prior_local_certificate_sha256"] !=
            sha256_file(PRIOR_CERT) or
            cert["prior_local_readback_sha256"] !=
            sha256_file(PRIOR_READBACK)):
        raise ValueError("five-neighbor audit lacks source binding")
    prior_by_seed = {r["seed"]: r for r in prior["runs"]}
    runs = []
    for row in cert["runs"]:
        seed = int(row["seed"])
        old = prior_by_seed[seed]
        path = Path(row["result_path"])
        old_path = Path(old["result_path"])
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        source_path = runtime.data_dir / "features.pt"
        if (row["dataset"] != "mgtab" or row["accounts"] != 1020 or
                row["k"] != 5 or
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
            raise ValueError("saved source, reference, or model hash differs")
        with np.load(path, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            centroid = z["centroid_probability"].astype(float)
            donor_rows = z["donor_rows"].astype(int)
            donor_distances = z[
                "donor_distances_in_64d_projection"].astype(float)
            probabilities = z["donor_probabilities"].astype(float)
            selected = z["selected_only_projection"].astype(bool)
            threshold = float(z["threshold"])
        with np.load(old_path, allow_pickle=False) as z:
            if (not np.array_equal(ids, z["ids"].astype(str)) or
                    not np.array_equal(selected, z[
                        "selected_only_projection"].astype(bool)) or
                    not np.array_equal(donor_rows[:, 0], z[
                        "local_donor_row"].astype(int)) or
                    np.max(np.abs(full-z["full_probability"])) > 1e-12 or
                    np.max(np.abs(centroid-z[
                        "centroid_probability"])) > 1e-12 or
                    np.max(np.abs(probabilities[:, 0]-z[
                        "local_donor_probability"])) > 1e-12 or
                    np.max(np.abs(donor_distances[:, 0]-z[
                        "local_donor_distance_in_64d_projection"])) > 1e-10 or
                    threshold != float(z["threshold"])):
                raise ValueError("first neighbor differs from prior audit")
        if (donor_rows.shape != (1020, 5) or
                donor_distances.shape != (1020, 5) or
                probabilities.shape != (1020, 5) or
                threshold != runtime.threshold):
            raise ValueError("five-neighbor population or threshold differs")
        with np.load(runtime.projection_path, allow_pickle=False) as z:
            train_rows = np.sort(z["train_indices"].astype(int))
            mean = z["scaler_mean"].astype(float)
            scale = z["scaler_scale"].astype(float)
            pca_mean = z["pca_mean"].astype(float)
            components = z["pca_components"].astype(float)
        source = torch.load(source_path, map_location="cpu",
                            weights_only=True).numpy()
        test_rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        if (source.shape != (10199, 788) or len(train_rows) != 7139 or
                len(np.unique(train_rows)) != len(train_rows) or
                np.any(np.isin(test_rows, train_rows))):
            raise ValueError("source population or train/test split differs")
        train_projection = ((source[train_rows, 20:].astype(float)-mean) /
                            scale-pca_mean) @ components.T
        test_projection = ((source[test_rows, 20:].astype(float)-mean) /
                           scale-pca_mean) @ components.T
        if np.max(np.abs(test_projection-runtime.facts[
                test_rows, 86:150])) > 1e-10:
            raise ValueError("published test projection differs")
        expected_rows = np.empty((1020, 5), dtype=int)
        expected_distances = np.empty((1020, 5), dtype=float)
        for start in range(0, 1020, 32):
            stop = min(start+32, 1020)
            square = cdist(test_projection[start:stop], train_projection,
                           metric="sqeuclidean")
            positions = np.argsort(square, axis=1, kind="stable")[:, :5]
            expected_rows[start:stop] = train_rows[positions]
            expected_distances[start:stop] = np.sqrt(np.take_along_axis(
                square, positions, axis=1))
        if (not np.array_equal(donor_rows, expected_rows) or
                np.max(np.abs(donor_distances-expected_distances)) > 1e-10):
            raise ValueError("five nearest source rows or distances differ")
        base = runtime.facts[test_rows]
        recomputed = np.empty((1020, 5), dtype=float)
        for rank in range(5):
            modified = base.copy()
            donor_content = source[donor_rows[:, rank], 20:].astype(float)
            modified[:, 86:150] = ((donor_content-mean)/scale-pca_mean) @ (
                components.T)
            batch = runtime.reference.transform(modified)
            reliability = batch.reliability.copy()
            reliability[:, 62:86] *= runtime.supports[test_rows] / (
                runtime.supports[test_rows]+5.0)
            inputs = np.column_stack((np.log1p(batch.upper)*reliability,
                                      np.log1p(batch.lower)*reliability))
            recomputed[:, rank] = runtime.model.predict_proba(inputs)[:, 1]
        replay_error = float(np.max(np.abs(recomputed-probabilities)))
        if replay_error > 1e-10:
            raise ValueError("five frozen donor probabilities differ")
        flip = (probabilities >= threshold) != (full >= threshold)[:, None]
        flip_count = flip.sum(axis=1)
        counts = {
            "at_least_one_of_k_class_flips": int(np.sum(flip_count > 0)),
            "all_k_class_flip": int(np.sum(flip_count == 5)),
            "mixed_class_flip_across_k": int(np.sum(
                (flip_count > 0) & (flip_count < 5))),
            "no_k_class_flip": int(np.sum(flip_count == 0)),
            "first_donor_class_flips": int(np.sum(flip[:, 0])),
            "first_no_flip_but_another_flips": int(np.sum(
                ~flip[:, 0] & (flip_count > 0))),
            "first_flips_but_another_does_not": int(np.sum(
                flip[:, 0] & (flip_count < 5))),
            "only_projection_displayed_accounts": int(selected.sum()),
            "only_projection_any_k_class_flips": int(np.sum(
                selected & (flip_count > 0))),
            "median_fifth_minus_first_distance_64d": float(np.median(
                donor_distances[:, -1]-donor_distances[:, 0])),
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
        raise ValueError("three MGTAB model seeds differ")
    OUT.write_text(json.dumps({
        "status": "five_local_donor_rows_distances_and_probabilities_read_back",
        "source_certificate_sha256": sha256_file(CERT),
        "accounts": sum(r["accounts"] for r in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "accounts": 3060,
        "any_flip": sum(r["at_least_one_of_k_class_flips"] for r in runs),
        "mixed": sum(r["mixed_class_flip_across_k"] for r in runs),
        "all_flip": sum(r["all_k_class_flip"] for r in runs),
        "max_probability_replay_error": max(
            r["max_probability_replay_error"] for r in runs)}))


if __name__ == "__main__":
    main()
