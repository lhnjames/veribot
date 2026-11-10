"""Read back the two published-content source replacement comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/mgtab_content_source_baseline_stability.json"
PRIOR = ROOT / "research/certificates/mgtab_content_source_centroid_response.json"
PRIOR_READBACK = ROOT / "research/certificates/mgtab_content_source_centroid_response_readback.json"
OUT = ROOT / "research/certificates/mgtab_content_source_baseline_stability_readback.json"


def main() -> None:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if (cert["status"] !=
            "two_label_blind_published_content_replacements_compared" or
            cert["prior_centroid_certificate_sha256"] != sha256_file(PRIOR) or
            cert["prior_centroid_readback_sha256"] !=
            sha256_file(PRIOR_READBACK)):
        raise ValueError("source replacement comparison lacks prior binding")
    source_path = resolve_mgtab_source_dir(ROOT) / "features.pt"
    published = torch.load(source_path, map_location="cpu",
                           weights_only=True).numpy()
    if published.shape != (10199, 788):
        raise ValueError("published source tensor differs")
    previous_by_seed = {row["seed"]: row for row in prior["runs"]}
    runs = []
    for row in cert["runs"]:
        seed = row["seed"]
        old = previous_by_seed[seed]
        path = Path(row["result_path"])
        old_path = Path(old["result_path"])
        projection_path = (ROOT / "output/v3_mgtab_ab_spline_content" /
                           f"selected/seed{seed}/projection.npz")
        model_path = (ROOT / "output/v3_mgtab_ab_group_additive_tree" /
                      f"seed{seed}/model.joblib")
        reference_path = (ROOT / "output/v3_mgtab_ab_group_additive_tree" /
                          f"seed{seed}/label_blind_reference.json.gz")
        if (row["dataset"] != "mgtab" or row["accounts"] != 1020 or
                sha256_file(path) != row["result_sha256"] or
                sha256_file(old_path) != row["prior_result_sha256"] or
                sha256_file(source_path) != row[
                    "published_features_sha256"] or
                sha256_file(projection_path) != row[
                    "projection_artifact_sha256"] or
                sha256_file(model_path) != row["model_sha256"] or
                sha256_file(reference_path) != row["reference_sha256"]):
            raise ValueError("saved source, projection or model changed")
        with np.load(path, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            center = z["centroid_probability"].astype(float)
            exemplar = z["train_exemplar_probability"].astype(float)
            threshold = float(z["threshold"])
            nearest_row = int(z["nearest_train_row"])
            selected = z["selected_only_projection"].astype(bool)
        with np.load(old_path, allow_pickle=False) as z:
            old_ids = z["ids"].astype(str)
            old_full = z["full_probability"].astype(float)
            old_center = z["centroid_source_probability"].astype(float)
            old_threshold = float(z["threshold"])
            old_selected = z["selected_only_projection"].astype(bool)
        with np.load(projection_path, allow_pickle=False) as z:
            train = z["train_indices"].astype(int)
            mean = z["scaler_mean"].astype(float)
            scale = z["scaler_scale"].astype(float)
            pca_mean = z["pca_mean"].astype(float)
            components = z["pca_components"].astype(float)
        if (not np.array_equal(ids, old_ids) or
                not np.array_equal(selected, old_selected) or
                np.max(np.abs(full-old_full)) > 1e-12 or
                np.max(np.abs(center-old_center)) > 1e-12 or
                threshold != old_threshold or len(train) != 7139 or
                nearest_row not in set(train.tolist()) or
                np.any(np.isin(ids.astype(int), train))):
            raise ValueError("test population or prior source response differs")
        projected_train = ((published[train, 20:].astype(float)-mean) /
                           scale-pca_mean) @ components.T
        center_projection = (-pca_mean) @ components.T
        distances = np.linalg.norm(projected_train-center_projection, axis=1)
        position = int(np.lexsort((train, distances))[0])
        if (nearest_row != int(train[position]) or
                abs(float(distances[position]) - row[
                    "nearest_train_source_distance_in_64d_projection"])
                > 1e-10):
            raise ValueError("nearest observed train source row differs")
        full_class = full >= threshold
        center_flip = (center >= threshold) != full_class
        exemplar_flip = (exemplar >= threshold) != full_class
        disagreement = center_flip != exemplar_flip
        endpoint_margin = np.minimum.reduce((
            np.abs(full-threshold), np.abs(center-threshold),
            np.abs(exemplar-threshold)))
        counts = {
            "centroid_class_flips": int(center_flip.sum()),
            "nearest_train_exemplar_class_flips": int(exemplar_flip.sum()),
            "class_flip_disagreement": int(disagreement.sum()),
            "both_class_flip": int(np.sum(center_flip & exemplar_flip)),
            "centroid_only_class_flip": int(np.sum(center_flip & ~exemplar_flip)),
            "exemplar_only_class_flip": int(np.sum(exemplar_flip & ~center_flip)),
            "only_projection_displayed_accounts": int(selected.sum()),
            "only_projection_class_flip_disagreement": int(np.sum(
                selected & disagreement)),
            "disagreements_with_all_endpoint_margins_at_least_0p001": int(
                np.sum(disagreement & (endpoint_margin >= 0.001))),
            "disagreements_with_all_endpoint_margins_at_least_0p01": int(
                np.sum(disagreement & (endpoint_margin >= 0.01))),
        }
        if any(value != row[key] for key, value in counts.items()):
            raise ValueError("source replacement stability count differs")
        runs.append({"seed": seed, "accounts": len(ids),
                     "nearest_train_source_row": nearest_row, **counts})
    if ([row["seed"] for row in runs] != [0, 1, 2] or
            sum(row["class_flip_disagreement"] for row in runs) != 480):
        raise ValueError("three MGTAB source stability runs differ")
    OUT.write_text(json.dumps({
        "status": "all_published_content_replacement_stability_rows_read_back",
        "source_certificate_sha256": sha256_file(CERT),
        "accounts": sum(row["accounts"] for row in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accounts": 3060,
                      "disagreements": 480,
                      "disagreements_margin_0p01": sum(row[
                          "disagreements_with_all_endpoint_margins_at_least_0p01"]
                          for row in runs)}))


if __name__ == "__main__":
    main()
