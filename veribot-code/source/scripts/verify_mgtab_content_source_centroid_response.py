"""Read back all published-content replacement results independently."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/mgtab_content_source_centroid_response.json"
GROUP_CERT = ROOT / "research/certificates/best_source_group_effects.json"
BALANCE_CERT = ROOT / "research/certificates/best_reason_direction_balance.json"
BALANCE = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
OUT = ROOT / "research/certificates/mgtab_content_source_centroid_response_readback.json"
PROJECTION_LEVEL = "published_averaged_tweet_tensor_train_only_projection"


def main() -> None:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    groups = json.loads(GROUP_CERT.read_text(encoding="utf-8"))
    if (cert["status"] !=
            "published_content_row_replacement_and_input_mask_compared" or
            cert["balance_certificate_sha256"] !=
            sha256_file(BALANCE_CERT) or
            cert["group_certificate_sha256"] != sha256_file(GROUP_CERT) or
            cert["balanced_rows_sha256"] != sha256_file(BALANCE)):
        raise ValueError("published content response certificate differs")
    selected: dict[int, set[str]] = {seed: set() for seed in (0, 1, 2)}
    with gzip.open(BALANCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            card = json.loads(line)
            if card["dataset"] == "mgtab" and all(
                    item["source_level"] == PROJECTION_LEVEL
                    for item in card["balanced_displayed_items"]):
                selected[card["seed"]].add(card["account_id"])
    group_runs = {(row["dataset"], row["seed"]): row
                  for row in groups["runs"]}
    source_features = resolve_mgtab_source_dir(ROOT) / "features.pt"
    runs = []
    for row in cert["runs"]:
        seed = row["seed"]
        group = group_runs["mgtab", seed]
        result = Path(row["result_path"])
        group_path = Path(group["group_effect_path"])
        model_path = (ROOT / "output/v3_mgtab_ab_group_additive_tree" /
                      f"seed{seed}/model.joblib")
        reference_path = (ROOT / "output/v3_mgtab_ab_group_additive_tree" /
                          f"seed{seed}/label_blind_reference.json.gz")
        if (row["dataset"] != "mgtab" or row["accounts"] != 1020 or
                sha256_file(result) != row["result_sha256"] or
                sha256_file(group_path) != row["group_matrix_sha256"] or
                sha256_file(model_path) != row["model_sha256"] or
                sha256_file(reference_path) != row["reference_sha256"] or
                sha256_file(source_features) != row[
                    "published_features_sha256"] or
                sha256_file(ROOT / "output/v3_mgtab_ab_spline_content" /
                            f"selected/seed{seed}/projection.npz") !=
                row["projection_artifact_sha256"]):
            raise ValueError("frozen content response artifact differs")
        with np.load(result, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            full = z["full_probability"].astype(float)
            source = z["centroid_source_probability"].astype(float)
            masked = z["encoded_group_zero_probability"].astype(float)
            flag = z["selected_only_projection"].astype(bool)
            threshold = float(z["threshold"])
            centroid = z["train_centroid_projection"].astype(float)
        with np.load(group_path, allow_pickle=False) as z:
            group_ids = z["ids"].astype(str)
            group_names = z["group_names"].astype(str).tolist()
            saved_full = z["full_probability"].astype(float)
            saved_mask = z["without_group_probability"].astype(float)
            group_threshold = float(z["threshold"])
        projection_path = (ROOT / "output/v3_mgtab_ab_spline_content" /
                           f"selected/seed{seed}/projection.npz")
        with np.load(projection_path, allow_pickle=False) as z:
            expected_centroid = (-z["pca_mean"].astype(float)) @ z[
                "pca_components"].astype(float).T
        if (len(ids) != 1020 or not np.array_equal(ids, group_ids) or
                group_names[2] != "published_averaged_tweet_projection" or
                threshold != group_threshold or
                np.max(np.abs(full-saved_full)) > 1e-10 or
                np.max(np.abs(masked-saved_mask[:, 2])) > 1e-10 or
                np.max(np.abs(centroid-expected_centroid)) > 1e-12 or
                set(ids[flag]) != selected[seed]):
            raise ValueError("source population, centroid or saved mask differs")
        original_class = full >= threshold
        source_flip = (source >= threshold) != original_class
        mask_flip = (masked >= threshold) != original_class
        checks = {
            "only_projection_displayed_accounts": int(flag.sum()),
            "only_projection_predicted_bot": int(np.sum(flag & original_class)),
            "source_centroid_class_flips": int(source_flip.sum()),
            "encoded_group_zero_class_flips": int(mask_flip.sum()),
            "class_flip_disagreement": int(np.sum(source_flip != mask_flip)),
            "source_only_class_flips": int(np.sum(source_flip & ~mask_flip)),
            "encoded_mask_only_class_flips": int(np.sum(mask_flip & ~source_flip)),
            "only_projection_source_centroid_class_flips": int(np.sum(
                flag & source_flip)),
            "only_projection_encoded_group_zero_class_flips": int(np.sum(
                flag & mask_flip)),
            "only_projection_class_flip_disagreement": int(np.sum(
                flag & (source_flip != mask_flip))),
        }
        for key, value in checks.items():
            if value != row[key]:
                raise ValueError(f"published content count differs: {key}")
        margin_source = float(np.min(np.minimum(
            np.abs(full[source_flip]-threshold),
            np.abs(source[source_flip]-threshold))))
        margin_mask = float(np.min(np.minimum(
            np.abs(full[mask_flip]-threshold),
            np.abs(masked[mask_flip]-threshold))))
        if (abs(margin_source-row[
                "minimum_source_flip_endpoint_threshold_margin"]) > 1e-15 or
                abs(margin_mask-row[
                    "minimum_encoded_mask_flip_endpoint_threshold_margin"])
                > 1e-15):
            raise ValueError("source or encoded flip margin differs")
        runs.append({"seed": seed, "accounts": len(ids), **checks,
                     "minimum_source_flip_margin": margin_source,
                     "minimum_encoded_mask_flip_margin": margin_mask})
    if ([run["seed"] for run in runs] != [0, 1, 2] or
            sum(run["only_projection_displayed_accounts"]
                for run in runs) != 302 or
            sum(run["class_flip_disagreement"] for run in runs) != 830):
        raise ValueError("all MGTAB fixed runs differ")
    OUT.write_text(json.dumps({
        "status": "all_published_content_source_responses_read_back",
        "source_certificate_sha256": sha256_file(CERT),
        "accounts": sum(run["accounts"] for run in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accounts": 3060,
                      "source_flips": sum(r["source_centroid_class_flips"]
                                          for r in runs),
                      "mask_flips": sum(r["encoded_group_zero_class_flips"]
                                        for r in runs),
                      "disagreements": 830}))


if __name__ == "__main__":
    main()
