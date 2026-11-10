"""Compare published content-row replacement with encoded group withholding."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
GROUP_CERT = ROOT / "research/certificates/best_source_group_effects.json"
BALANCE_CERT = ROOT / "research/certificates/best_reason_direction_balance.json"
BALANCE_ROWS = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
OUT_DIR = ROOT / "output/v3_mgtab_content_source_centroid_response"
OUT_CERT = ROOT / "research/certificates/mgtab_content_source_centroid_response.json"
PROJECTION_LEVEL = "published_averaged_tweet_tensor_train_only_projection"


def encode(runtime: MGTABFullAtomicRuntime, facts: np.ndarray,
           supports: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + 5.0)
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def main() -> None:
    group_cert = json.loads(GROUP_CERT.read_text(encoding="utf-8"))
    balance_cert = json.loads(BALANCE_CERT.read_text(encoding="utf-8"))
    if (group_cert["status"] !=
            "frozen_source_group_input_effects_bound_to_full_fact_effects" or
            balance_cert["status"] !=
            "all_current_reason_cards_direction_display_audited" or
            balance_cert["account_sidecar_sha256"] !=
            sha256_file(BALANCE_ROWS)):
        raise ValueError("current source, group or balance certificates differ")
    only_projection: dict[int, set[str]] = {seed: set() for seed in (0, 1, 2)}
    with gzip.open(BALANCE_ROWS, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["dataset"] == "mgtab" and all(
                    item["source_level"] == PROJECTION_LEVEL
                    for item in row["balanced_displayed_items"]):
                only_projection[row["seed"]].add(row["account_id"])
    group_runs = {(row["dataset"], row["seed"]): row
                  for row in group_cert["runs"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        group = group_runs["mgtab", seed]
        group_path = Path(group["group_effect_path"])
        if sha256_file(group_path) != group["group_effect_sha256"]:
            raise ValueError("group mask matrix changed")
        with np.load(group_path, allow_pickle=False) as saved:
            ids = saved["ids"].astype(str)
            group_names = saved["group_names"].astype(str).tolist()
            full_saved = saved["full_probability"].astype(float)
            without_saved = saved["without_group_probability"].astype(float)
            threshold = float(saved["threshold"])
        if (group_names != ["published_profile",
                            "published_typed_relations",
                            "published_averaged_tweet_projection"] or
                threshold != runtime.threshold or len(ids) != 1020):
            raise ValueError("MGTAB frozen group and population differ")
        indices = np.array([runtime.index[uid] for uid in ids], dtype=int)
        projection_path = runtime.projection_path
        with np.load(projection_path, allow_pickle=False) as projection:
            scaler_mean = projection["scaler_mean"].astype(float)
            scaler_scale = projection["scaler_scale"].astype(float)
            pca_mean = projection["pca_mean"].astype(float)
            components = projection["pca_components"].astype(float)
        if (scaler_mean.shape != (768,) or
                scaler_scale.shape != (768,) or
                pca_mean.shape != (768,) or
                components.shape != (64, 768) or
                np.any(scaler_scale <= 0)):
            raise ValueError("frozen train-only PCA differs")
        published = torch.load(runtime.data_dir / "features.pt",
                               map_location="cpu", weights_only=True).numpy()
        if published.shape != (10199, 788):
            raise ValueError("published content tensor differs")
        original_projection = ((published[indices, 20:].astype(float)
                                - scaler_mean) / scaler_scale - pca_mean)
        original_projection = original_projection @ components.T
        projection_error = float(np.max(np.abs(
            original_projection - runtime.facts[indices, 86:150])))
        if projection_error > 1e-9:
            raise ValueError("published source-to-projection replay differs")
        facts = runtime.facts[indices].copy()
        original_encoded = encode(runtime, facts, runtime.supports[indices])
        full = runtime.model.predict_proba(original_encoded)[:, 1]
        full_error = float(np.max(np.abs(full-full_saved)))
        if full_error > 1e-10:
            raise ValueError("frozen full prediction differs")
        withheld = original_encoded.copy()
        withheld[:, 86:150] = 0.0
        withheld[:, 236:300] = 0.0
        without = runtime.model.predict_proba(withheld)[:, 1]
        mask_error = float(np.max(np.abs(without-without_saved[:, 2])))
        if mask_error > 1e-10:
            raise ValueError("frozen content-group mask differs")
        # Replace the released average-tweet tensor row, then re-run its
        # frozen train-only projection. This is not a corrected tweet record.
        edited_content_rows = published[indices, 20:].astype(float).copy()
        edited_content_rows[:] = scaler_mean
        edited_projection = ((edited_content_rows - scaler_mean) /
                             scaler_scale - pca_mean) @ components.T
        centroid_projection = (-pca_mean) @ components.T
        source_projection_error = float(np.max(np.abs(
            edited_projection - centroid_projection)))
        if source_projection_error > 1e-12:
            raise ValueError("edited published content row does not replay")
        facts[:, 86:150] = edited_projection
        replaced_encoded = encode(runtime, facts, runtime.supports[indices])
        if not np.array_equal(replaced_encoded[:, :86],
                              original_encoded[:, :86]) or not np.array_equal(
                                  replaced_encoded[:, 150:236],
                                  original_encoded[:, 150:236]):
            raise ValueError("content-only source replacement altered other groups")
        replaced = runtime.model.predict_proba(replaced_encoded)[:, 1]
        original_class = full >= threshold
        source_flip = (replaced >= threshold) != original_class
        mask_flip = (without >= threshold) != original_class
        selected = np.isin(ids, list(only_projection[seed]))
        source_only = source_flip & ~mask_flip
        mask_only = mask_flip & ~source_flip
        source_margin = float(np.min(np.minimum(
            np.abs(full[source_flip] - threshold),
            np.abs(replaced[source_flip] - threshold))))
        mask_margin = float(np.min(np.minimum(
            np.abs(full[mask_flip] - threshold),
            np.abs(without[mask_flip] - threshold))))
        out_path = OUT_DIR / f"seed{seed}.npz"
        np.savez_compressed(out_path, ids=ids, full_probability=full,
                            centroid_source_probability=replaced,
                            encoded_group_zero_probability=without,
                            threshold=np.asarray(threshold),
                            selected_only_projection=selected,
                            train_centroid_projection=centroid_projection)
        row = {
            "dataset": "mgtab", "seed": seed, "accounts": len(ids),
            "only_projection_displayed_accounts": int(selected.sum()),
            "source_centroid_class_flips": int(source_flip.sum()),
            "encoded_group_zero_class_flips": int(mask_flip.sum()),
            "class_flip_disagreement": int(np.sum(source_flip != mask_flip)),
            "source_only_class_flips": int(source_only.sum()),
            "encoded_mask_only_class_flips": int(mask_only.sum()),
            "only_projection_predicted_bot": int(np.sum(
                selected & original_class)),
            "only_projection_source_centroid_class_flips": int(
                source_flip[selected].sum()),
            "only_projection_encoded_group_zero_class_flips": int(
                mask_flip[selected].sum()),
            "only_projection_class_flip_disagreement": int(np.sum(
                source_flip[selected] != mask_flip[selected])),
            "minimum_source_flip_endpoint_threshold_margin": source_margin,
            "minimum_encoded_mask_flip_endpoint_threshold_margin": mask_margin,
            "mean_absolute_probability_difference_between_operations": float(
                np.mean(np.abs(replaced-without))),
            "max_original_projection_replay_difference": projection_error,
            "max_edited_source_projection_difference":
                source_projection_error,
            "max_full_probability_replay_difference": full_error,
            "max_encoded_mask_replay_difference": mask_error,
            "published_features_sha256": sha256_file(
                runtime.data_dir / "features.pt"),
            "projection_artifact_sha256": sha256_file(projection_path),
            "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
            "reference_sha256": sha256_file(
                runtime.run_dir / "label_blind_reference.json.gz"),
            "group_matrix_sha256": sha256_file(group_path),
            "result_path": str(out_path),
            "result_sha256": sha256_file(out_path),
        }
        runs.append(row)
        print(json.dumps({key: row[key] for key in (
            "seed", "accounts", "source_centroid_class_flips",
            "encoded_group_zero_class_flips", "class_flip_disagreement",
            "only_projection_displayed_accounts")}), flush=True)
    if sum(row["only_projection_displayed_accounts"] for row in runs) != 302:
        raise ValueError("published projection-only report population differs")
    OUT_CERT.write_text(json.dumps({
        "status": "published_content_row_replacement_and_input_mask_compared",
        "scope": "three_mgtab_fixed_test_runs_all_3060_decisions",
        "source_operation": "replace one published average-tweet tensor row with the train-only content centroid; reapply the frozen PCA projection, reference evidence and classifier",
        "encoded_operation": "zero both evidence tails of all 64 content-projection inputs",
        "source_operation_is_not_a_known_error_correction": True,
        "fixed_test_labels_read": False,
        "balance_certificate_sha256": sha256_file(BALANCE_CERT),
        "group_certificate_sha256": sha256_file(GROUP_CERT),
        "balanced_rows_sha256": sha256_file(BALANCE_ROWS),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
