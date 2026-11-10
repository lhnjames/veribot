"""Audit source-only inference against all frozen MGTAB reduced-model predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_reduced_projection_runtime import (
    MGTABReducedProjectionRuntime, sha256_file)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_reduced_projection_ab_group"


def main() -> None:
    original_torch_load = torch.load
    original_joblib_load = joblib.load
    def no_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels_bot.pt", "label.pt", "labels.pt"}:
            raise AssertionError("Inference opened a label tensor")
        return original_torch_load(path, *args, **kwargs)
    def only_new_model(path, *args, **kwargs):
        if Path(path).resolve().parent.parent != OUT.resolve():
            raise AssertionError("Inference loaded another classifier's weight")
        return original_joblib_load(path, *args, **kwargs)
    runs = []
    with patch.object(torch, "load", no_labels), patch.object(joblib, "load", only_new_model):
        for seed in (0, 1, 2):
            runtime = MGTABReducedProjectionRuntime(ROOT, seed)
            path = OUT / f"seed{seed}/test_predictions.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["manifest"]["model_sha256"] != sha256_file(
                    OUT / f"seed{seed}/model.joblib"):
                raise ValueError("Saved prediction model hash differs")
            ids = [row["account_id"] for row in saved["predictions"]]
            indices = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
            scores = runtime.predict_rows(indices)
            saved_probability = np.asarray([row["probability"] for row in saved["predictions"]])
            saved_logit = np.asarray([row["logit"] for row in saved["predictions"]])
            saved_residual = np.asarray([row["projection_group_raw_logit"]
                                         for row in saved["predictions"]])
            difference = float(np.max(np.abs(scores["probability"] - saved_probability)))
            logit_difference = float(np.max(np.abs(scores["logit"] - saved_logit)))
            residual_difference = float(np.max(np.abs(
                scores["projection_raw_logit"] - saved_residual)))
            if (difference > 1e-12 or logit_difference > 1e-11
                    or residual_difference > 1e-11):
                raise ValueError("Live source inference differs from frozen predictions")
            x = runtime.evidence_matrix(indices)
            named_x = x.copy()
            named_x[:, 86:94] = 0.0
            named_x[:, 180:188] = 0.0
            booster = runtime.model.booster_
            zero = np.asarray(booster.predict(np.zeros_like(x), raw_score=True))
            named_effects = []
            for name, members in runtime.group_indices.items():
                if name == "published_tweet_projections":
                    continue
                masked = named_x.copy()
                masked[:, members] = 0.0
                masked[:, np.asarray(members) + 94] = 0.0
                named_effects.append(scores["named_logit"] - np.asarray(
                    booster.predict(masked, raw_score=True)))
            reconstructed = zero + sum(named_effects) + scores["projection_bounded_logit"]
            full_additivity = float(np.max(np.abs(reconstructed - scores["logit"])))
            if full_additivity > 1e-9:
                raise ValueError("Full fixed-test source groups do not reconstruct logit")
            denominator = sum(np.abs(effect) for effect in named_effects) + \
                np.abs(scores["projection_bounded_logit"])
            projection_share = np.divide(
                np.abs(scores["projection_bounded_logit"]), denominator,
                out=np.zeros_like(denominator), where=denominator > 0)
            labels = np.asarray([row["label"] for row in saved["predictions"]], dtype=int)
            with_projection = scores["probability"] >= runtime.threshold
            named_only = expit(scores["named_logit"]) >= runtime.threshold
            with_projection_correct = with_projection == labels
            named_only_correct = named_only == labels
            probes = [0, len(ids)//2, len(ids)-1]
            non_test = next(uid for uid in runtime.ids if uid not in set(ids))
            cards = [runtime.predict(ids[i]) for i in probes]
            cards.append(runtime.predict(non_test))
            for j, card in zip(probes, cards[:3], strict=True):
                if (abs(card["probability"] - saved_probability[j]) > 1e-12
                        or abs(card["projection_absolute_effect_share"] -
                               projection_share[j]) > 1e-12):
                    raise ValueError("Source card differs from saved prediction")
            if any(card["group_logit_additivity_error"] > 1e-9 for card in cards):
                raise ValueError("Source card groups do not reconstruct logit")
            share_path = OUT / f"seed{seed}/test_projection_effect_share.npz"
            np.savez_compressed(
                share_path, ids=np.asarray(ids),
                projection_absolute_effect_share=projection_share,
                named_only_probability=expit(scores["named_logit"]),
                full_probability=scores["probability"],
                projection_changes_class=(with_projection != named_only),
                content_effect_was_clipped=(np.abs(scores["projection_raw_logit"]) >
                                            runtime.cap))
            high_half = projection_share > .5
            high_quarter = projection_share > .25
            runs.append({
                "seed": seed,
                "test_accounts_checked": len(ids),
                "non_test_probe_accounts": 1,
                "source_cards_checked": len(cards),
                "max_probability_difference": difference,
                "max_logit_difference": logit_difference,
                "max_projection_residual_difference": residual_difference,
                "max_group_logit_additivity_error": max(
                    card["group_logit_additivity_error"] for card in cards),
                "full_test_max_group_logit_additivity_error": full_additivity,
                "projection_changes_test_class": int(np.count_nonzero(
                    with_projection != named_only)),
                "projection_rescues_named_only_errors": int(np.count_nonzero(
                    with_projection_correct & ~named_only_correct)),
                "projection_harms_named_only_correct": int(np.count_nonzero(
                    ~with_projection_correct & named_only_correct)),
                "named_only_test_correct": int(named_only_correct.sum()),
                "full_test_correct": int(with_projection_correct.sum()),
                "projection_absolute_effect_share_median": float(np.median(
                    projection_share)),
                "projection_absolute_effect_share_p90": float(np.quantile(
                    projection_share, .9)),
                "projection_share_above_25_percent_accounts": int(high_quarter.sum()),
                "projection_share_above_50_percent_accounts": int(high_half.sum()),
                "projection_share_above_50_percent_correct": int(np.count_nonzero(
                    with_projection_correct & high_half)),
                "projection_share_above_50_percent_class_changes": int(
                    np.count_nonzero((with_projection != named_only) & high_half)),
                "per_account_projection_share_sha256": sha256_file(share_path),
                "label_tensor_read_attempts": 0,
                "other_classifier_weight_reads": 0,
                "model_sha256": saved["manifest"]["model_sha256"],
                "saved_predictions_sha256": sha256_file(path),
            })
            print(runs[-1], flush=True)
    certificate = {
        "scope": "mgtab_reduced_projection_full_fixed_test_live_source_replay",
        "inference_manifest_sha256": sha256_file(OUT / "inference_manifest.json"),
        "source_depth": "published numeric profile and typed-edge tensors plus averaged-tweet projection",
        "semantic_content_truth_established": False,
        "runs": runs,
    }
    path = ROOT / "research/certificates/mgtab_reduced_projection_full_live.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
