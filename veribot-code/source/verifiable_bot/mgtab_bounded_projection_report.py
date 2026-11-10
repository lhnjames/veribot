"""Expose the frozen MGTAB numeric projection as an explicit capped residual."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scipy.special import expit

from .mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_projection_report(runtime: MGTABFullAtomicRuntime,
                              decision: dict) -> dict:
    """Split named evidence from the validation-capped content contribution."""
    if runtime.variant != "ab_group_additive":
        raise ValueError("bounded projection needs the additive A/B model")
    lock_path = (runtime.root /
                 "output/v3_mgtab_explicit_bounded_projection_residual/selection_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "fixed_before_bounded_residual_test_evaluation":
        raise ValueError("bounded residual cap is not validation locked")
    expected = next((row for row in lock["models"]
                     if row["seed"] == runtime.seed), None)
    if (expected is None or
            expected["model_sha256"] != _sha256(runtime.run_dir / "model.joblib") or
            expected["reference_version"] != runtime.reference.version or
            expected["threshold"] != runtime.threshold):
        raise ValueError("bounded residual model differs from selection lock")
    cap = float(lock["selected_cap"])
    if not 0 < cap < 10:
        raise ValueError("invalid bounded residual cap")
    group = decision["group_dependencies"]["published_tweet_projections"]
    full_logit = float(decision["logit"])
    raw_residual = float(group["group_logit_difference"])
    named_logit = full_logit - raw_residual
    if abs(float(expit(named_logit)) - float(
            group["probability_without_group"])) > 1e-12:
        raise ValueError("named evidence and direct projection mask differ")
    bounded_residual = max(-cap, min(cap, raw_residual))
    bounded_logit = named_logit + bounded_residual
    bounded_probability = float(expit(bounded_logit))
    named_absolute_effect = sum(
        abs(float(item["group_logit_difference"]))
        for name, item in decision["group_dependencies"].items()
        if name != "published_tweet_projections")
    total_absolute_effect = named_absolute_effect + abs(bounded_residual)
    residual_absolute_effect_share = (
        abs(bounded_residual) / total_absolute_effect
        if total_absolute_effect > 0 else 0.0)
    if str(decision["account_id"]) not in runtime.index:
        raise ValueError("account differs from source-bound runtime")
    return {
        "account_id": str(decision["account_id"]),
        "seed": runtime.seed,
        "cap_lock_sha256": _sha256(lock_path),
        "cap_logit": cap,
        "named_evidence_logit": named_logit,
        "named_evidence_probability": float(expit(named_logit)),
        "named_evidence_prediction": int(expit(named_logit) >= runtime.threshold),
        "numeric_projection_residual_raw_logit": raw_residual,
        "numeric_projection_residual_bounded_logit": bounded_residual,
        "numeric_projection_residual_absolute_effect_share":
            residual_absolute_effect_share,
        "named_absolute_group_effect_sum": named_absolute_effect,
        "residual_was_clipped": raw_residual != bounded_residual,
        "bounded_logit": bounded_logit,
        "bounded_probability": bounded_probability,
        "bounded_prediction": int(bounded_probability >= runtime.threshold),
        "bounded_class_differs_from_original": bool(
            (bounded_probability >= runtime.threshold) !=
            bool(decision["prediction"])),
        "projection_changes_original_class": bool(
            (expit(named_logit) >= runtime.threshold) !=
            bool(decision["prediction"])),
        "named_source_scope": "published profile tensor and typed relation edges",
        "residual_source_scope": "published average tweet tensor via train-only 64-coordinate projection",
        "residual_capacity": "64 projection coordinates, fitted group-additive trees; output bounded but not low capacity",
    }
