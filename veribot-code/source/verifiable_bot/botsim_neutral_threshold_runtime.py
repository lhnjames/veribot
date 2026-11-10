"""Source-bound BotSim inference with the dev-locked neutral threshold."""

from __future__ import annotations

import json
from pathlib import Path

from scipy.special import expit

from .botsim_literal_zero_runtime import BotSimObservedMeanRuntime
from .compact_evidence_runtime import _sha256


class BotSimNeutralThresholdRuntime(BotSimObservedMeanRuntime):
    def __init__(self, root: Path, seed: int):
        super().__init__(root, seed)
        path = self.root / "output/v3_botsim_neutral_threshold/selection_lock.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        row = lock["runs"][seed]
        if (lock["scope"] !=
                "BotSim_12_concept_existing_reasoner_neutral_threshold_dev_locked" or
                lock["test_predictions_or_metrics_used_for_selection"] is not False or
                row["seed"] != seed or
                row["reasoner_sha256"] !=
                self.source_binding["reasoner_sha256"]):
            raise ValueError("BotSim neutral selection differs")
        threshold = float(expit(self.reasoner.intercept))
        if abs(threshold - row["neutral_threshold"]) > 1e-12:
            raise ValueError("BotSim frozen neutral threshold changed")
        self.threshold = threshold
        self.neutral_lock_sha256 = _sha256(path)

    def predict_many(self, account_ids: list[str]) -> list[dict]:
        rows = super().predict_many(account_ids)
        for row in rows:
            shift = float(row["logit"] - row["intercept"])
            if abs(shift) <= 1e-10:
                raise ValueError("BotSim account has no signed evidence movement")
            if bool(row["prediction"]) != bool(shift > 0):
                raise ValueError("BotSim class differs from signed evidence")
            row["model_variant"] = "12_concept_neutral_threshold"
            row["neutral_lock_sha256"] = self.neutral_lock_sha256
            row["evidence_logit_shift_from_neutral"] = shift
            row["evidence_strengthens_predicted_class"] = True
        return rows
