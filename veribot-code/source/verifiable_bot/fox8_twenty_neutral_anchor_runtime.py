"""Raw-source Fox8 20-concept inference with dev-locked neutral anchor."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime, _sha256


class Fox8TwentyNeutralAnchorRuntime(Fox8TargetedTwentyRuntime):
    def __init__(self, root: Path, seed: int):
        super().__init__(root, seed)
        folder = self.root / "output/v3_fox8_twenty_neutral_anchor"
        path = folder / "selection_lock.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        row = lock["runs"][seed]
        model_path = folder / f"seed{seed}/model.joblib"
        contract_path = folder / f"seed{seed}/direction_contract.npz"
        if (lock["scope"] !=
                "Fox8_20_concept_ab_neutral_anchor_selected_on_dev_before_test" or
                lock["test_predictions_or_metrics_used_for_selection"] is not False or
                row["seed"] != seed or
                row["split_sha256"] != self.record["split_sha256"] or
                row["reference_version"] != self.reference.version or
                row["model_path"] != str(model_path) or
                row["direction_contract_path"] != str(contract_path) or
                _sha256(model_path) != row["model_sha256"] or
                _sha256(contract_path) != row["direction_contract_sha256"]):
            raise ValueError("Fox8 neutral deployment binding differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        with np.load(contract_path, allow_pickle=False) as z:
            signs = z["monotone_constraints"].astype(int).tolist()
            groups = z["group_names"].astype(str).tolist()
        if (self.model.get_params()["monotone_constraints"] != signs or
                groups != list(self.groups)):
            raise ValueError("Fox8 neutral direction or source groups differ")
        zero = np.zeros((1, 40), dtype=float)
        threshold = float(self.model.predict_proba(zero)[0, 1])
        if abs(threshold-row["neutral_threshold"]) > 1e-12:
            raise ValueError("Fox8 neutral threshold changed")
        self.threshold = threshold
        self.neutral_lock_sha256 = _sha256(path)
        self.neutral_model_sha256 = row["model_sha256"]

    def predict_many(self, account_ids: list[str]) -> list[dict]:
        cards = super().predict_many(account_ids)
        for card in cards:
            shift = card["logit"] - card["zero_evidence_logit"]
            if (abs(card["zero_evidence_probability"] - self.threshold) > 1e-12 or
                    abs(shift) <= 1e-10 or
                    bool(card["prediction"]) != bool(shift > 0)):
                raise ValueError("Fox8 class lacks signed source evidence")
            card["model_variant"] = "20_concept_neutral_anchor_ab_group_tree"
            card["evidence_logit_shift_from_neutral"] = shift
            card["evidence_strengthens_predicted_class"] = True
            card["model_binding"]["model_sha256"] = self.neutral_model_sha256
            card["model_binding"]["neutral_selection_lock_sha256"] = (
                self.neutral_lock_sha256)
        return cards
