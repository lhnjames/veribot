"""Source-bound inference for the dev-locked neutral-anchor MGTAB model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .mgtab_full_atomic_runtime import MGTABFullAtomicRuntime, _sha256


class MGTABNeutralAnchorRuntime(MGTABFullAtomicRuntime):
    """Use the verified 150 fact source path with a neutral-threshold model."""

    def __init__(self, root: Path, seed: int):
        super().__init__(root, seed, variant="ab_group_additive")
        folder = self.root / "output/v3_mgtab_150_neutral_anchor"
        lock = json.loads((folder / "selection_lock.json")
                          .read_text(encoding="utf-8"))
        run = lock["runs"][seed]
        model_path = folder / f"seed{seed}/model.joblib"
        contract_path = folder / f"seed{seed}/direction_contract.npz"
        if (lock["scope"] !=
                "MGTAB_150_fact_neutral_anchor_selected_on_dev_before_test" or
                lock["test_predictions_or_metrics_used_for_selection"] is not False or
                lock["published_source_features_sha256"] !=
                self.lock["source_features_sha256"] or
                run["split_sha256"] !=
                self.lock["split_sha256_by_seed"][str(seed)] or
                run["reference_version"] != self.reference.version or
                run["model_path"] != str(model_path) or
                run["direction_contract_path"] != str(contract_path) or
                _sha256(model_path) != run["model_sha256"] or
                _sha256(contract_path) != run["direction_contract_sha256"]):
            raise ValueError("Neutral-anchor deployment binding differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        with np.load(contract_path, allow_pickle=False) as contract:
            signs = contract["monotone_constraints"].astype(int).tolist()
            groups = contract["group_names"].astype(str).tolist()
        if (self.model.get_params()["monotone_constraints"] != signs or
                groups != list(self.group_fact_indices)):
            raise ValueError("Neutral-anchor direction or groups differ")
        zero = np.zeros((1, 2 * len(self.names)), dtype=float)
        threshold = float(self.model.predict_proba(zero)[0, 1])
        if abs(threshold - run["zero_evidence_threshold"]) > 1e-12:
            raise ValueError("Neutral-anchor score differs")
        self.threshold = threshold
        self.neutral_anchor_lock_sha256 = _sha256(folder / "selection_lock.json")
        self.neutral_anchor_model_sha256 = run["model_sha256"]

    def predict(self, account_id: str) -> dict:
        card = super().predict(account_id)
        card["model_variant"] = "150_fact_neutral_anchor_ab_group_additive"
        card["neutral_anchor_probability"] = self.threshold
        card["neutral_anchor_lock_sha256"] = self.neutral_anchor_lock_sha256
        card["neutral_anchor_model_sha256"] = self.neutral_anchor_model_sha256
        if abs(card["all_evidence_zero_probability"] - self.threshold) > 1e-12:
            raise ValueError("Zero-evidence account score differs from threshold")
        if abs(card["probability"] - self.threshold) <= 1e-10:
            raise ValueError("Account has no signed evidence movement")
        certificate = card["source_group_decision_certificate"]
        if not certificate["evidence_strengthens_predicted_class"]:
            raise ValueError("Predicted class lacks signed source evidence")
        return card
