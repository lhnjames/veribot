"""Audit the learned input projection for the missing TwiBot-20 HGT coordinate.

The checkpoint and current feature builder disagree by one numeric coordinate.
This script reads only metadata and frozen weights, not account labels or outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "source/hgt_import/checkpoints/clean_hgt_best.pt"
CONTRACT = ROOT / "research/certificates/twibot20_hgt_checkpoint_input_contract.json"
BUILDER = ROOT / "source/hgt_import/build_features_and_graph.py"
MODEL = ROOT / "source/hgt_import/code/hgt_model.py"
OUT = ROOT / "research/certificates/twibot20_hgt_unknown_numeric_projection.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["checkpoint_sha256"] != digest(CHECKPOINT) or
            contract["checkpoint_user_width"] != 2054 or
            contract["current_builder_total_width"] != 2053 or
            contract["current_builder_numeric_width"] != 5 or
            contract["saved_build_log_numeric_width"] != 6 or
            contract["saved_graph_exists"] is not False):
        raise ValueError("Saved HGT input-contract premises changed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["in_channels"] != {"user": 2054}:
        raise ValueError("Frozen HGT input-width metadata changed")
    weight = checkpoint["model_state"]["lin_dict.user.weight"].detach().cpu().numpy()
    if weight.shape != (128, 2054) or not np.isfinite(weight).all():
        raise ValueError("Frozen HGT first input projection changed")
    norms = np.linalg.norm(weight, axis=0)
    numeric = norms[-6:]
    unknown = weight[:, -1]
    report = {
        "status": "unknown_sixth_numeric_coordinate_has_nonzero_learned_projection",
        "scope": "frozen_TwiBot20_HGT_checkpoint_first_linear_projection_only",
        "checkpoint_sha256": digest(CHECKPOINT),
        "prior_input_contract_sha256": digest(CONTRACT),
        "current_builder_sha256": digest(BUILDER),
        "hgt_model_definition_sha256": digest(MODEL),
        "checkpoint_user_input_width": int(weight.shape[1]),
        "current_builder_input_width": 2053,
        "text_coordinates": 2048,
        "checkpoint_numeric_coordinates": 6,
        "current_builder_numeric_coordinates": 5,
        "unknown_checkpoint_coordinate_zero_based": 2053,
        "numeric_column_first_linear_weight_l2": [float(x) for x in numeric],
        "unknown_coordinate_nonzero_projection_weights": int(np.count_nonzero(unknown)),
        "unknown_coordinate_weight_l2": float(numeric[-1]),
        "unknown_coordinate_weight_max_abs": float(np.max(np.abs(unknown))),
        "unknown_coordinate_weight_norm_rank_among_all_2054_descending":
            int(np.count_nonzero(norms > numeric[-1]) + 1),
        "unit_change_in_unknown_standardized_coordinate_pre_activation_l2":
            float(numeric[-1]),
        "unknown_raw_feature_identity_recovered": False,
        "original_2054_column_graph_available": False,
        "semantic_or_final_class_dependence_claimed": False,
        "account_labels_predictions_or_test_metrics_read": False,
        "interpretation": (
            "The saved model has 128 nonzero first-layer weights on the missing "
            "coordinate, so substituting an arbitrary sixth feature or silently "
            "padding zero changes the input contract. Weight norm establishes a "
            "pre-activation path, not the original feature identity or a measured "
            "final-class effect. Exact source-to-decision claims for this checkpoint "
            "need the original sixth-feature definition and graph tensor."),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print({"unknown_weight_l2": report["unknown_coordinate_weight_l2"],
           "nonzero_weights": report["unknown_coordinate_nonzero_projection_weights"],
           "norm_rank": report[
               "unknown_coordinate_weight_norm_rank_among_all_2054_descending"]})
    print(OUT)


if __name__ == "__main__":
    main()
