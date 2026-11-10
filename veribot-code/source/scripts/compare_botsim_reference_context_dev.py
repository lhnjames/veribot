"""Paired BotSim dev comparison: activity-stratified versus pooled reference."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from gate1_group_additive_schema_v2_dev import _run


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
EXISTING = ROOT / "output/gate1_group_additive_schema_v2/dev_only.json"
OUT = ROOT / "research/certificates/botsim_context_pooled_dev.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(row: dict) -> dict:
    return {
        "reference_version": row["reference_version"],
        "split_sha256": row["split_sha256"],
        "evidence_group_gam_at_half": row["evidence_group_gam"],
        "crossfit_group_gam_at_half": row["crossfit_group_gam_at_half"],
        "evidence_group_gam_dev_calibrated": row[
            "evidence_group_gam_dev_calibrated"],
        "crossfit_group_gam_dev_calibrated": row[
            "crossfit_group_gam_dev_calibrated"],
    }


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    existing = json.loads(EXISTING.read_text(encoding="utf-8"))
    if (schema["version"] != "candidate-v2" or
            existing["schema_sha256"] != digest(SCHEMA) or
            existing["test_predictions_or_metrics_accessed"] is not False):
        raise ValueError("existing conditioned experiment is not the expected dev run")
    spec = deepcopy(schema["schema"]["botsim"])
    if spec["reference_context"]["rule"] != \
            "unlabeled_training_tertiles_with_global_fallback":
        raise ValueError("BotSim original context rule differs")
    spec["reference_context"] = {
        "fact": None,
        "meaning": "all observed training accounts form one label-free reference",
        "rule": "pooled_unlabeled_training_reference",
        "minimum_observed_accounts_per_stratum": 30,
    }
    pooled = _run("botsim", spec)
    conditioned = existing["runs"]["botsim"]
    if len(pooled) != len(conditioned) or len(pooled) != 3:
        raise ValueError("expected three paired seeds")
    paired = []
    for old, new in zip(conditioned, pooled, strict=True):
        if (old["seed"] != new["seed"] or
                old["split_sha256"] != new["split_sha256"] or
                old["A_direction_accounts"] != new["A_direction_accounts"] or
                old["B_fit_accounts"] != new["B_fit_accounts"] or
                old["dev_accounts"] != new["dev_accounts"]):
            raise ValueError("paired train/dev population changed")
        paired.append({
            "seed": old["seed"],
            "dev_accounts": old["dev_accounts"],
            "A_direction_accounts": old["A_direction_accounts"],
            "B_fit_accounts": old["B_fit_accounts"],
            "activity_conditioned": summary(old),
            "pooled_all_training": summary(new),
            "at_half_error_difference_conditioned_minus_pooled":
                old["evidence_group_gam"]["errors"] -
                new["evidence_group_gam"]["errors"],
            "at_half_macro_f1_difference_conditioned_minus_pooled":
                old["evidence_group_gam"]["macro_f1"] -
                new["evidence_group_gam"]["macro_f1"],
        })
    result = {
        "scope": "BotSim-24 schema-v2 evidence model, paired train/dev only",
        "same_evidence_vocabulary_split_and_A_B_reasoner": True,
        "only_changed_design_choice": "label-free reference context rule",
        "training_and_reference_are_refit_for_each_rule": True,
        "dev_threshold_selection_reuses_dev_and_is_exploratory": True,
        "test_predictions_or_metrics_accessed": False,
        "schema_sha256": digest(SCHEMA),
        "conditioned_existing_sha256": digest(EXISTING),
        "paired": paired,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
