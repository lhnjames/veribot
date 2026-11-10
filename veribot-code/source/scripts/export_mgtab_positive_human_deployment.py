"""Seal label-free inference inputs for the dev-locked MGTAB Human-evidence model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_positive_human_evidence"
BASE_DEPLOYMENT = ROOT / "output/v3_current_compact_deployment/mgtab_manifest.json"
OUT = FOLDER / "inference_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    lock_path = FOLDER / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    prior = json.loads(BASE_DEPLOYMENT.read_text(encoding="utf-8"))
    if (lock["test_metric_or_prediction_used_to_create_lock"] is not False or
            lock["selected"]["ratio"] != 0.05 or
            prior["status"] != "inference_only_metadata_from_audited_frozen_model" or
            prior["selection_lock_sha256"] !=
            sha256(ROOT / "output/v3_mgtab_targeted_20_concept_ab_group_tree/selection_lock.json")):
        raise ValueError("locked selection or existing source deployment differs")
    rows = []
    for seed in (0, 1, 2):
        old = prior["runs"][seed]
        model = FOLDER / f"seed{seed}/model.joblib"
        reference = ROOT / f"output/v3_mgtab_targeted_20_concept_ab_group_tree/seed{seed}/reference.json.gz"
        source_meta = ROOT / f"output/v3_current_compact_deployment/mgtab_source_seed{seed}.json"
        if (old["seed"] != seed or
                old["reference_sha256"] != sha256(reference) or
                old["source_metadata_sha256"] != sha256(source_meta)):
            raise ValueError("prior audited source or reference differs")
        rows.append({
            "seed": seed,
            "model_path": str(model.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": sha256(model),
            "reference_path": str(reference.relative_to(ROOT)).replace("\\", "/"),
            "reference_sha256": sha256(reference),
            "source_metadata_path": str(source_meta.relative_to(ROOT)).replace("\\", "/"),
            "source_metadata_sha256": sha256(source_meta),
            "split_sha256": lock["split_sha256_by_seed"][str(seed)],
            "reference_version": lock["reference_version_by_seed"][str(seed)],
            "selected_content": lock["selected_content_by_seed"][str(seed)],
            "threshold": lock["selected"]["thresholds"][str(seed)],
        })
    OUT.write_text(json.dumps({
        "dataset": "mgtab",
        "status": "inference_only_positive_human_evidence_metadata",
        "selection_lock_sha256": sha256(lock_path),
        "prior_source_deployment_sha256": sha256(BASE_DEPLOYMENT),
        "source_audit_sha256": prior["source_audit_sha256"],
        "normality_rule": "source_present * exp(-max(group_high_tail_score, group_low_tail_score))",
        "normality_direction": "nonincreasing Bot risk",
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT, sha256(OUT))


if __name__ == "__main__":
    main()
