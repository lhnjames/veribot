"""Seal the dev-locked 309-channel MGTAB model for label-free inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_ab_group_positive_human"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lock_path = FOLDER / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock["selected"] is not True or
            lock["test_metric_or_prediction_used_to_create_lock"] is not False or
            lock["model_config"] != [31, 400, 50, 10.0] or
            lock["anchor_ratio"] != 0.05 or
            lock["dev_result_sha256"] != sha256(FOLDER / "dev_only.json")):
        raise ValueError("positive Human train/dev selection differs")
    source_audit = ROOT / "research/certificates/mgtab_full_source_audit.json"
    runs = []
    for seed in (0, 1, 2):
        run_dir = FOLDER / f"seed{seed}"
        source_meta = (ROOT /
            f"output/v3_current_compact_deployment/mgtab_source_seed{seed}.json")
        entry = {
            "seed": seed,
            "model_path": str((run_dir / "model.joblib").relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": sha256(run_dir / "model.joblib"),
            "reference_path": str((run_dir / "label_blind_reference.json.gz").relative_to(ROOT)).replace("\\", "/"),
            "reference_sha256": sha256(run_dir / "label_blind_reference.json.gz"),
            "ab_contract_path": str((run_dir / "ab_direction_contract.npz").relative_to(ROOT)).replace("\\", "/"),
            "ab_contract_sha256": sha256(run_dir / "ab_direction_contract.npz"),
            "source_metadata_path": str(source_meta.relative_to(ROOT)).replace("\\", "/"),
            "source_metadata_sha256": sha256(source_meta),
            "threshold": lock["threshold_by_seed"][str(seed)],
            "split_sha256": lock["split_sha256_by_seed"][str(seed)],
            "reference_version": lock["reference_version_by_seed"][str(seed)],
        }
        runs.append(entry)
    manifest = {
        "dataset": "mgtab", "model": "ab_group_additive_positive_human_309",
        "status": "inference_only_metadata_no_test_results",
        "selection_lock_sha256": sha256(lock_path),
        "source_audit_sha256": sha256(source_audit),
        "source_features_sha256": lock["source_features_sha256"],
        "runs": runs,
    }
    path = FOLDER / "inference_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
