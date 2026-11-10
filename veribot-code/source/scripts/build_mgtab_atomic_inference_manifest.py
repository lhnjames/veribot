"""Seal MGTAB atomic models for inference without test results or labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {
    "full_atomic": "output/v3_mgtab_full_atomic_evidence_lightgbm",
    "ab_group_additive": "output/v3_mgtab_ab_group_additive_tree",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source_audit = ROOT / "research/certificates/mgtab_full_source_audit.json"
    for variant, relative in FOLDERS.items():
        folder = ROOT / relative
        lock_path = folder / "selection_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (lock["selected_config"] != [31, 400, 50, 10.0] or
                lock["test_metric_or_prediction_used_to_create_lock"] is not False):
            raise ValueError("MGTAB selection was not sealed on train/dev")
        runs = []
        for seed in (0, 1, 2):
            run_dir = folder / f"seed{seed}"
            source_meta = (ROOT /
                f"output/v3_current_compact_deployment/mgtab_source_seed{seed}.json")
            entry = {
                "seed": seed,
                "model_path": str((run_dir / "model.joblib").relative_to(ROOT)).replace("\\", "/"),
                "model_sha256": sha256(run_dir / "model.joblib"),
                "reference_path": str((run_dir / "label_blind_reference.json.gz").relative_to(ROOT)).replace("\\", "/"),
                "reference_sha256": sha256(run_dir / "label_blind_reference.json.gz"),
                "reference_version": lock["reference_version_by_seed"][str(seed)],
                "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                "threshold": (lock["threshold_by_seed"][str(seed)]
                              if variant == "ab_group_additive" else lock["threshold"]),
                "source_metadata_path": str(source_meta.relative_to(ROOT)).replace("\\", "/"),
                "source_metadata_sha256": sha256(source_meta),
            }
            if variant == "ab_group_additive":
                contract = run_dir / "ab_direction_contract.npz"
                entry["ab_contract_path"] = str(contract.relative_to(ROOT)).replace("\\", "/")
                entry["ab_contract_sha256"] = sha256(contract)
            runs.append(entry)
        payload = {
            "dataset": "mgtab", "variant": variant,
            "status": "inference_only_metadata_no_test_results",
            "selection_lock_sha256": sha256(lock_path),
            "source_audit_sha256": sha256(source_audit),
            "source_features_sha256": lock["source_features_sha256"],
            "runs": runs,
        }
        path = folder / "inference_manifest.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
