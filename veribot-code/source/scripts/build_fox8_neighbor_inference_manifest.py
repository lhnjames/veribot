"""Seal the 74-fact Fox8 model using train/dev lock and source artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_source_path import resolve_fox8_source_dir


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lock_path = FOLDER / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock["selected"]["variant"] != "raw_neighbor_74" or
            lock["config"] != [15, 200, 30, 10.0] or
            lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("Fox8 train/dev selection differs")
    source_audit = ROOT / "research/certificates/fox8_neighbor_volume_full_raw_source.json"
    source = json.loads(source_audit.read_text(encoding="utf-8"))
    data_dir = resolve_fox8_source_dir(ROOT)
    if sha256(data_dir / "features.csv") != source["published_feature_sha256"]:
        raise ValueError("Fox8 published feature table differs")
    runs = []
    for seed in (0, 1, 2):
        run_dir = FOLDER / f"seed{seed}"
        source_meta = (ROOT /
            f"output/v3_current_compact_deployment/fox8_source_seed{seed}.json")
        entry = {
            "seed": seed,
            "model_path": str((run_dir / "model.joblib").relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": sha256(run_dir / "model.joblib"),
            "reference_path": str((run_dir / "label_blind_reference.json.gz").relative_to(ROOT)).replace("\\", "/"),
            "reference_sha256": sha256(run_dir / "label_blind_reference.json.gz"),
            "reference_version": lock["reference_version_by_seed"][str(seed)],
            "split_sha256": lock["split_sha256_by_seed"][str(seed)],
            "threshold": lock["threshold"],
            "source_metadata_path": str(source_meta.relative_to(ROOT)).replace("\\", "/"),
            "source_metadata_sha256": sha256(source_meta),
        }
        runs.append(entry)
    payload = {
        "dataset": "fox8", "model": "raw_neighbor_74",
        "status": "inference_only_metadata_no_test_results",
        "selection_lock_sha256": sha256(lock_path),
        "source_audit_sha256": sha256(source_audit),
        "published_feature_sha256": source["published_feature_sha256"],
        "native_event_index_sha256": source["event_index_sha256"],
        "runs": runs,
    }
    path = FOLDER / "inference_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
