"""Bind the inference-only positive-Human manifest to frozen test artifacts."""

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
    manifest_path = FOLDER / "inference_manifest.json"
    lock_path = FOLDER / "selection_lock.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (manifest["status"] != "inference_only_metadata_no_test_results" or
            manifest["model"] != "ab_group_additive_positive_human_309" or
            manifest["selection_lock_sha256"] != sha256(lock_path) or
            len(manifest["runs"]) != 3):
        raise ValueError("positive Human deployment manifest differs")
    rows = []
    for seed, entry in enumerate(manifest["runs"]):
        frozen_path = FOLDER / f"seed{seed}/test_predictions.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["manifest"]
        for key in ("model_sha256", "reference_sha256", "ab_contract_sha256",
                    "reference_version", "split_sha256", "threshold"):
            if entry[key] != frozen[key]:
                raise ValueError(f"seed {seed} {key} differs from frozen model")
        if (entry["seed"] != seed or
                entry["threshold"] != lock["threshold_by_seed"][str(seed)] or
                entry["reference_version"] !=
                lock["reference_version_by_seed"][str(seed)] or
                any(sha256(ROOT / entry[key]) != entry[digest]
                    for key, digest in (
                        ("model_path", "model_sha256"),
                        ("reference_path", "reference_sha256"),
                        ("ab_contract_path", "ab_contract_sha256"),
                        ("source_metadata_path", "source_metadata_sha256")))):
            raise ValueError(f"seed {seed} inference artifacts differ")
        rows.append({"seed": seed, "model_sha256": entry["model_sha256"],
                     "threshold": entry["threshold"]})
    output = (ROOT /
        "research/certificates/mgtab_ab_positive_human_inference_manifest_audit.json")
    output.write_text(json.dumps({
        "status": "inference_only_manifest_matches_dev_locked_frozen_model",
        "manifest_sha256": sha256(manifest_path),
        "selection_lock_sha256": sha256(lock_path),
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
