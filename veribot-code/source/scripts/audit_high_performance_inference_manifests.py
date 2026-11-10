"""Bind inference-only manifests to the originally frozen prediction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "fox8": ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab_ab_group_additive": ROOT / "output/v3_mgtab_ab_group_additive_tree",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audited = {}
    for name, folder in SOURCES.items():
        manifest_path = folder / "inference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock_path = folder / "selection_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (manifest["status"] != "inference_only_metadata_no_test_results" or
                manifest["selection_lock_sha256"] != sha256(lock_path) or
                len(manifest["runs"]) != 3):
            raise ValueError(f"{name} inference manifest differs from dev lock")
        runs = []
        for seed, row in enumerate(manifest["runs"]):
            frozen_path = folder / f"seed{seed}/test_predictions.json"
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["manifest"]
            pairs = (
                ("model_sha256", "model_sha256"),
                ("reference_sha256", "reference_sha256"),
                ("reference_version", "reference_version"),
                ("split_sha256", "split_sha256"),
                ("threshold", "threshold"),
            )
            if (row["seed"] != seed or frozen["seed"] != seed or
                    any(row[ours] != frozen[theirs] for ours, theirs in pairs) or
                    row["split_sha256"] !=
                    lock["split_sha256_by_seed"][str(seed)] or
                    row["reference_version"] !=
                    lock["reference_version_by_seed"][str(seed)] or
                    row["threshold"] != (lock["threshold"] if name == "fox8"
                                         else lock["threshold_by_seed"][str(seed)]) or
                    sha256(ROOT / row["model_path"]) != row["model_sha256"] or
                    sha256(ROOT / row["reference_path"]) !=
                    row["reference_sha256"] or
                    sha256(ROOT / row["source_metadata_path"]) !=
                    row["source_metadata_sha256"]):
                raise ValueError(f"{name} seed {seed} inference differs")
            if name == "mgtab_ab_group_additive":
                if (row["ab_contract_sha256"] != frozen["ab_contract_sha256"]
                        or sha256(ROOT / row["ab_contract_path"]) !=
                        row["ab_contract_sha256"]):
                    raise ValueError("A/B contract differs from frozen model")
            runs.append({"seed": seed, "model_sha256": row["model_sha256"],
                         "reference_sha256": row["reference_sha256"],
                         "threshold": row["threshold"],
                         "frozen_manifest_sha256": sha256(frozen_path)})
        audited[name] = {
            "manifest_sha256": sha256(manifest_path),
            "selection_lock_sha256": sha256(lock_path),
            "runs": runs,
        }
    output = (ROOT /
        "research/certificates/high_performance_inference_manifests_audit.json")
    output.write_text(json.dumps({
        "status": "inference_only_manifests_match_frozen_model_artifacts",
        "models": audited,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
