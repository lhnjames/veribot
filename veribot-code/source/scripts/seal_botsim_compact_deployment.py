"""Seal the audited twelve-concept BotSim model for label-free inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    selected = root / "output/v3_computable/botsim_group_tail_12"
    full = root / "output/v3_computable/botsim_native_graph"
    deploy = root / "output/v3_current_compact_deployment"
    deploy.mkdir(parents=True, exist_ok=True)
    lock_path = selected / "selection_lock.json"
    verification_path = selected / "independent_verification.json"
    source_audit_path = root / "research/certificates/botsim24_full_source_audit.json"
    lock, verification, audit = map(read, (lock_path, verification_path, source_audit_path))
    if (lock["dataset"] != "botsim" or verification["dataset"] != "botsim"
            or audit["status"] != "verified_full_population_against_raw_comments"
            or len(verification["runs"]) != 3):
        raise ValueError("BotSim model or source verification is incomplete")
    runs = []
    for seed, checked in enumerate(verification["runs"]):
        selected_path = selected / f"seed{seed}/report.json"
        full_path = full / f"seed{seed}/report.json"
        selected_report, full_report = read(selected_path), read(full_path)
        chosen, source = selected_report["manifest"], full_report["manifest"]
        if (checked["seed"] != seed
                or checked["status"] != "all_source_to_concept_to_decision_cards_replayed"
                or checked["model_report_sha256"] != sha256_file(selected_path)
                or chosen["dataset"] != "botsim" or chosen["seed"] != seed
                or chosen["concept_count"] != 12
                or chosen["selection_lock_sha256"] != sha256_file(lock_path)
                or chosen["split_sha256"] != source["split_sha256"]
                or chosen["reference_version"] != source["reference_version"]
                or source["reference_artifact_sha256"] !=
                sha256_file(full / f"seed{seed}/reference.json.gz")):
            raise ValueError(f"BotSim seed{seed} frozen model differs from audit")
        source_files = {path: digest for path, digest in source["files_sha256"].items()
                        if "label" not in Path(path).name.lower()}
        if len(source_files) != len(source["files_sha256"]) - 1:
            raise ValueError("expected exactly one label file to remove")
        source_metadata = {
            "manifest": {"dataset": "botsim", "files_sha256": source_files},
            "source_fact_names": list(full_report["reasoner"]["directions"]),
        }
        source_metadata_path = deploy / f"botsim_source_seed{seed}.json"
        write(source_metadata_path, source_metadata)
        selected_fields = ("dataset", "seed", "split_sha256", "reference_version",
                           "concept_version", "concept_count", "concept_names", "threshold")
        runtime_metadata = {
            "selected": {"manifest": {key: chosen[key] for key in selected_fields},
                         "reasoner": selected_report["reasoner"]},
            "full": {"manifest": {key: source[key] for key in
                                  ("split_sha256", "reference_version",
                                   "reference_artifact_sha256")},
                     "reasoner": {"groups": full_report["reasoner"]["groups"]}},
        }
        runtime_metadata_path = deploy / f"botsim_runtime_seed{seed}.json"
        write(runtime_metadata_path, runtime_metadata)
        runs.append({
            "seed": seed,
            "selected_report_sha256": sha256_file(selected_path),
            "full_report_sha256": sha256_file(full_path),
            "reference_sha256": sha256_file(full / f"seed{seed}/reference.json.gz"),
            "source_metadata_sha256": sha256_file(source_metadata_path),
            "runtime_metadata_sha256": sha256_file(runtime_metadata_path),
        })
    manifest_path = deploy / "botsim_manifest.json"
    write(manifest_path, {
        "dataset": "botsim", "concept_count": 12,
        "selection_lock_sha256": sha256_file(lock_path),
        "source_audit_sha256": sha256_file(source_audit_path),
        "independent_verification_sha256": sha256_file(verification_path),
        "runs": runs,
        "status": "inference_only_metadata_from_audited_frozen_model",
    })
    certificate_path = root / "research/certificates/botsim_compact_deployment_manifest_audit.json"
    write(certificate_path, {"dataset": "botsim", "manifest_sha256": sha256_file(manifest_path),
                             "status": "deployment_manifest_bound_to_source_and_model_audits"})
    print(json.dumps({"manifest": str(manifest_path), "runs": len(runs)}))


if __name__ == "__main__":
    main()
