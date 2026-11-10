"""Seal inference-only metadata for the audited compact Fox8/MGTAB models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


CONFIG = {
    "fox8": {
        "folder": "output/v3_fox8_targeted_neighbor_20_concepts",
        "lock": "selection_lock_ab_group_tree.json",
        "report": "fixed_test_ab_group_tree.json",
        "replay": "research/certificates/fox8_targeted_20_concept_ab_tree_replay.json",
        "source_audit": "research/certificates/fox8_neighbor_volume_full_raw_source.json",
        "model": "ab_group_tree.joblib",
        "reference": "label_blind_reference.json.gz",
        "full_source_dir": "output/v3_fox8_native_spline_l2/selected",
        "replay_status": "all_accounts_concepts_group_contributions_and_masks_verified",
    },
    "mgtab": {
        "folder": "output/v3_mgtab_targeted_20_concept_ab_group_tree",
        "lock": "selection_lock.json",
        "report": "fixed_test_report.json",
        "replay": "research/certificates/mgtab_targeted_20_concept_ab_group_tree_replay.json",
        "source_audit": "research/certificates/mgtab_full_source_audit.json",
        "model": "model.joblib",
        "reference": "reference.json.gz",
        "full_source_dir": "output/v3_mgtab_ab_spline_content/selected",
        "replay_status": "all_test_facts_concepts_monotonicity_and_group_masks_verified",
    },
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_current_compact_deployment"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, setting in CONFIG.items():
        model_dir = root / setting["folder"]
        lock_path = model_dir / setting["lock"]
        report_path = model_dir / setting["report"]
        replay_path = root / setting["replay"]
        source_path = root / setting["source_audit"]
        lock, report, replay, source = map(read, (
            lock_path, report_path, replay_path, source_path))
        if (report["selection_lock_sha256"] != sha256_file(lock_path)
                or replay["fixed_test_report_sha256"] != sha256_file(report_path)
                or replay["status"] != setting["replay_status"]
                or report["concept_count"] != 20
                or report["source_group_count"] != 10):
            raise ValueError("audited fixed-model chain differs")
        if dataset == "fox8":
            if report["full_raw_source_audit_sha256"] != sha256_file(source_path):
                raise ValueError("Fox8 raw source audit differs")
        elif (source["status"] != "verified_all_published_tensor_derived_fact_cells"
              or report["source_features_sha256"] !=
              source["source_sha256"]["features.pt"]):
            raise ValueError("MGTAB published source audit differs")
        runs = []
        for seed in (0, 1, 2):
            audited = report["runs"][seed]
            run_dir = model_dir / f"seed{seed}"
            full_source_report_path = (root / setting["full_source_dir"] /
                                       f"seed{seed}/report.json")
            full_source_report = read(full_source_report_path)
            original_manifest = full_source_report["manifest"]
            if (original_manifest["dataset"] != dataset
                    or original_manifest["split_sha256"] !=
                    lock["split_sha256_by_seed"][str(seed)]):
                raise ValueError("source population does not match locked split")
            files = {path: digest for path, digest in
                     original_manifest["files_sha256"].items()
                     if "label" not in Path(path).name.lower()}
            if not files or len(files) == len(original_manifest["files_sha256"]):
                raise ValueError("source metadata must exclude account label paths")
            source_manifest = {key: original_manifest[key] for key in (
                ("dataset", "native_facts_sha256") if dataset == "fox8" else
                ("dataset", "projection_artifact_sha256", "projection_fingerprint"))}
            source_manifest["files_sha256"] = files
            source_metadata = {"manifest": source_manifest,
                               "source_fact_names": list(
                                   full_source_report["reasoner"]["directions"])}
            metadata_path = folder / f"{dataset}_source_seed{seed}.json"
            metadata_path.write_text(json.dumps(source_metadata, indent=2) + "\n",
                                     encoding="utf-8")
            model_hash = sha256_file(run_dir / setting["model"])
            reference_hash = sha256_file(run_dir / setting["reference"])
            if (audited["seed"] != seed
                    or audited["model_sha256"] != model_hash
                    or audited["reference_sha256"] != reference_hash):
                raise ValueError("sealed weights differ from audited fixed model")
            runs.append({
                "seed": seed, "model_sha256": model_hash,
                "reference_sha256": reference_hash,
                "threshold": lock["threshold_by_seed"][str(seed)],
                "reference_version": lock["reference_version_by_seed"][str(seed)],
                "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                "source_metadata_sha256": sha256_file(metadata_path),
                "source_model_report_sha256": sha256_file(full_source_report_path),
            })
        sealed = {
            "dataset": dataset,
            "selection_lock_sha256": sha256_file(lock_path),
            "source_audit_sha256": sha256_file(source_path),
            "replay_certificate_sha256": sha256_file(replay_path),
            "fixed_model_audit_sha256": sha256_file(report_path),
            "concept_count": 20, "source_group_count": 10,
            "runs": runs,
            "status": "inference_only_metadata_from_audited_frozen_model",
        }
        path = folder / f"{dataset}_manifest.json"
        path.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
        if any(key in sealed for key in ("test", "label", "predictions")):
            raise ValueError("inference manifest contains evaluation fields")
        rows.append({"dataset": dataset, "manifest_sha256": sha256_file(path),
                     "manifest_file": str(path.relative_to(root)).replace("\\", "/")})
    audit = {"scope": "inference_only_compact_model_metadata",
             "runs": rows,
             "status": "deployment_manifests_bound_to_audited_frozen_models"}
    path = root / "research/certificates/current_compact_deployment_manifest_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
