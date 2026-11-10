"""Check source-bound predictions and rejection of altered source files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from predict_compact_from_verified_sources import CONFIG, certificate_binding
from verifiable_bot.mgtab_profile_dictionary import (
    PAPER_URL, published_profile_description)


SAMPLES = {
    "botsim": [(0, "irfvivs", "botsim_irfvivs.json")],
    "fox8": [(0, "830915719", "fox8_830915719.json")],
    "mgtab": [(0, "2272", "mgtab_2272.json"),
              (1, "8567", "mgtab_seed1_8567.json"),
              (2, "1876", "mgtab_seed2_1876.json")],
}


def audit(root: Path) -> dict:
    rows = []
    for dataset, samples in SAMPLES.items():
        config = CONFIG[dataset]
        for seed, account_id, filename in samples:
            path = root / "output/v3_source_bound_inference" / filename
            result = json.loads(path.read_text(encoding="utf-8"))
            decisions = result["decisions"]
            if (result["dataset"] != dataset or result["seed"] != seed
                    or result["classification_labels_accessed"] is not False
                    or result["caller_supplied_fact_matrix_accepted"] is not False
                    or len(decisions) != 1 or decisions[0]["account_id"] != account_id
                    or result["max_raw_to_released_fact_difference"] > 1e-5):
                raise ValueError("source-bound sample or source rebuild differs")
            certificate = certificate_binding(root, dataset, config)
            if certificate != result["source_binding"]:
                raise ValueError("source-binding hashes differ from current files")
            selected = root / config["selected"] / f"seed{seed}"
            full = root / config["full"] / f"seed{seed}"
            current_model = {
                "selected_report_sha256": sha256_file(selected / "report.json"),
                "full_report_sha256": sha256_file(full / "report.json"),
                "reference_sha256": sha256_file(full / "reference.json.gz"),
            }
            if dataset == "mgtab":
                current_model["train_only_projection_sha256"] = sha256_file(
                    full / "projection.npz")
            if dataset == "fox8":
                current_model["prespecified_groups_sha256"] = sha256_file(
                    root / config["groups"])
            if current_model != result["model_binding"]:
                raise ValueError("frozen model binding differs from current files")
            saved = json.loads((selected / "test_predictions.json")
                               .read_text(encoding="utf-8"))
            match = [row for row in saved["predictions"]
                     if row["account_id"] == account_id]
            if (len(match) != 1 or abs(match[0]["probability"] -
                                       decisions[0]["probability"]) > 1e-10):
                raise ValueError("source-bound probability differs from frozen test")
            terms = decisions[0]["terms"]
            for term in terms:
                query = term.get("recompute_query")
                if term["source_fact_name"] is None:
                    if query is not None:
                        raise ValueError("unavailable term has a source query")
                elif (not isinstance(query, dict) or
                      query["fact_name"] != term["source_fact_name"] or
                      query["account_id"] != account_id or
                      not query["source_ids"]):
                    raise ValueError("available term lacks a source query")
                if dataset == "mgtab" and term["source_fact_name"] is not None:
                    expected = published_profile_description(term["source_fact_name"])
                    if term.get("published_profile_semantics") != expected:
                        raise ValueError("MGTAB official profile dimension semantics differ")
            if dataset == "mgtab":
                dictionary_path = root / "research/certificates/mgtab_published_profile_dictionary_audit.json"
                if result.get("published_profile_dictionary_binding") != {
                        "audit_sha256": sha256_file(dictionary_path),
                        "table_a5_reference": PAPER_URL,
                        "source_level": "published_normalized_profile_feature"}:
                    raise ValueError("MGTAB profile dictionary binding differs")
            rows.append({"dataset": dataset, "seed": seed,
                         "account_id": account_id,
                         "probability_difference": abs(
                             match[0]["probability"] - decisions[0]["probability"]),
                         "source_located_terms": sum(
                             term["recompute_query"] is not None for term in terms),
                         "terms": len(terms), "sample_sha256": sha256_file(path)})
    tamper = []
    for dataset, source_name in (("botsim", "account"),
                                 ("fox8", "native"),
                                 ("mgtab", "structural")):
        config = CONFIG[dataset].copy()
        original = root / config[source_name]
        with TemporaryDirectory() as directory:
            altered = Path(directory) / original.name
            shutil.copyfile(original, altered)
            with altered.open("ab") as stream:
                stream.write(b"\nALTERED_SOURCE_FOR_REJECTION_CHECK\n")
            config[source_name] = str(altered)
            try:
                certificate_binding(root, dataset, config)
            except ValueError as error:
                if "differs from independent audit" not in str(error):
                    raise
                tamper.append({"dataset": dataset,
                               "altered_source_kind": source_name,
                               "rejected": True})
            else:
                raise AssertionError("altered source file was accepted")
    return {"scope": "source_bound_inference_samples_and_source_tamper_rejection",
            "saved_test_labels_used_for_model_or_threshold": False,
            "samples": rows, "tamper_checks": tamper}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.project_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(result["samples"]),
                      "tamper_rejected": len(result["tamper_checks"])}))


if __name__ == "__main__":
    main()
