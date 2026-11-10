"""Summarize classification effect of restoring omitted published support edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_FROZEN = ROOT / "research/certificates/twibot20_group_tree_support_edge_response_dev.json"
RETRAIN = ROOT / "research/certificates/twibot20_corrected_sample_graph_ab_dev.json"
SOURCE_AUDIT = ROOT / "research/certificates/twibot20_support_edge_omission.json"
OUT = ROOT / "research/certificates/twibot20_support_correction_utility.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frozen = json.loads(OLD_FROZEN.read_text(encoding="utf-8"))
    retrain = json.loads(RETRAIN.read_text(encoding="utf-8"))
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if (frozen["support_edge_audit_sha256"] != sha256(SOURCE_AUDIT)
            or retrain["support_edge_audit_sha256"] != sha256(SOURCE_AUDIT)
            or frozen["old_fact_cache_sha256"] != retrain["old_fact_cache_sha256"]
            or frozen["corrected_fact_cache_sha256"] != retrain["corrected_fact_cache_sha256"]
            or frozen["old_reference_sha256"] != retrain["old_reference_sha256"]
            or frozen["candidate_test_evaluated"]
            or retrain["candidate_test_evaluated"]):
        raise ValueError("Old-model and retrained source-correction contracts differ")
    flips = [r for r in frozen["changed_dev_rows"]
             if r["old_class"] != r["new_class_old_frozen_model"]]
    improvements = [r for r in flips
                    if r["new_class_old_frozen_model"] == r["evaluation_label"]]
    regressions = [r for r in flips
                   if r["old_class"] == r["evaluation_label"]]
    if (len(flips) != frozen["dev_class_changed"]
            or len(improvements) - len(regressions) !=
            frozen["dev_old_errors"] - frozen["dev_new_errors_under_old_model"]):
        raise ValueError("Frozen correction flip and error totals disagree")
    rows = []
    for item in retrain["rows"]:
        old = item["old_labeled_only_graph_dev"]
        corrected = item["all_provided_sample_graph_dev"]
        rows.append({
            "ab_seed": item["ab_seed"],
            "dev_accounts": old["n"],
            "old_dev_errors": old["errors"],
            "corrected_retrained_dev_errors": corrected["errors"],
            "corrected_minus_old_dev_errors": corrected["errors"] - old["errors"],
            "old_dev_macro_f1": old["macro_f1"],
            "corrected_retrained_dev_macro_f1": corrected["macro_f1"],
            "corrected_minus_old_dev_macro_f1": corrected["macro_f1"] - old["macro_f1"],
        })
    if (len(rows) != 3 or [r["ab_seed"] for r in rows] != [0, 1, 2]
            or frozen["source_changed_dev_accounts"] != 325
            or frozen["dev_logit_changed"] != 214
            or (len(flips), len(improvements), len(regressions)) != (7, 2, 5)
            or [r["corrected_minus_old_dev_errors"] for r in rows] != [3, 3, 10]):
        raise ValueError("Published support correction population differs")
    report = {
        "scope": "TwiBot-20 official dev; restore previously omitted released support-row neighbor declarations in source graph",
        "source_operation": "correct preprocessing omission of support.json sampled relations; old frozen model input correction and separate same-protocol retraining are distinct estimands",
        "not_natural_user_data_error": True,
        "candidate_test_evaluated": False,
        "old_frozen_source_response_sha256": sha256(OLD_FROZEN),
        "corrected_retraining_report_sha256": sha256(RETRAIN),
        "support_edge_audit_sha256": sha256(SOURCE_AUDIT),
        "source_changed_dev_accounts": frozen["source_changed_dev_accounts"],
        "old_frozen_dev_logit_changed": frozen["dev_logit_changed"],
        "old_frozen_dev_class_changed": len(flips),
        "old_frozen_dev_class_improvements": len(improvements),
        "old_frozen_dev_class_regressions": len(regressions),
        "old_frozen_dev_errors_before": frozen["dev_old_errors"],
        "old_frozen_dev_errors_after_corrected_input": frozen["dev_new_errors_under_old_model"],
        "same_protocol_retraining_by_ab_seed": rows,
        "interpretation": "The released graph source was incompletely ingested. Correcting the input restores source integrity, but neither feeding it to the old frozen model nor retraining the specified group tree reduced dev errors in these runs. Do not claim correction utility or causal effect on labels from this audit.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
