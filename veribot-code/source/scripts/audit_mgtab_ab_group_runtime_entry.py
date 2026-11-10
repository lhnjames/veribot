"""Sample label-free source-bound MGTAB A/B group-additive predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original = torch.load
    original_read_text = Path.read_text
    original_open = Path.open
    forbidden_reads = []

    def protected(path) -> bool:
        name = Path(path).name.lower()
        return (name in {"test_predictions.json", "fixed_test_report.json",
                         "report.json", "labels_bot.pt", "labels.pt",
                         "source_group_decision_certificates.jsonl"} or
                "test_evidence" in name)

    def no_test_text(path, *args, **kwargs):
        if protected(path):
            forbidden_reads.append(str(path))
            raise AssertionError("frozen inference read test artifact")
        return original_read_text(path, *args, **kwargs)

    def no_test_open(path, *args, **kwargs):
        if protected(path):
            forbidden_reads.append(str(path))
            raise AssertionError("frozen inference opened test artifact")
        return original_open(path, *args, **kwargs)

    def no_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels_bot.pt", "labels.pt"}:
            raise AssertionError("frozen inference read class labels")
        return original(path, *args, **kwargs)

    rows = []
    with patch.object(torch, "load", no_labels):
        for seed in (0, 1, 2):
            path = root / ("output/v3_mgtab_ab_group_additive_tree/"
                           f"seed{seed}/test_predictions.json")
            saved = json.loads(path.read_text(encoding="utf-8"))["predictions"]
            offline_path = path.parent / "source_group_decision_certificates.jsonl"
            offline = {row["account_id"]: row for row in (
                json.loads(line) for line in offline_path.read_text(
                    encoding="utf-8").splitlines())}
            by_id = {item["account_id"]: item for item in saved}
            projection_dependent = next(
                row["account_id"] for row in offline.values()
                if "published_tweet_projection" in row["pivotal_source_groups"])
            projection_invariant = next(
                row["account_id"] for row in offline.values()
                if "published_tweet_projection" not in row["pivotal_source_groups"])
            chosen_ids = list(dict.fromkeys((
                projection_dependent, projection_invariant, saved[-1]["account_id"])))
            chosen = [by_id[uid] for uid in chosen_ids]
            with patch.object(Path, "read_text", no_test_text), \
                    patch.object(Path, "open", no_test_open):
                runtime = MGTABFullAtomicRuntime(root, seed,
                                                 variant="ab_group_additive")
                observations = [runtime.predict(uid) for uid in chosen_ids]
            maximum = 0.0
            for item, observed in zip(chosen, observations, strict=True):
                maximum = max(maximum,
                              abs(observed["probability"] - item["probability"]))
                if (len(observed["evidence"]) != 150
                        or observed["group_logit_additivity_error"] > 1e-9
                        or abs(observed["intercept_logit"] + sum(
                            term["logit_contribution"] for term in observed["evidence"])
                            - observed["logit"]) > 1e-9):
                    raise ValueError("inference evidence fails logit reconstruction")
                certificate = observed["source_group_decision_certificate"]
                expected = offline[item["account_id"]]
                if (certificate is None
                        or certificate["pivotal_source_groups"] !=
                        expected["pivotal_source_groups"]
                        or certificate["smallest_sufficient_source_group_sets"] !=
                        expected["smallest_sufficient_source_group_sets"]
                        or certificate["zero_evidence_prediction"] !=
                        expected["zero_evidence_prediction"]
                        or certificate["evidence_strengthens_predicted_class"] !=
                        expected["evidence_strengthens_predicted_class"]
                        or certificate["tweet_projection_changes_prediction"] !=
                        ("published_tweet_projection" in
                         expected["pivotal_source_groups"])
                        or certificate["decision_source_scope"] != (
                            "published_averaged_tweet_tensor_required_for_class"
                            if "published_tweet_projection" in
                            expected["pivotal_source_groups"] else
                            "class_reproduced_with_published_profile_and_typed_relations")
                        or any(abs(certificate[
                            "source_group_logit_effect_when_removed"][name] -
                            expected["source_group_logit_effect_when_removed"][name])
                            > 1e-9 for name in expected[
                                "source_group_logit_effect_when_removed"])):
                    raise ValueError("live source group certificate differs from audit")
            if maximum > 1e-12 or forbidden_reads:
                raise ValueError("source-bound prediction differs from saved test")
            rows.append({"seed": seed, "accounts": len(chosen),
                         "max_probability_difference": maximum,
                         "group_subset_certificate_matches": len(chosen),
                         "projection_dependent_sample_checked": True,
                         "projection_invariant_sample_checked": True,
                         "label_file_read_attempts": 0,
                         "test_artifact_read_attempts": len(forbidden_reads)})
            print(rows[-1], flush=True)
    path = root / "research/certificates/mgtab_ab_group_source_bound_entry.json"
    path.write_text(json.dumps({"scope": "mgtab_ab_group_additive_label_free_entry",
                                "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
