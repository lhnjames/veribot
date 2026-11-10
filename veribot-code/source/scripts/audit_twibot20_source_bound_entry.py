"""Check official-source inference against the frozen test record and tampering."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from predict_twibot20_from_verified_zip import predict
from verifiable_bot.twibot20_sparse_runtime import sha256


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ids = ["17461978", "1224667050301255681", "1188812492010487808"]
    result = predict(root, ids)
    sample_path = root / "output/v3_source_bound_inference/twibot20_official_split_samples.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    assert [row["account_id"] for row in result["decisions"]] == ids
    assert all(len(row["structured_recompute_queries"]) == 22
               for row in result["decisions"])
    assert all(len(row["active_unigram_terms"]) == 109
               for row in result["decisions"])
    assert all(abs(row["logit"] - row["intercept"] -
                   sum(row["structured_contributions"]) -
                   sum(row["active_unigram_contributions"])) < 1e-10
               for row in result["decisions"])
    saved_path = root / "output/v3_twibot20_sparse_text_evidence/selected/test_predictions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    official = next(row for row in saved["predictions"]
                    if row["account_id"] == ids[-1])
    probability_difference = abs(
        result["decisions"][-1]["probability"] - official["probability"])
    assert probability_difference < 1e-12
    original = root / "data/processed/twibot20_v3/facts.npz"
    rejected = False
    rejection_reason = None
    with tempfile.TemporaryDirectory() as temporary:
        altered = Path(temporary) / "facts_changed.npz"
        shutil.copyfile(original, altered)
        with altered.open("ab") as stream:
            stream.write(b"source-integrity-test")
        try:
            predict(root, [ids[-1]], base_facts=altered)
        except ValueError as error:
            rejection_reason = str(error)
            rejected = "source, audit or model binding differs" in rejection_reason
    assert rejected, "modified fact cache must be rejected before prediction"
    output = {
        "dataset": "twibot20",
        "official_archive_sha256": result["source_binding"]["archive"],
        "model_sha256": result["model_binding"]["artifact"],
        "account_ids_and_splits": dict(zip(ids, ["train", "dev", "test"], strict=True)),
        "official_test_probability_difference": probability_difference,
        "max_raw_to_audited_fact_difference": result["max_raw_to_audited_fact_difference"],
        "max_independent_active_unigram_value_difference": result[
            "max_independent_active_unigram_value_difference"],
        "structured_query_counts": [len(row["structured_recompute_queries"])
                                    for row in result["decisions"]],
        "active_unigram_counts": [len(row["active_unigram_terms"])
                                  for row in result["decisions"]],
        "located_active_term_counts": [len(row["active_unigram_source_locations"])
                                       for row in result["decisions"]],
        "individual_pivotal_term_counts": [
            len(row["decision_dependency"]["individually_pivotal_terms"])
            for row in result["decisions"]],
        "minimum_count_support_sizes": [
            len(row["decision_dependency"]["minimum_count_support_under_other_terms_masked"])
            for row in result["decisions"]],
        "altered_fact_cache_rejected": rejected,
        "altered_fact_cache_rejection_reason": rejection_reason,
        "classification_label_field_used": result["classification_label_field_used"],
        "caller_supplied_fact_matrix_accepted": result[
            "caller_supplied_fact_matrix_accepted"],
        "frozen_test_predictions_sha256": sha256(saved_path),
        "source_bound_sample_sha256": sha256(sample_path),
    }
    target = root / "research/twibot20_source_bound_inference_2026-11-03.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
