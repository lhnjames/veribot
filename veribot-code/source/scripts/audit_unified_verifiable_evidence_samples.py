"""Independently check four saved source-bound evidence-to-decision examples."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scipy.special import expit


CASES = (
    ("botsim", 0, "irfvivs", "output/v3_botsim_observed_means_12/seed0/test_predictions.json", 12),
    ("fox8", 0, "830915719", "output/v3_fox8_native_spline_l2/group_tail_20_selected/seed0/test_predictions.json", 20),
    ("mgtab", 0, "2272", "output/v3_mgtab_ab_spline_content/direct_selected_20/seed0/test_predictions.json", 20),
    ("twibot20", None, "1188812492010487808", "output/v3_twibot20_sparse_text_evidence/selected/test_predictions.json", 153),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    samples = root / "output/v3_unified_verified_evidence"
    rows = []
    for dataset, seed, account_id, relative, expected_terms in CASES:
        name = (f"{dataset}_seed{seed}_{account_id}.json" if seed is not None
                else f"{dataset}_{account_id}.json")
        path = samples / name
        sample = read(path)
        prediction_path = root / relative
        saved = read(prediction_path)
        saved_threshold = (read(root / "output/v3_botsim_observed_means_12/dev_only.json")
                           ["runs"][seed]["threshold"] if dataset == "botsim" else
                           saved["manifest"]["threshold"])
        prediction = next(row for row in saved["predictions"]
                          if row["account_id"] == account_id)
        terms = sample["evidence_items"]
        if (sample["contract"] != "audited_source_to_evidence_to_frozen_decision_v1"
                or sample["dataset"] != dataset or sample["seed"] != seed
                or sample["account_id"] != account_id or len(terms) != expected_terms
                or sample["threshold"] != saved_threshold
                or not sample["source_binding"] or not sample["model_binding"]
                or sample["caller_supplied_fact_matrix_accepted"] is not False
                or sample["classification_label_field_used"] is not False):
            raise ValueError(f"sample contract differs: {name}")
        score = sample["intercept"] + math.fsum(
            item["logit_contribution"] for item in terms)
        probability = float(expit(score))
        predicted = int(probability >= sample["threshold"])
        if (abs(score - sample["logit"]) > 1e-9 or
                abs(probability - sample["probability"]) > 1e-10 or
                abs(probability - prediction["probability"]) > 1e-10 or
                predicted != sample["prediction"] or
                predicted != int(prediction["probability"] >= sample["threshold"])):
            raise ValueError(f"sample does not reproduce fixed test prediction: {name}")
        for item in terms:
            if (abs(item["logit_contribution"]) > 1e-12 and
                    (not item["source_query"] or
                     not item["source_query"].get("source_ids"))):
                raise ValueError("nonzero evidence lacks a source locator")
            if dataset != "twibot20":
                tail = item["selected_tail"]
                if tail not in ("upper", "lower"):
                    raise ValueError("selected evidence tail differs")
                selected = item[f"evidence_{tail}"]
                magnitude = math.log1p(selected) * item["evidence_reliability"]
                if abs(magnitude - item["evidence_value"]) > 1e-12:
                    raise ValueError("evidence magnitude does not replay")
        if dataset == "mgtab":
            projections = [item for item in terms if item["source_tier"] ==
                           "published_mean_tweet_projection"]
            if len(projections) != 16 or any(
                    item["published_profile_semantics"] is not None
                    for item in projections):
                raise ValueError("MGTAB content projection is mislabeled")
            named = [item for item in terms if item["source_tier"] ==
                     "published_numeric_profile_or_typed_edge"]
            mask = sample["published_content_projection_mask_audit"]
            masked_probability = float(expit(sample["intercept"] + math.fsum(
                item["logit_contribution"] for item in named)))
            if (len(named) != 4 or
                    abs(masked_probability - mask["content_masked_probability"]) > 1e-12 or
                    int(masked_probability >= sample["threshold"]) !=
                    mask["content_masked_prediction"] or
                    mask["content_projection_changes_class"] !=
                    (mask["content_masked_prediction"] != predicted)):
                raise ValueError("MGTAB content mask decision differs")
        if dataset == "twibot20":
            active = [item for item in terms if item["source_tier"] ==
                      "official_raw_tweet_token" and item["evidence_value"] > 0]
            if not active or any(not item["source_query"]["source_ids"]
                                 for item in active):
                raise ValueError("TwiBot active unigram lacks a tweet location")
        rows.append({
            "dataset": dataset, "seed": seed, "account_id": account_id,
            "evidence_items": len(terms),
            "prediction": predicted,
            "probability_difference_from_saved": probability - prediction["probability"],
            "logit_reconstruction_difference": score - sample["logit"],
            "decision_basis_status": sample["decision_basis_status"],
            "sample_sha256": digest(path),
            "saved_predictions_sha256": digest(prediction_path),
        })
    result = {"scope": "four_source_bound_examples_against_frozen_test_predictions",
              "rows": rows}
    destination = root / "research/four_dataset_unified_verifiable_evidence_observed_means_samples_2026-11-03.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
