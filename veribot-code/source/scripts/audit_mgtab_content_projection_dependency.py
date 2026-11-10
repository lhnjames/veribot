"""Measure exact fixed-model reliance on MGTAB's unnamed content projections."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    selected = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    verifier_path = selected / "independent_verification.json"
    verifier = read(verifier_path)
    rows = []
    for seed in (0, 1, 2):
        folder = selected / f"seed{seed}"
        cards_path = folder / "evidence_cards.jsonl.gz"
        report_path = folder / "report.json"
        predictions_path = folder / "test_predictions.json"
        hypbot_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        report = read(report_path)
        predictions = read(predictions_path)
        hypbot = read(hypbot_path)
        verified = next(item for item in verifier["runs"] if item["seed"] == seed)
        if (digest(cards_path) != report["cards_sha256"] or
                digest(cards_path) != verified["cards_sha256"] or
                digest(report_path) != verified["model_report_sha256"] or
                predictions["manifest"] != report["manifest"] or
                predictions["manifest"]["split_sha256"] !=
                hypbot["manifest"]["split_sha256"]):
            raise ValueError("fixed cards or source audit differ")
        saved = predictions["predictions"]
        other = hypbot["predictions"]
        if len(saved) != 1020 or len(other) != 1020:
            raise ValueError("fixed test population differs")
        threshold = report["manifest"]["threshold"]
        h_threshold = hypbot["manifest"]["threshold"]
        counts = {key: 0 for key in (
            "full_correct", "named_only_correct", "hypbot_correct",
            "content_mask_changes_class", "content_mask_repairs_error",
            "content_mask_creates_error", "content_net_correct",
            "content_group_supports_current_class", "content_group_larger_than_named",
            "at_least_one_content_term_individually_pivotal")}
        per_account = []
        maximum_probability_difference = 0.0
        with gzip.open(cards_path, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= len(saved):
                    raise ValueError("more cards than saved test rows")
                card = json.loads(line)
                prior, baseline = saved[index], other[index]
                if (card["account_id"] != prior["account_id"] or
                        card["account_id"] != baseline["account_id"] or
                        prior["label"] != baseline["label"] or
                        len(card["terms"]) != 20):
                    raise ValueError("paired account or evidence count differs")
                named = [term for term in card["terms"] if not
                         term["name"].startswith("published_tweet_projection_")]
                content = [term for term in card["terms"] if
                           term["name"].startswith("published_tweet_projection_")]
                if len(named) != 4 or len(content) != 16 or any(
                        term["source_locator"]["source_level"] !=
                        "published_averaged_tweet_tensor_projection"
                        for term in content):
                    raise ValueError("published content source tier differs")
                named_sum = math.fsum(term["logit_contribution"] for term in named)
                content_sum = math.fsum(term["logit_contribution"] for term in content)
                logit = card["intercept"] + math.fsum((named_sum, content_sum))
                full_probability = float(expit(logit))
                masked_probability = float(expit(card["intercept"] + named_sum))
                maximum_probability_difference = max(
                    maximum_probability_difference,
                    abs(full_probability - card["probability"]),
                    abs(full_probability - prior["probability"]))
                if abs(logit - card["logit"]) > 1e-10:
                    raise ValueError("fixed model logit does not reconstruct")
                full = int(full_probability >= threshold)
                masked = int(masked_probability >= threshold)
                h = int(baseline["probability"] >= h_threshold)
                if full != card["prediction"]:
                    raise ValueError("fixed model prediction differs")
                label = prior["label"]
                full_ok, masked_ok, h_ok = full == label, masked == label, h == label
                counts["full_correct"] += full_ok
                counts["named_only_correct"] += masked_ok
                counts["hypbot_correct"] += h_ok
                counts["content_mask_changes_class"] += full != masked
                counts["content_mask_repairs_error"] += masked_ok and not full_ok
                counts["content_mask_creates_error"] += full_ok and not masked_ok
                counts["content_net_correct"] += full_ok - masked_ok
                counts["content_group_supports_current_class"] += (
                    content_sum > 0 if full else content_sum < 0)
                counts["content_group_larger_than_named"] += (
                    abs(content_sum) > abs(named_sum))
                pivotal = any(int(expit(logit - term["logit_contribution"]) >=
                                  threshold) != full for term in content)
                counts["at_least_one_content_term_individually_pivotal"] += pivotal
                per_account.append({
                    "account_id": card["account_id"], "label": label,
                    "full_prediction": full, "named_only_prediction": masked,
                    "hypbot_prediction": h,
                    "full_probability": full_probability,
                    "named_only_probability": masked_probability,
                    "named_logit_contribution": named_sum,
                    "content_logit_contribution": content_sum,
                    "content_mask_changes_class": full != masked,
                    "content_individually_pivotal": pivotal,
                })
            if len(per_account) != len(saved):
                raise ValueError("card population differs")
        if (maximum_probability_difference > 1e-10 or
                abs(counts["full_correct"] / 1020 - report["test"]["acc"]) > 1e-10 or
                counts["content_net_correct"] !=
                counts["content_mask_creates_error"] -
                counts["content_mask_repairs_error"]):
            raise ValueError("fixed content mask audit does not balance")
        labels = [item["label"] for item in per_account]
        quality = {
            "full_macro_f1": float(f1_score(
                labels, [item["full_prediction"] for item in per_account],
                average="macro")),
            "content_masked_macro_f1": float(f1_score(
                labels, [item["named_only_prediction"] for item in per_account],
                average="macro")),
            "full_auroc": float(roc_auc_score(
                labels, [item["full_probability"] for item in per_account])),
            "content_masked_auroc": float(roc_auc_score(
                labels, [item["named_only_probability"] for item in per_account])),
        }
        rows.append({
            "seed": seed, "n": len(saved),
            "cards_sha256": digest(cards_path),
            "hypbot_predictions_sha256": digest(hypbot_path),
            "max_probability_reconstruction_difference": maximum_probability_difference,
            **counts, **quality, "accounts": per_account,
        })
        print(seed, counts["full_correct"], counts["named_only_correct"],
              counts["hypbot_correct"], counts["content_mask_changes_class"],
              flush=True)
    result = {
        "scope": "fixed_mgtab_20_concept_model_content_projection_zero_mask",
        "meaning": "same frozen model and threshold; zero-masking is not retraining a named-only model",
        "independent_verification_sha256": digest(verifier_path),
        "runs": rows,
    }
    output = root / "research/mgtab_selected_content_projection_dependency_2026-11-03.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
