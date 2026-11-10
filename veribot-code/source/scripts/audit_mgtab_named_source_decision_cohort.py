"""Evaluate where the frozen MGTAB class survives removal of content projections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_correct(prediction: np.ndarray, label: np.ndarray,
                  population: np.ndarray) -> int:
    return int(np.count_nonzero((prediction == label) & population))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_ab_group_additive_tree"
    certificate_path = root / "research/certificates/mgtab_ab_group_decision_certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        prediction_path = run_dir / "test_predictions.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        manifest = prediction["manifest"]
        model_path = run_dir / "model.joblib"
        evidence_path = run_dir / "test_evidence.npz"
        if (sha256_file(model_path) != manifest["model_sha256"] or
                sha256_file(evidence_path) != manifest["evidence_sha256"]):
            raise ValueError("frozen model or evidence artifact differs")
        model = joblib.load(model_path)
        with np.load(evidence_path, allow_pickle=False) as saved:
            matrix = saved["evidence_matrix"].copy()
            probability = saved["probability"].copy()
            labels = saved["label"].astype(int)
            ids = saved["ids"].astype(str).tolist()
        if matrix.shape != (1020, 300):
            raise ValueError("MGTAB evidence population differs")
        predicted = model.predict_proba(matrix)[:, 1]
        if np.max(np.abs(predicted - probability)) > 1e-12:
            raise ValueError("frozen full prediction differs")
        masked = matrix.copy()
        masked[:, 86:150] = 0.0
        masked[:, 236:300] = 0.0
        masked_probability = model.predict_proba(masked)[:, 1]
        threshold = float(manifest["threshold"])
        full_class = probability >= threshold
        named_class = masked_probability >= threshold
        invariant = full_class == named_class
        dependent = ~invariant
        prior = certificate["runs"][seed]
        card_path = root / prior["certificate_file"]
        if sha256_file(card_path) != prior["certificate_sha256"]:
            raise ValueError("source-group card digest differs")
        cards = [json.loads(line) for line in card_path.read_text(
            encoding="utf-8").splitlines()]
        if (len(cards) != len(ids) or any(
                card["account_id"] != uid or
                ("published_tweet_projection" in
                 card["pivotal_source_groups"]) != bool(dependent[j])
                for j, (card, uid) in enumerate(zip(cards, ids, strict=True)))):
            raise ValueError("new projection mask disagrees with independent group cards")
        row = {
            "seed": seed,
            "test_accounts": len(ids),
            "full_correct": count_correct(full_class, labels,
                                          np.ones(len(ids), dtype=bool)),
            "named_groups_only_correct": count_correct(
                named_class, labels, np.ones(len(ids), dtype=bool)),
            "class_invariant_without_projection_accounts": int(invariant.sum()),
            "class_invariant_without_projection_correct": count_correct(
                full_class, labels, invariant),
            "class_invariant_without_projection_acc": float(np.mean(
                full_class[invariant] == labels[invariant])),
            "projection_dependent_accounts": int(dependent.sum()),
            "projection_dependent_correct": count_correct(
                full_class, labels, dependent),
            "projection_dependent_acc": float(np.mean(
                full_class[dependent] == labels[dependent])),
            "projection_dependent_bot_predictions": int(np.count_nonzero(
                dependent & full_class)),
            "projection_dependent_human_predictions": int(np.count_nonzero(
                dependent & ~full_class)),
            "invariant_bot_predictions": int(np.count_nonzero(
                invariant & full_class)),
            "invariant_bot_correct": count_correct(
                full_class, labels, invariant & full_class),
            "invariant_human_predictions": int(np.count_nonzero(
                invariant & ~full_class)),
            "invariant_human_correct": count_correct(
                full_class, labels, invariant & ~full_class),
            "projection_dependent_bot_correct": count_correct(
                full_class, labels, dependent & full_class),
            "projection_dependent_human_correct": count_correct(
                full_class, labels, dependent & ~full_class),
            "maximum_full_probability_replay_error": float(np.max(
                np.abs(predicted - probability))),
            "model_sha256": manifest["model_sha256"],
            "group_decision_cards_sha256": prior["certificate_sha256"],
        }
        if row["projection_dependent_accounts"] != prior[
                "pivotal_source_group_counts"]["published_tweet_projection"]:
            raise ValueError("projection dependent cohort count differs")
        runs.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "scope": "mgtab_ab_group_tree_projection_mask_decision_cohorts",
        "interpretation": "Frozen-model class retention after zeroing tensor-level content projection evidence; not external semantic truth",
        "source_group_certificate_sha256": sha256_file(certificate_path),
        "runs": runs,
    }
    output = root / "research/certificates/mgtab_named_source_decision_cohort.json"
    output.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
