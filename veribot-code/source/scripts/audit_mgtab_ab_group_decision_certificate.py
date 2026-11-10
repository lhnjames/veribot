"""Exact source-group decision certificates for the frozen A/B MGTAB tree."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit


GROUPS = {
    "published_profile": np.arange(0, 20, dtype=int),
    "published_native_relations": np.arange(20, 86, dtype=int),
    "published_tweet_projection": np.arange(86, 150, dtype=int),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_ab_group_additive_tree"
    names = list(GROUPS)
    all_masks = list(range(1 << len(names)))
    summary = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        model_path = run_dir / "model.joblib"
        evidence_path = run_dir / "test_evidence.npz"
        predictions = json.loads((run_dir / "test_predictions.json")
                                 .read_text(encoding="utf-8"))
        manifest = predictions["manifest"]
        if (sha256_file(model_path) != manifest["model_sha256"]
                or sha256_file(evidence_path) != manifest["evidence_sha256"]):
            raise ValueError("frozen model or evidence digest differs")
        model = joblib.load(model_path)
        threshold = float(manifest["threshold"])
        with np.load(evidence_path, allow_pickle=False) as saved:
            evidence = saved["evidence_matrix"].copy()
            expected_probability = saved["probability"].copy()
            ids = saved["ids"].astype(str)
            labels = saved["label"].copy()
        if evidence.shape != (1020, 300):
            raise ValueError("unexpected evidence population")
        if any(row["account_id"] != account_id or
               row["label"] != int(label) or
               abs(row["probability"] - float(prob)) > 1e-14
               for row, account_id, label, prob in zip(
                   predictions["predictions"], ids, labels,
                   expected_probability, strict=True)):
            raise ValueError("saved test IDs, labels or probabilities differ")
        probability = np.empty((len(all_masks), len(ids)), dtype=np.float64)
        raw = np.empty_like(probability)
        for mask in all_masks:
            retained = np.zeros_like(evidence)
            for group_index, group in enumerate(names):
                if mask & (1 << group_index):
                    indices = GROUPS[group]
                    retained[:, indices] = evidence[:, indices]
                    retained[:, indices + 150] = evidence[:, indices + 150]
            raw[mask] = model.booster_.predict(retained, raw_score=True)
            probability[mask] = expit(raw[mask])
        if np.max(np.abs(probability[7] - expected_probability)) > 1e-12:
            raise ValueError("full-evidence prediction differs from frozen artifact")
        if np.max(np.abs(raw[7] - raw[0] - sum(
                raw[7] - raw[7 ^ (1 << j)] for j in range(3)))) > 1e-9:
            raise ValueError("source-group logit additivity failed")
        classes = probability >= threshold
        full_class = classes[7]
        empty_class = classes[0]
        logit_shift = raw[7] - raw[0]
        strengthens_prediction = np.where(full_class, logit_shift > 1e-12,
                                          logit_shift < -1e-12)
        minimal = []
        pivotal = []
        for account in range(len(ids)):
            candidates = [mask for mask in all_masks
                          if classes[mask, account] == full_class[account]]
            smallest = min(mask.bit_count() for mask in candidates)
            minimal.append([mask for mask in candidates
                            if mask.bit_count() == smallest])
            pivotal.append([name for j, name in enumerate(names)
                            if classes[7 ^ (1 << j), account] != full_class[account]])
        output = run_dir / "source_group_decision_certificates.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for j, account_id in enumerate(ids):
                stream.write(json.dumps({
                    "account_id": account_id,
                    "prediction": int(full_class[j]),
                    "full_probability": float(probability[7, j]),
                    "zero_evidence_probability": float(probability[0, j]),
                    "zero_evidence_prediction": int(empty_class[j]),
                    "zero_evidence_changes_prediction": bool(
                        empty_class[j] != full_class[j]),
                    "evidence_logit_shift_from_zero": float(logit_shift[j]),
                    "evidence_strengthens_predicted_class": bool(
                        strengthens_prediction[j]),
                    "pivotal_source_groups": pivotal[j],
                    "smallest_sufficient_source_group_sets": [
                        [names[k] for k in range(3) if mask & (1 << k)]
                        for mask in minimal[j]],
                    "source_group_logit_effect_when_removed": {
                        names[k]: float(raw[7, j] - raw[7 ^ (1 << k), j])
                        for k in range(3)},
                }, ensure_ascii=False) + "\n")
        count_by_minimum_groups = {
            str(k): int(sum(minimal[j][0].bit_count() == k
                            for j in range(len(ids)))) for k in range(4)}
        run = {
            "seed": seed,
            "test_accounts": len(ids),
            "full_correct": int(np.count_nonzero(full_class == labels)),
            "zero_evidence_correct": int(np.count_nonzero(empty_class == labels)),
            "zero_evidence_changes_prediction": int(np.count_nonzero(
                empty_class != full_class)),
            "full_bot_predictions": int(np.count_nonzero(full_class)),
            "full_human_predictions": int(np.count_nonzero(~full_class)),
            "human_predictions_strengthened_by_evidence": int(
                np.count_nonzero((~full_class) & strengthens_prediction)),
            "human_predictions_with_weaker_margin_than_zero_evidence": int(
                np.count_nonzero((~full_class) & (~strengthens_prediction))),
            "evidence_changed_and_correct": int(np.count_nonzero(
                (empty_class != full_class) & (full_class == labels))),
            "evidence_changed_and_wrong": int(np.count_nonzero(
                (empty_class != full_class) & (full_class != labels))),
            "minimum_sufficient_group_count_distribution": count_by_minimum_groups,
            "pivotal_source_group_counts": {
                name: int(sum(name in groups for groups in pivotal))
                for name in names},
            "bot_predictions_supported_by_single_group_alone": {
                name: int(np.count_nonzero(full_class & classes[1 << k]))
                for k, name in enumerate(names)},
            "bot_predictions_without_profile_group": int(np.count_nonzero(
                full_class & classes[6])),
            "certificate_file": str(output.relative_to(root)),
            "certificate_sha256": sha256_file(output),
        }
        summary.append(run)
        print(json.dumps(run), flush=True)
    path = root / "research/certificates/mgtab_ab_group_decision_certificate.json"
    path.write_text(json.dumps({
        "scope": "mgtab_ab_group_additive_all_test_accounts_exact_group_subsets",
        "semantics": "Frozen-model source-group input dependence, not a causal intervention on real accounts",
        "runs": summary}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
