"""Aggregate source-group ablations for the frozen A/B MGTAB model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


GROUPS = {
    "published_profile": range(0, 20),
    "published_native_relations": range(20, 86),
    "published_tweet_projection": range(86, 150),
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_ab_group_additive_tree"
    results = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        model = joblib.load(run_dir / "model.joblib")
        prediction = json.loads((run_dir / "test_predictions.json")
                                .read_text(encoding="utf-8"))
        threshold = float(prediction["manifest"]["threshold"])
        with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as saved:
            x = saved["evidence_matrix"].copy()
            y = saved["label"].copy()
            probability = saved["probability"].copy()
        original = model.predict_proba(x)[:, 1]
        if np.max(np.abs(original - probability)) > 1e-14:
            raise ValueError("frozen MGTAB probability differs")
        full_class = probability >= threshold
        full_raw = np.asarray(model.booster_.predict(x, raw_score=True))
        empty_raw = np.asarray(model.booster_.predict(
            np.zeros_like(x), raw_score=True))
        masks = {}
        deltas = []
        for group, items in GROUPS.items():
            indices = np.asarray(list(items), dtype=int)
            masked = x.copy()
            masked[:, indices] = 0.0
            masked[:, indices + 150] = 0.0
            p = model.predict_proba(masked)[:, 1]
            raw = np.asarray(model.booster_.predict(masked, raw_score=True))
            deltas.append(full_raw - raw)
            masks[group] = {
                "changed_test_classes": int(np.count_nonzero(
                    (p >= threshold) != full_class)),
                "full_correct": int(np.count_nonzero(full_class == y)),
                "masked_correct": int(np.count_nonzero((p >= threshold) == y)),
                "maximum_absolute_logit_difference": float(np.max(np.abs(
                    full_raw - raw))),
            }
        error = float(np.max(np.abs(full_raw - empty_raw - sum(deltas))))
        if error > 1e-9:
            raise ValueError("three source-group logit terms do not add")
        results.append({"seed": seed, "test_accounts": len(y),
                        "max_three_group_logit_additivity_error": error,
                        "groups": masks})
        print(json.dumps({"seed": seed, "relation_changed": masks[
            "published_native_relations"]["changed_test_classes"],
            "content_changed": masks[
                "published_tweet_projection"]["changed_test_classes"]}), flush=True)
    path = root / "research/certificates/mgtab_ab_group_aggregate_masks.json"
    path.write_text(json.dumps({"scope": "mgtab_ab_group_additive_three_source_group_masks",
                                "runs": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
