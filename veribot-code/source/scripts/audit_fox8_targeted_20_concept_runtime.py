"""Check live Fox8 compact predictions from sources without class labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_twenty_concept_runtime import Fox8TwentyConceptRuntime


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original_read_csv = pd.read_csv

    def reject_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels.csv", "label.csv", "labels_bot.csv"}:
            raise AssertionError("Fox8 compact runtime attempted to read labels")
        return original_read_csv(path, *args, **kwargs)

    rows = []
    with patch.object(pd, "read_csv", reject_labels):
        for seed in (0, 1, 2):
            folder = root / "output/v3_fox8_targeted_neighbor_20_concepts" / f"seed{seed}"
            predictions = json.loads((folder / "ab_group_tree_test_predictions.json")
                                     .read_text(encoding="utf-8"))["predictions"]
            with np.load(folder / "ab_group_tree_test_evidence.npz",
                         allow_pickle=False) as saved:
                concept_scores = saved["concept_scores"].copy()
                concept_names = saved["concept_names"].astype(str).tolist()
                mask_p = saved["neighbor_mask_probability"].copy()
                threshold = float(json.loads((root /
                    "output/v3_fox8_targeted_neighbor_20_concepts/selection_lock_ab_group_tree.json")
                    .read_text(encoding="utf-8"))["threshold_by_seed"][str(seed)])
            p = np.asarray([row["probability"] for row in predictions])
            changed = (p >= threshold) != (mask_p >= threshold)
            chosen = list(dict.fromkeys((int(np.flatnonzero(changed)[0]),
                                         int(np.flatnonzero(~changed)[0]),
                                         len(predictions) - 1)))
            runtime = Fox8TwentyConceptRuntime(root, seed)
            maximum = 0.0
            for j in chosen:
                result = runtime.predict(predictions[j]["account_id"])
                maximum = max(maximum,
                              abs(result["probability"] - predictions[j]["probability"]))
                if (len(result["evidence"]) != 20
                        or [item["concept"] for item in result["evidence"]] !=
                        concept_names
                        or any(abs(item["value"] - concept_scores[j, k]) > 1e-14
                               for k, item in enumerate(result["evidence"]))
                        or abs(result["source_group_effects"]["native_neighbor_volume"]
                               ["probability_without_group"] - mask_p[j]) > 1e-14):
                    raise ValueError("Live source-bound prediction differs from saved test")
            if maximum > 1e-12:
                raise ValueError("Fox8 compact runtime probability differs")
            rows.append({"seed": seed, "accounts_checked": len(chosen),
                         "neighbor_dependent_sample_checked": True,
                         "neighbor_invariant_sample_checked": True,
                         "max_probability_difference": maximum,
                         "label_file_read_attempts": 0})
            print(json.dumps(rows[-1]), flush=True)
    path = root / "research/certificates/fox8_targeted_20_concept_source_bound_entry.json"
    path.write_text(json.dumps({
        "scope": "Fox8_twenty_concept_label_free_source_bound_entry",
        "runs": rows,
        "status": "selected_live_predictions_match_frozen_test_without_labels"},
        indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
