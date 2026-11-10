"""Check label-free source-bound compact MGTAB inference on selected accounts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_twenty_concept_runtime import MGTABTwentyConceptRuntime


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original_load = torch.load

    def reject_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels_bot.pt", "labels.pt"}:
            raise AssertionError("Compact runtime attempted to read account labels")
        return original_load(path, *args, **kwargs)

    result = []
    with patch.object(torch, "load", reject_labels):
        for seed in (0, 1, 2):
            run_dir = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree" / f"seed{seed}"
            predictions = json.loads((run_dir / "test_predictions.json")
                                     .read_text(encoding="utf-8"))["predictions"]
            with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as arr:
                masks = arr["group_mask_probability"].copy()
                concept = arr["concept_scores"].copy()
                names = arr["concept_names"].astype(str).tolist()
            threshold = json.loads((root /
                "output/v3_mgtab_targeted_20_concept_ab_group_tree/selection_lock.json")
                .read_text(encoding="utf-8"))["threshold_by_seed"][str(seed)]
            content_change = np.any((masks[:, 6:] >= threshold) !=
                                    (np.asarray([p["probability"] for p in predictions])[:, None]
                                     >= threshold), axis=1)
            chosen = [int(np.flatnonzero(content_change)[0]),
                      int(np.flatnonzero(~content_change)[0]),
                      len(predictions) - 1]
            chosen = list(dict.fromkeys(chosen))
            runtime = MGTABTwentyConceptRuntime(root, seed)
            max_diff = 0.0
            for j in chosen:
                card = runtime.predict(predictions[j]["account_id"])
                max_diff = max(max_diff,
                               abs(card["probability"] - predictions[j]["probability"]))
                if (len(card["evidence"]) != 20
                        or [item["concept"] for item in card["evidence"]] != names
                        or any(abs(item["value"] - concept[j, k]) > 1e-14
                               for k, item in enumerate(card["evidence"]))
                        or any(abs(card["source_group_effects"][name][
                            "probability_without_group"] - masks[j, k]) > 1e-14
                            for k, name in enumerate(card["source_group_effects"]))):
                    raise ValueError("Live compact source prediction differs from saved test")
            if max_diff > 1e-12:
                raise ValueError("Label-free runtime probability differs")
            result.append({"seed": seed, "accounts_checked": len(chosen),
                           "projection_dependent_sample_checked": True,
                           "projection_invariant_sample_checked": True,
                           "max_probability_difference": max_diff,
                           "label_file_read_attempts": 0})
            print(json.dumps(result[-1]), flush=True)
    path = root / "research/certificates/mgtab_targeted_20_concept_source_bound_entry.json"
    path.write_text(json.dumps({
        "scope": "MGTAB_twenty_concept_label_free_source_bound_entry",
        "runs": result,
        "status": "selected_live_predictions_match_frozen_test_without_labels"},
        indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
