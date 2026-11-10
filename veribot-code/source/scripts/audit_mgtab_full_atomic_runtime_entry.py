"""Sample the label-free source-bound MGTAB prediction interface."""

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

    def no_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels_bot.pt", "labels.pt"}:
            raise AssertionError("frozen inference read class labels")
        return original(path, *args, **kwargs)

    rows = []
    with patch.object(torch, "load", no_labels):
        for seed in (0, 1, 2):
            path = root / ("output/v3_mgtab_full_atomic_evidence_lightgbm/"
                           f"seed{seed}/test_predictions.json")
            saved = json.loads(path.read_text(encoding="utf-8"))["predictions"]
            runtime = MGTABFullAtomicRuntime(root, seed)
            chosen = [saved[0], saved[len(saved) // 2], saved[-1]]
            maximum = 0.0
            for item in chosen:
                observed = runtime.predict(item["account_id"])
                maximum = max(maximum,
                              abs(observed["probability"] - item["probability"]))
                if (len(observed["evidence"]) != 150
                        or abs(observed["intercept_logit"] + sum(
                            term["logit_contribution"] for term in observed["evidence"])
                            - observed["logit"]) > 1e-9):
                    raise ValueError("inference evidence fails logit reconstruction")
            if maximum > 1e-12:
                raise ValueError("source-bound prediction differs from saved test")
            rows.append({"seed": seed, "accounts": len(chosen),
                         "max_probability_difference": maximum,
                         "label_file_read_attempts": 0})
            print(rows[-1], flush=True)
    path = root / "research/certificates/mgtab_full_atomic_source_bound_entry.json"
    path.write_text(json.dumps({"scope": "mgtab_150_atomic_evidence_label_free_entry",
                                "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
