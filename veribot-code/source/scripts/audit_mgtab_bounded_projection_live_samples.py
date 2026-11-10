"""Check capped projection reports against the locked full-test matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_bounded_projection_report import bounded_projection_report
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_explicit_bounded_projection_residual"
CERT = ROOT / "research/certificates/mgtab_bounded_projection_live_samples.json"


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        artifact = OUT / f"seed{seed}_test_effects.npz"
        with np.load(artifact, allow_pickle=False) as array:
            ids = array["ids"].astype(str)
            residual = array["projection_residual"]
            clipped = np.flatnonzero(np.abs(residual) > 3)
            plain = np.flatnonzero(np.abs(residual) <= 3)
            selected = [int(clipped[0]), int(plain[0]), len(ids) // 2,
                        len(ids) - 1]
            selected = sorted(set(selected))
            maximum = 0.0
            checks = []
            for j in selected:
                original = runtime.predict(str(ids[j]))
                report = bounded_projection_report(runtime, original)
                differences = [
                    abs(report["bounded_logit"] - float(array["bounded_logit"][j])),
                    abs(report["named_evidence_logit"] - float(array["named_logit"][j])),
                    abs(report["numeric_projection_residual_raw_logit"] -
                        float(residual[j])),
                    abs(report["numeric_projection_residual_absolute_effect_share"] -
                        float(array["residual_absolute_effect_share"][j])),
                    abs(float(original["logit"]) - float(array["full_logit"][j])),
                ]
                maximum = max(maximum, *differences)
                if (report["bounded_prediction"] !=
                    int(report["bounded_probability"] >= runtime.threshold)):
                    raise ValueError("live bounded class differs")
                checks.append({
                    "account_id": str(ids[j]),
                    "residual_was_clipped": report["residual_was_clipped"],
                    "original_prediction": original["prediction"],
                    "bounded_prediction": report["bounded_prediction"],
                })
        if maximum > 1e-12 or not any(
                row["residual_was_clipped"] for row in checks):
            raise ValueError("live bounded report differs from test matrix")
        runs.append({
            "seed": seed,
            "effect_matrix_sha256": sha256_file(artifact),
            "sample_count": len(checks),
            "max_live_vs_batch_logit_difference": maximum,
            "samples": checks,
        })
    CERT.write_text(json.dumps({
        "status": "source_bound_bounded_residual_runtime_matches_locked_test_matrix",
        "classification_labels_loaded_by_runtime": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print({"max_difference": max(row["max_live_vs_batch_logit_difference"]
                                  for row in runs),
           "sample_count": sum(row["sample_count"] for row in runs)})
    print(CERT)


if __name__ == "__main__":
    main()
