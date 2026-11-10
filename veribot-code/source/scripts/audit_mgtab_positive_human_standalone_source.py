"""Check standalone Human-variant source state against the prior full source loader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_ab_positive_human_runtime import MGTABABPositiveHumanRuntime
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rows = []
    for seed in (0, 1, 2):
        old = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        current = MGTABABPositiveHumanRuntime(ROOT, seed)
        if (old.ids != current.ids or old.names != current.names or
                not np.array_equal(old.facts, current.facts, equal_nan=True) or
                not np.array_equal(old.supports, current.supports, equal_nan=True) or
                old.reference.version != current.reference.version or
                old.group_fact_indices != current.group_fact_indices):
            raise ValueError("standalone source population differs")
        saved = json.loads((ROOT /
            f"output/v3_mgtab_ab_group_positive_human/seed{seed}/test_predictions.json")
            .read_text(encoding="utf-8"))["predictions"]
        sample = saved[0]
        actual = current.predict(sample["account_id"])
        if abs(actual["probability"]-sample["probability"]) > 1e-12:
            raise ValueError("standalone positive Human prediction differs")
        rows.append({
            "seed": seed, "source_accounts": len(current.ids),
            "source_fact_columns": len(current.names),
            "source_values_equal_including_missing": True,
            "neighbor_support_values_equal": True,
            "source_group_partition_equal": True,
            "sample_probability_difference":
                abs(actual["probability"]-sample["probability"]),
        })
        print(rows[-1], flush=True)
    path = (ROOT /
        "research/certificates/mgtab_ab_positive_human_standalone_source.json")
    path.write_text(json.dumps({
        "scope": "standalone_source_initialization_equivalent_for_all_published_accounts",
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
