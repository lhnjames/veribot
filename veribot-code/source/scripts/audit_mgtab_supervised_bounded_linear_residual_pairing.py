"""Paired accounting for the dev-locked supervised MGTAB residual."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "output/v3_mgtab_reduced_projection_ab_group"
CANDIDATE = ROOT / "output/v3_mgtab_supervised_bounded_linear_residual"
OUT = ROOT / "research/certificates/mgtab_supervised_bounded_linear_residual_pairing.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    a_lock = read(PARENT / "selection_lock.json")
    b_lock = read(CANDIDATE / "selection_lock.json")
    results = []
    for seed in (0, 1, 2):
        a = read(PARENT / f"seed{seed}/test_predictions.json")
        b = read(CANDIDATE / f"seed{seed}/test_predictions.json")
        if ([((r["account_id"]), r["label"]) for r in a["predictions"]] !=
                [((r["account_id"]), r["label"]) for r in b["predictions"]]
                or a["manifest"]["split_sha256"] != b["manifest"]["split_sha256"]):
            raise ValueError("Frozen account pair differs")
        labels = np.asarray([r["label"] for r in a["predictions"]], dtype=int)
        parent = np.asarray([r["probability"] >=
                             a_lock["threshold_by_seed"][str(seed)]
                             for r in a["predictions"]], dtype=bool)
        candidate = np.asarray([r["probability"] >=
                                b_lock["threshold_by_seed"][str(seed)]
                                for r in b["predictions"]], dtype=bool)
        pa = parent == labels
        cb = candidate == labels
        row = {"seed": seed, "n": len(labels),
               "parent_correct": int(pa.sum()),
               "linear_correct": int(cb.sum()),
               "parent_only_correct": int(np.count_nonzero(pa & ~cb)),
               "linear_only_correct": int(np.count_nonzero(cb & ~pa)),
               "class_changes": int(np.count_nonzero(parent != candidate)),
               "parent_only_correct_by_true_class": {
                   str(c): int(np.count_nonzero(pa & ~cb & (labels == c)))
                   for c in (0, 1)},
               "linear_only_correct_by_true_class": {
                   str(c): int(np.count_nonzero(cb & ~pa & (labels == c)))
                   for c in (0, 1)}}
        results.append(row)
        print(row, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "same_accounts_eight_coordinate_tree_vs_supervised_sixteen_coefficient_residual",
        "candidate_selection_did_not_use_test_metrics": True,
        "same_test_population_previously_explored": True,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
