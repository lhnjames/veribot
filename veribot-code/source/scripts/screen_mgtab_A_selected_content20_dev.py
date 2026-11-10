"""Dev-only test of A-selected content coordinates within the same 20-fact MGTAB model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from screen_mgtab_twenty_source_facts_dev import candidate, choose, rank_facts
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "output/v3_mgtab_twenty_source_facts/dev_only.json"
OUT = ROOT / "output/v3_mgtab_twenty_source_facts/A_selected_content_dev.json"
QUOTA = (4, 8, 8)


def content_scores(x_a: np.ndarray, y_a: np.ndarray) -> np.ndarray:
    result = np.zeros(64)
    for local in range(64):
        j = 86 + local
        for col in (j, j + 150):
            values = x_a[:, col]
            if len(np.unique(values)) > 1:
                result[local] = max(result[local],
                                    abs(roc_auc_score(y_a, values) - .5))
    return result


def main() -> None:
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if prior["test_predictions_or_metrics_accessed"] or len(prior["runs"]) != 3:
        raise ValueError("20-fact control screen differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if ids != runtime.ids or source["split_sha256"] != prior["runs"][seed]["split_sha256"]:
            raise ValueError("MGTAB source or official split differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a, b = contract["a_indices"].astype(int), contract["b_indices"].astype(int)
        if len(set(a) & set(b)) or not np.array_equal(
                np.sort(np.r_[a, b]), np.sort(split["train"])):
            raise ValueError("A/B train partition differs")
        dev = split["val"]
        x_a, x_b, x_dev = (evidence_design(runtime, rows)
                           for rows in (a, b, dev))
        named_scores = rank_facts(x_a, labels[a])
        first8 = choose(runtime, named_scores, *QUOTA)
        reference = candidate(runtime, x_a, x_b, x_dev, labels,
                              a, b, dev, first8, QUOTA)
        earlier = prior["runs"][seed]["candidates"][0]
        if (reference["validation"] != earlier["validation"] or
                reference["threshold"] != earlier["threshold"] or
                first8.tolist() != earlier["selected_fact_indices"]):
            raise ValueError("first-eight control did not replay")
        scores = content_scores(x_a, labels[a])
        content = sorted(sorted(range(86, 150),
                                key=lambda j: (-scores[j - 86], j))[:8])
        selected = np.r_[first8[:12], content]
        if len(selected) != 20 or len(set(selected)) != 20:
            raise ValueError("selected fact budget differs")
        proposed = candidate(runtime, x_a, x_b, x_dev, labels,
                             a, b, dev, selected, QUOTA)
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "A_only_content_auc_scores": scores.tolist(),
            "first_eight_control": reference,
            "A_selected_content_candidate": proposed,
        })
        print({"seed": seed, "control_dev_f1": reference["validation"]["macro_f1"],
               "selected_dev_f1": proposed["validation"]["macro_f1"],
               "selected_content_indices": content}, flush=True)
    gains = [run["A_selected_content_candidate"]["validation"]["macro_f1"] -
             run["first_eight_control"]["validation"]["macro_f1"]
             for run in runs]
    pass_gate = bool(np.mean(gains) >= .005 and min(gains) >= -.002)
    OUT.write_text(json.dumps({
        "scope": "mgtab_A_only_content_coordinate_choice_in_twenty_fact_model",
        "control_screen_sha256": sha256_file(PRIOR),
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "selection": "A-only maximum absolute ROC-AUC of either evidence tail, top 8 of 64",
        "gate": "mean dev Macro-F1 improvement >=0.005 and every seed loss <=0.002",
        "mean_dev_macro_f1_gain": float(np.mean(gains)),
        "gate_passed_for_next_stage": pass_gate,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print({"gate_passed": pass_gate, "mean_dev_f1_gain": float(np.mean(gains))}, flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
