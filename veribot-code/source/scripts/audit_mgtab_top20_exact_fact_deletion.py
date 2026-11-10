"""Replay exact fact-deletion effects of the frozen 20-source-evidence classifier.

This is a model-input intervention: it removes both evidence tails for one fact,
then reevaluates the actual frozen booster. It does not claim that changing a
social account's real-world property would have the same effect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_named_top20_lightgbm"
    lock_path = folder / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads((folder / "selected/paired_summary.json")
                         .read_text(encoding="utf-8"))
    if summary["selection_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("Frozen selection lock differs")
    source_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    if (lock["source_features_sha256"] != sha256_file(source_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(structural)
            or lock["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("Frozen published source differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=source_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor,
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    runs = []
    for seed in (0, 1, 2):
        base = folder / f"selected/seed{seed}"
        report = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        selected = np.asarray(lock["selected_fact_indices_by_seed"][str(seed)],
                              dtype=int)
        if (len(selected) != 20 or len(set(map(int, selected))) != 20
                or report["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("Frozen fact set or official split differs")
        test = split["test"]
        parent_reference = (root / "output/v3_mgtab_named_evidence_lightgbm"
                            / f"selected/seed{seed}/reference.json.gz")
        reference = LabelBlindReference.load(parent_reference)
        if list(reference.names) != names[:86]:
            raise ValueError("Named evidence reference differs")
        xt = matrix(transform_with_support(
            reference, x[test, :86], split["supports"][test]))
        xt = xt[:, np.r_[selected, selected + 86]]
        model_path = base / "model.txt"
        if sha256_file(model_path) != summary["runs"][seed]["model_sha256"]:
            raise ValueError("Frozen model differs")
        booster = lightgbm.Booster(model_file=str(model_path))
        probability = np.asarray(booster.predict(xt, num_threads=4))
        logit = np.asarray(booster.predict(xt, raw_score=True, num_threads=4))
        saved = json.loads((base / "test_predictions.json").read_text())
        predictions = saved["predictions"] if isinstance(saved, dict) else saved
        if len(predictions) != len(test):
            raise ValueError("Frozen test prediction count differs")
        max_probability_error = 0.0
        for position, index in enumerate(test):
            item = predictions[position]
            if item["account_id"] != str(int(index)) or item["label"] != int(y[index]):
                raise ValueError("Frozen test account order differs")
            max_probability_error = max(
                max_probability_error,
                abs(float(item["probability"]) - float(probability[position])))
        if max_probability_error > 1e-12:
            raise ValueError("Frozen probability replay differs")
        threshold = float(lock["threshold_by_seed"][str(seed)])
        full_class = probability >= threshold
        deletion_logit = np.empty((len(test), 20), dtype=float)
        deletion_probability = np.empty_like(deletion_logit)
        for fact in range(20):
            masked = xt.copy()
            masked[:, (fact, fact + 20)] = 0.0
            deletion_logit[:, fact] = booster.predict(
                masked, raw_score=True, num_threads=4)
            deletion_probability[:, fact] = booster.predict(
                masked, num_threads=4)
        effects = logit[:, None] - deletion_logit
        flips = full_class[:, None] != (deletion_probability >= threshold)
        nonzero = np.abs(effects) > 1e-10
        records = []
        for position, index in enumerate(test):
            records.append({
                "account_id": str(int(index)),
                "label": int(y[index]),
                "prediction": int(full_class[position]),
                "probability": float(probability[position]),
                "raw_logit": float(logit[position]),
                "fact_deletions": [
                    {
                        "fact_name": names[int(selected[j])],
                        "raw_fact_value": float(x[index, selected[j]]),
                        "upper_evidence_input": float(xt[position, j]),
                        "lower_evidence_input": float(xt[position, j + 20]),
                        "logit_without_fact": float(deletion_logit[position, j]),
                        "probability_without_fact": float(deletion_probability[position, j]),
                        "exact_input_removal_logit_effect": float(effects[position, j]),
                        "class_flips_on_removal": bool(flips[position, j]),
                    }
                    for j in range(20)
                ],
            })
        result_path = base / "exact_fact_deletion.jsonl"
        with result_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        run = {
            "seed": seed,
            "n": len(test),
            "model_sha256": sha256_file(model_path),
            "source_reference_sha256": sha256_file(parent_reference),
            "max_frozen_probability_replay_error": max_probability_error,
            "account_fact_deletions": int(np.prod(effects.shape)),
            "nonzero_exact_logit_effects": int(np.count_nonzero(nonzero)),
            "accounts_with_at_least_one_nonzero_effect":
                int(np.count_nonzero(np.any(nonzero, axis=1))),
            "accounts_with_at_least_one_class_flip":
                int(np.count_nonzero(np.any(flips, axis=1))),
            "profile_fact_class_flips": int(np.count_nonzero(flips[:, selected < 20])),
            "native_relation_fact_class_flips":
                int(np.count_nonzero(flips[:, selected >= 20])),
            "certificate_sha256": sha256_file(result_path),
            "certificate_path": str(result_path),
        }
        runs.append(run)
        print(json.dumps({"seed": seed, "n": len(test),
                          "account_fact_deletions": run["account_fact_deletions"],
                          "accounts_with_at_least_one_class_flip":
                          run["accounts_with_at_least_one_class_flip"]}), flush=True)
    report = {
        "scope": "frozen_mgtab_test_exact_named_evidence_input_deletion_audit",
        "semantics": "Sets both tails of one source fact to the neutral zero-evidence input and reevaluates the frozen booster; no real-world causal claim",
        "selection_lock_sha256": sha256_file(lock_path),
        "runs": runs,
    }
    path = root / "research/mgtab_top20_exact_fact_deletion_2026-11-03.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
