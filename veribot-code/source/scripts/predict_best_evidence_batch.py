"""Batch account-ID inference through the three selected evidence models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_botsim_observed_means_from_sources import predict as predict_botsim
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


MODELS = {"botsim": "source_group_tail_12_observed_means",
          "fox8": "raw_neighbor_atomic_74",
          "mgtab": "ab_group_additive_150"}


def predict(root: Path, dataset: str, seed: int,
            account_ids: list[str]) -> dict:
    """Return frozen probabilities from evidence values, never caller facts/labels."""
    if dataset not in MODELS or seed not in (0, 1, 2):
        raise ValueError("choose a frozen dataset and seed")
    ids = [str(uid) for uid in account_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("provide distinct account IDs")
    root = Path(root).resolve()
    if dataset == "botsim":
        sourced = predict_botsim(root, dataset, seed, ids)
        decisions = sourced["decisions"]
        if ([row["account_id"] for row in decisions] != ids or
                sourced["classification_labels_accessed"] is not False or
                sourced["caller_supplied_fact_matrix_accepted"] is not False):
            raise ValueError("BotSim source-bound batch contract differs")
        probabilities = [float(row["probability"]) for row in decisions]
        predictions = [int(row["prediction"]) for row in decisions]
        threshold = float(decisions[0]["threshold"])
        reference_version = decisions[0]["atomic_reference_version"]
        source_binding = sourced["source_binding"]
        evidence_terms = 12
    else:
        runtime = (Fox8NeighborAtomicRuntime(root, seed) if dataset == "fox8"
                   else MGTABFullAtomicRuntime(root, seed,
                                               variant="ab_group_additive"))
        if not set(ids) <= set(runtime.index):
            raise KeyError("account absent from frozen source population")
        rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
        atomic = runtime.reference.transform(runtime.facts[rows])
        reliability = atomic.reliability.copy()
        if dataset == "fox8":
            reliability[:, 43:74] *= runtime.supports[rows] / (
                runtime.supports[rows] + 5.0)
            evidence_terms = 74
        else:
            reliability[:, 62:86] *= runtime.supports[rows] / (
                runtime.supports[rows] + 5.0)
            evidence_terms = 150
        matrix = np.column_stack((np.log1p(atomic.upper) * reliability,
                                  np.log1p(atomic.lower) * reliability))
        if (matrix.shape != (len(ids), 2*evidence_terms) or
                atomic.reference_version != runtime.reference.version):
            raise ValueError("frozen evidence matrix differs")
        probability = np.asarray(runtime.model.predict_proba(matrix)[:, 1],
                                 dtype=float)
        probabilities = probability.tolist()
        threshold = float(runtime.threshold)
        predictions = (probability >= threshold).astype(int).tolist()
        reference_version = runtime.reference.version
        source_binding = {
            "inference_manifest": str(root / (
                "output/v3_fox8_neighbor_atomic_evidence_capacity"
                if dataset == "fox8" else
                "output/v3_mgtab_ab_group_additive_tree") /
                "inference_manifest.json"),
        }
    if (len(probabilities) != len(ids) or
            not np.isfinite(probabilities).all() or
            any(not 0 <= p <= 1 for p in probabilities) or
            len(predictions) != len(ids)):
        raise ValueError("frozen batch output differs")
    return {
        "dataset": dataset,
        "seed": seed,
        "model_id": MODELS[dataset],
        "reference_version": reference_version,
        "threshold": threshold,
        "source_binding": source_binding,
        "evidence_terms_per_account": evidence_terms,
        "classification_labels_accessed": False,
        "caller_supplied_fact_matrix_accepted": False,
        "predictions": [{"account_id": uid, "probability": p,
                         "prediction": prediction}
                        for uid, p, prediction in zip(
                            ids, probabilities, predictions, strict=True)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--account-id", action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = predict(Path(__file__).resolve().parents[2], args.dataset,
                     args.seed, args.account_id)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(args.out)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
