"""Screen a fixed monotone spline evidence reasoner on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import groups_for, rows
from train_v3_twibot20 import groups as twibot_groups, metric, threshold_on_dev


def run_tabular(args: argparse.Namespace, seed: int) -> dict:
    x, y, ids, names, split, manifest = load(
        args.dataset, args.data_dir, args.account_file, seed, args.native_graph_facts)
    train, validation = split["train"], split["val"]
    reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
    a, b = train_test_split(train, test_size=0.5, stratify=y[train], random_state=seed)
    if set(a) & set(b):
        raise AssertionError("direction and coefficient split overlap")
    train_evidence = reference.transform(x[train])
    by_index = {int(idx): position for position, idx in enumerate(train)}
    a_local = np.asarray([by_index[int(idx)] for idx in a])
    b_local = np.asarray([by_index[int(idx)] for idx in b])
    reasoner = MonotoneSplineReasoner(groups_for(args.dataset, names), l2=0.01)
    reasoner.fit(rows(train_evidence, a_local), y[a], rows(train_evidence, b_local), y[b])
    probability = reasoner.predict_proba(reference.transform(x[validation]))
    threshold = best_validation_threshold(probability, y[validation])
    return {"manifest": {**manifest, "model": "monotone_interval_spline_evidence",
                         "screen_scope": "validation_only", "test_predictions_accessed": False,
                         "reference_version": reference.version,
                         "direction_split_size": len(a), "coefficient_split_size": len(b),
                         "knots_from": "coefficient_train_values_without_labels",
                         "threshold": threshold},
            "reasoner": reasoner.manifest(),
            "validation": evaluate(y[validation], probability, threshold),
            "validation_rows": [{"account_id": ids[int(i)], "label": int(y[i]),
                                 "probability": float(p)}
                                for i, p in zip(validation, probability)],
            "reference": reference}


def run_twibot(args: argparse.Namespace, seed: int) -> dict:
    with np.load(args.facts, allow_pickle=False) as cached:
        x = cached["values"].astype(float)
        y = cached["labels"].astype(int)
        ids = cached["ids"].astype(str)
        split = cached["split"].astype(str)
        contexts = cached["contexts"].astype(str)
        names = cached["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "dev")
    if (len(train), len(validation)) != (8278, 2365):
        raise ValueError("official TwiBot-20 train/dev split differs")
    reference = LabelBlindReference(min_context_n=30).fit(
        x[train], names, contexts=contexts[train])
    a, b = train_test_split(train, test_size=0.5, stratify=y[train], random_state=seed)
    model = MonotoneSplineReasoner(twibot_groups(names), l2=0.01)
    model.fit(reference.transform(x[a], contexts=contexts[a]), y[a],
              reference.transform(x[b], contexts=contexts[b]), y[b])
    probability = model.predict_proba(reference.transform(
        x[validation], contexts=contexts[validation]))
    threshold = threshold_on_dev(probability, y[validation])
    source = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    return {"manifest": {"dataset": "twibot20", "seed": seed,
                         "model": "monotone_interval_spline_evidence",
                         "screen_scope": "validation_only", "test_predictions_accessed": False,
                         "official_split": True, "facts_sha256": sha256_file(args.facts),
                         "archive_sha256": source["archive_sha256"],
                         "reference_version": reference.version,
                         "direction_split_size": len(a), "coefficient_split_size": len(b),
                         "knots_from": "coefficient_train_values_without_labels",
                         "threshold": threshold},
            "reasoner": model.manifest(),
            "validation": metric(y[validation], probability, threshold),
            "validation_rows": [{"account_id": str(ids[i]), "label": int(y[i]),
                                 "probability": float(p)}
                                for i, p in zip(validation, probability)],
            "reference": reference}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("twibot20", "mgtab", "botsim"), required=True)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-graph-facts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    if args.dataset == "twibot20" and args.facts is None:
        parser.error("TwiBot-20 requires --facts")
    if args.dataset != "twibot20" and args.data_dir is None:
        parser.error("MGTAB/BotSim require --data-dir")
    summaries = []
    for seed in args.seeds:
        result = run_twibot(args, seed) if args.dataset == "twibot20" else run_tabular(args, seed)
        out = args.output / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        reference = result.pop("reference")
        reference_path = out / "reference.json.gz"
        reference.save(reference_path)
        result["manifest"]["reference_artifact_sha256"] = sha256_file(reference_path)
        (out / "validation_report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summaries.append({"seed": seed, "validation": result["validation"],
                          "report": str(out / "validation_report.json")})
    print(json.dumps({"dataset": args.dataset, "screen_scope": "validation_only",
                      "runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
