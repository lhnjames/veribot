"""Dev-only 2-per-group evidence-concept capacity screen across graph datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def selected_facts(args: argparse.Namespace, seed: int,
                   report: dict) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    x, y, ids, names, split, source = load(
        args.dataset, args.data_dir, args.account_file, seed,
        args.native_graph_facts)
    if args.dataset == "mgtab":
        if args.structural_facts is None or args.neighbor_facts is None:
            raise ValueError("MGTAB requires source-audited structural and neighbor facts")
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        if len(names) != 86 or supports is None:
            raise ValueError("MGTAB source vocabulary differs")
        with np.load(args.selected_model_dir / f"seed{seed}" / "projection.npz",
                     allow_pickle=False) as artifact:
            scaler_mean = artifact["scaler_mean"]
            scaler_scale = artifact["scaler_scale"]
            pca_mean = artifact["pca_mean"]
            components = artifact["pca_components"]
            train_indices = artifact["train_indices"]
            added = artifact["feature_names"].astype(str).tolist()
        if (not np.array_equal(train_indices, split["train"])
                or len(added) != 64
                or fingerprint(scaler_mean, scaler_scale, pca_mean, components) !=
                report["manifest"]["projection_fingerprint"]):
            raise ValueError("MGTAB train-only projection differs")
        published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                               weights_only=True).numpy().astype(np.float64)
        # Match StandardScaler.transform followed by sklearn PCA.transform.
        # PCA subtracts the projected mean after the matrix product; centering
        # first can move a borderline validation account across a threshold.
        normalized = published[:, 20:].copy()
        normalized -= scaler_mean
        normalized /= scaler_scale
        projected = normalized @ components.T
        projected -= pca_mean.reshape(1, -1) @ components.T
        x = np.column_stack((x, projected))
        names += added
        extra = {"supports": supports}
    elif args.dataset == "fox8":
        if args.fox8_native_facts is None:
            raise ValueError("Fox8 native interaction facts required")
        with np.load(args.fox8_native_facts, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("Fox8 native account order differs")
            x = np.column_stack((x, cache["values"].astype(float)))
            names += cache["names"].astype(str).tolist()
        extra = {}
    else:
        if args.native_graph_facts is None or len(names) != 22:
            raise ValueError("BotSim native reply facts required")
        extra = {}
    if (source["split_sha256"] != report["manifest"]["split_sha256"]
            or tuple(names) != tuple(report["reasoner"]["directions"])):
        raise ValueError("facts, split or selected model vocabulary differ")
    return x, y, names, {**split, **extra}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mgtab", "fox8", "botsim"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-graph-facts", type=Path)
    parser.add_argument("--structural-facts", type=Path)
    parser.add_argument("--neighbor-facts", type=Path)
    parser.add_argument("--fox8-native-facts", type=Path)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path,
                        help="Optional prespecified source-group partition; validation only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    strengths = ((1e-5, 1e-4, 1e-3) if args.dataset == "mgtab"
                 else (1e-4, 1e-3, 1e-2))
    runs = []
    for seed in args.seeds:
        directory = args.selected_model_dir / f"seed{seed}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        matrix, y, names, split = selected_facts(args, seed, report)
        reference = LabelBlindReference.load(directory / "reference.json.gz")
        if reference.version != report["manifest"]["reference_version"]:
            raise ValueError("selected reference artifact differs")
        train, dev = split["train"], split["val"]
        if args.dataset == "mgtab":
            atomic_train = transform_with_support(reference, matrix[train],
                                                  split["supports"][train])
            atomic_dev = transform_with_support(reference, matrix[dev],
                                                split["supports"][dev])
        else:
            atomic_train = reference.transform(matrix[train])
            atomic_dev = reference.transform(matrix[dev])
        groups = (json.loads(args.groups_json.read_text(encoding="utf-8"))
                  if args.groups_json else report["reasoner"]["groups"])
        compact_train, winners_train = collapse_group_tails(atomic_train, groups)
        compact_dev, winners_dev = collapse_group_tails(atomic_dev, groups)
        if (compact_train.reference_version != compact_dev.reference_version
                or len(compact_train.names) > 20):
            raise ValueError("compact evidence contract differs")
        concept_groups = {group: [f"{group}:high_tail_max",
                                  f"{group}:low_tail_max"]
                          for group in groups}
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        alternatives = {}
        for strength in strengths:
            model = MonotoneSplineReasoner(concept_groups, l2=strength)
            model.fit(rows(compact_train, a), y[train][a],
                      rows(compact_train, b), y[train][b])
            probability = model.predict_proba(compact_dev)
            threshold = best_validation_threshold(probability, y[dev])
            alternatives[str(strength)] = {
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
                "nonzero_slopes": int(np.count_nonzero(model.coef > 0)),
            }
        winner_names = {concept: {names[int(j)]: int(np.sum(index == j))
                                   for j in np.unique(index) if j >= 0}
                        for concept, index in winners_dev.items()}
        runs.append({"seed": seed,
                     "split_sha256": report["manifest"]["split_sha256"],
                     "atomic_fact_count": len(names),
                     "concept_count": len(compact_train.names),
                     "concept_names": compact_train.names,
                     "bottleneck_version": compact_train.reference_version,
                     "selected_full_validation": report["validation"],
                     "winning_atomic_fact_counts_on_dev": winner_names,
                     "alternatives": alternatives})
    result = {"dataset": args.dataset,
              "scope": "validation_only_2_per_source_group_concept_capacity",
              "test_predictions_accessed": False,
              "prior_test_results_already_observed": True,
              "selected_model_dir": str(args.selected_model_dir),
              "groups_json_sha256": (sha256_file(args.groups_json)
                                     if args.groups_json else None),
              "groups": groups,
              "concept_formula": "max_available(log1p(atomic_tail_score)*atomic_reliability)",
              "l2_candidates": strengths, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "runs": [
        {"seed": row["seed"], "concept_count": row["concept_count"],
         "alternatives": {key: value["validation"]
                          for key, value in row["alternatives"].items()}}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
