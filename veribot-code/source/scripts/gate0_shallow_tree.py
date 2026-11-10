"""Gate 0: fixed depth-4 shortcut baseline on published, inspectable fields.

This is a diagnostic control, not the VeriBot evidence-bottleneck classifier.
It uses the same account population and split algorithms as train_v2_detector,
fits only on train labels, selects a threshold on validation, then reports test.
MGTAB profile coordinates remain indexed in model artifacts; the source paper's
Table A5 maps those indices to names, while the local release only supplies a
normalized tensor. Relation degrees can be checked against published edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_adapter import FEATURE_FIELDS as FOX8_FIELDS
from verifiable_bot.botsim_adapter import FEATURE_FIELDS as BOTSIM_FIELDS
from verifiable_bot.botsim_native_facts import FACT_NAMES as BOTSIM_GRAPH_FIELDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(*chunks: bytes) -> str:
    """Use the same length-prefixed fingerprint as the HypBot paired exporter."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "little"))
        digest.update(chunk)
    return digest.hexdigest()


def tabular_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for label in (1, 0):
        indices = np.where(labels == label)[0].copy()
        rng.shuffle(indices)
        n_train = int(0.70 * len(indices))
        n_val = int(0.15 * len(indices))
        parts["train"].append(indices[:n_train])
        parts["val"].append(indices[n_train:n_train + n_val])
        parts["test"].append(indices[n_train + n_val:])
    return {key: np.concatenate(values) for key, values in parts.items()}


def mgtab_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    train, rest = train_test_split(np.arange(len(labels)), test_size=0.30,
                                   stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=1 / 3,
                                 stratify=labels[rest], random_state=seed)
    return {"train": train, "val": val, "test": test}


def relation_degrees(edge_index: np.ndarray, edge_type: np.ndarray,
                     n_accounts: int) -> tuple[np.ndarray, list[str]]:
    relations = np.unique(edge_type)
    if not np.array_equal(relations, np.arange(len(relations))):
        raise ValueError("MGTAB relation type indices are not consecutive from zero")
    features = []
    names = []
    relation_names = ("followers", "friends", "mention", "reply", "quoted", "url", "hashtag")
    if len(relations) != len(relation_names):
        raise ValueError("MGTAB relation count differs from the official seven-type release")
    for relation in relations:
        selected = edge_type == relation
        for endpoint, direction in ((0, "out"), (1, "in")):
            counts = np.bincount(edge_index[endpoint, selected], minlength=n_accounts)
            features.append(np.log1p(counts).astype(np.float32))
            names.append(f"published_{relation_names[int(relation)]}_{direction}_degree_log1p")
    return np.stack(features, axis=1), names


def load(dataset: str, data_dir: Path, account_file: Path | None,
         seed: int, native_graph_facts: Path | None = None
         ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, np.ndarray], dict]:
    if dataset == "mgtab":
        paths = [data_dir / name for name in ("features.pt", "labels_bot.pt",
                                              "edge_index.pt", "edge_type.pt")]
        x_raw = torch.load(paths[0], map_location="cpu", weights_only=True).numpy()
        labels = torch.load(paths[1], map_location="cpu", weights_only=True).numpy().astype(int)
        edge_index = torch.load(paths[2], map_location="cpu", weights_only=True).numpy()
        edge_type = torch.load(paths[3], map_location="cpu", weights_only=True).numpy()
        if x_raw.shape[1] < 20 or edge_index.shape[0] != 2 or edge_index.shape[1] != len(edge_type):
            raise ValueError("MGTAB tensor shapes differ from the audited release")
        graph, graph_names = relation_degrees(edge_index, edge_type, len(labels))
        x = np.column_stack((x_raw[:, :20], graph))
        names = [f"published_profile_coordinate_{i:02d}" for i in range(20)] + graph_names
        ids = [str(i) for i in range(len(labels))]
        split = mgtab_split(labels, seed)
    else:
        paths = [data_dir / "features.csv", data_dir / "labels.csv"]
        frame = pd.read_csv(paths[0])
        labels = pd.read_csv(paths[1])["label"].to_numpy(dtype=int)
        expected = list(FOX8_FIELDS if dataset == "fox8" else BOTSIM_FIELDS)
        if dataset == "fox8":
            if list(frame.columns) != ["user_id", *expected]:
                raise ValueError("Fox8 fields differ from audited preprocessing")
            ids = frame["user_id"].astype(str).tolist()
            x = frame[expected].to_numpy(dtype=np.float32)
        else:
            if account_file is None:
                raise ValueError("BotSim requires --account-file pointing to Users.csv")
            paths.append(account_file)
            if list(frame.columns) != expected:
                raise ValueError("BotSim fields differ from audited preprocessing")
            users = pd.read_csv(account_file)
            ids = users["user_id"].astype(str).tolist()
            x = frame[expected].to_numpy(dtype=np.float32)
        names = expected
        if native_graph_facts is not None:
            if dataset != "botsim":
                raise ValueError("native graph fact cache currently applies only to BotSim")
            graph_manifest_path = native_graph_facts.with_name("manifest.json")
            graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
            if graph_manifest["facts_sha256"] != sha256_file(native_graph_facts):
                raise ValueError("native graph fact cache differs from its source manifest")
            if graph_manifest["input_sha256"].get(str(account_file.resolve())) != sha256_file(account_file):
                raise ValueError("BotSim Users.csv differs from native graph source manifest")
            with np.load(native_graph_facts, allow_pickle=False) as graph_cache:
                graph_ids = graph_cache["ids"].astype(str).tolist()
                graph_names = graph_cache["names"].astype(str).tolist()
                graph_values = graph_cache["values"].astype(np.float32)
            if graph_ids != ids or graph_values.shape != (len(ids), len(graph_names)):
                raise ValueError("native graph facts differ from BotSim account order or shape")
            if tuple(graph_names) != BOTSIM_GRAPH_FIELDS or not np.isfinite(graph_values).all():
                raise ValueError("native graph facts have unexpected fields or values")
            if len(set(graph_names)) != len(graph_names) or set(graph_names) & set(names):
                raise ValueError("native graph fact names are duplicated")
            x = np.column_stack((x, graph_values))
            names = names + graph_names
            paths.extend((native_graph_facts, graph_manifest_path))
        split = tabular_split(labels, seed)
    if len(x) != len(labels) or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("account, feature and label rows are not one-to-one")
    if not np.isfinite(x).all() or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("nonfinite features or nonbinary labels")
    sets = [set(map(int, split[k])) for k in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("split overlap")
    if sum(map(len, sets)) != len(labels):
        raise ValueError("split does not cover population")
    manifest = {
        "dataset": dataset, "seed": seed,
        "files_sha256": {str(path.resolve()): sha256_file(path) for path in paths},
        "split_sha256": hash_bytes(json.dumps(
            {key: [ids[int(i)] for i in split[key]] for key in ("train", "val", "test")},
            sort_keys=True).encode()),
        "feature_names": names, "population": len(labels),
        "split_sizes": {key: len(value) for key, value in split.items()},
        "provenance_level": "published_tensor_and_edges" if dataset == "mgtab"
                            else ("raw_records_and_native_reply_graph"
                                  if native_graph_facts is not None else "raw_record_derived_feature_table"),
    }
    return x, labels, ids, names, split, manifest


def best_validation_threshold(prob: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.5, prob, np.nextafter(1.0, 2.0)])
    scores = np.array([f1_score(labels, prob >= threshold, average="macro",
                                zero_division=0) for threshold in candidates])
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(candidates[tied[np.argmin(abs(candidates[tied] - 0.5))]])


def evaluate(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    pred = probability >= threshold
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        "errors": int(np.sum(pred != y)),
    }


def run(args: argparse.Namespace, seed: int) -> dict:
    x, y, ids, names, split, manifest = load(args.dataset, args.data_dir,
                                               args.account_file, seed,
                                               args.native_graph_facts)
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=args.min_leaf,
                                   random_state=seed)
    model.fit(x[split["train"]], y[split["train"]])
    val_probability = model.predict_proba(x[split["val"]])[:, 1]
    threshold = best_validation_threshold(val_probability, y[split["val"]])
    report = {"manifest": {**manifest, "max_depth": 4,
                           "min_samples_leaf": args.min_leaf,
                           "threshold_selected_on": "validation_macro_f1",
                           "threshold": threshold},
              "validation": evaluate(y[split["val"]], val_probability, threshold),
              "test": {}, "tree": export_text(model, feature_names=names),
              "feature_importances": {name: float(value) for name, value in
                                      zip(names, model.feature_importances_) if value > 0}}
    test_probability = model.predict_proba(x[split["test"]])[:, 1]
    report["test"] = evaluate(y[split["test"]], test_probability, threshold)
    out = args.output / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": report["manifest"],
        "predictions": [{"account_id": ids[int(i)], "label": int(y[i]),
                         "probability": float(p), "split": "test"}
                        for i, p in zip(split["test"], test_probability)],
    }, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "test": report["test"], "report": str(out / "report.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fox8", "botsim", "mgtab"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-graph-facts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--min-leaf", type=int, default=20)
    args = parser.parse_args()
    if args.min_leaf < 1:
        parser.error("--min-leaf must be at least one")
    reports = [run(args, seed) for seed in args.seeds]
    print(json.dumps({"dataset": args.dataset, "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
