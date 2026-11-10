"""Export complete per-account numeric evidence cards for frozen MGTAB tests."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def locator(index: int, name: str, account_index: int,
            projection_fingerprint: str) -> dict:
    if index < 20:
        return {"source_level": "published_profile_tensor_coordinate",
                "file": "features.pt", "row": account_index, "column": index}
    if index < 62:
        return {"source_level": "published_typed_edge_tensor",
                "files": ["edge_index.pt", "edge_type.pt"],
                "account_index": account_index, "fact_query": name}
    if index < 86:
        return {"source_level": "published_profile_and_typed_edge_tensors",
                "files": ["features.pt", "edge_index.pt", "edge_type.pt"],
                "account_index": account_index, "fact_query": name}
    return {"source_level": "published_averaged_tweet_tensor_projection",
            "file": "features.pt", "row": account_index,
            "content_columns_inclusive": [20, 787], "component": index - 86,
            "train_only_projection_fingerprint": projection_fingerprint}


def export_seed(args: argparse.Namespace, seed: int) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    predictions = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if manifest != predictions["manifest"] or manifest["seed"] != seed:
        raise ValueError("frozen report and prediction manifest differ")
    for path, key in ((args.data_dir / "features.pt", "source_features_sha256"),
                      (args.structural_facts, "structural_facts_sha256"),
                      (args.neighbor_facts, "neighbor_facts_sha256"),
                      (directory / "model.npz", "model_artifact_sha256"),
                      (directory / "reference.json.gz", "reference_artifact_sha256")):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"source or model differs: {path}")
    x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
    x, names, source = attach_mgtab_structural(
        x, names, ids, source, args.data_dir, args.structural_facts)
    x, names, supports, source = attach_mgtab_neighbor(
        x, names, ids, source, args.data_dir, args.neighbor_facts)
    if source["files_sha256"] != manifest["files_sha256"] or source["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("source population or split differs")
    with np.load(directory / "model.npz", allow_pickle=False) as saved:
        vocabulary = saved["vocabulary"].astype(str).tolist()
        coefficient = saved["coefficients"].astype(float)
        intercept = float(saved["intercept"][0])
        threshold = float(saved["threshold"][0])
        stored_parameters = tuple(saved[key] for key in
                                  ("scaler_mean", "scaler_scale", "pca_mean", "pca_components"))
        train_indices = saved["train_indices"]
    if (vocabulary[:86] != names or len(vocabulary) != 150
            or coefficient.shape != (300,) or not np.array_equal(train_indices, split["train"])):
        raise ValueError("frozen model vocabulary or training rows differ")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    scaler = StandardScaler().fit(content[split["train"]])
    normalized = scaler.transform(content)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
        normalized[split["train"]])
    parameters = (scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
    if (fingerprint(*parameters) != manifest["projection_fingerprint"]
            or any(not np.array_equal(a, b) for a, b in zip(parameters, stored_parameters))):
        raise ValueError("projection does not refit from published source")
    test = split["test"]
    raw_values = np.column_stack((x[test], pca.transform(normalized[test])))
    reference = LabelBlindReference.load(directory / "reference.json.gz")
    if reference.version != manifest["reference_version"] or reference.names != tuple(vocabulary):
        raise ValueError("frozen reference differs")
    evidence = transform_with_support(reference, raw_values, supports[test])
    upper_input = np.log1p(evidence.upper) * evidence.reliability
    lower_input = np.log1p(evidence.lower) * evidence.reliability
    upper_contribution = upper_input * coefficient[:150]
    lower_contribution = lower_input * coefficient[150:]
    totals = upper_contribution + lower_contribution
    logits = intercept + totals.sum(axis=1)
    probability = expit(logits)
    rows = predictions["predictions"]
    if (len(rows) != len(test) or any(
            row["account_id"] != ids[int(index)] or row["label"] != int(y[index])
            for row, index in zip(rows, test))):
        raise ValueError("test account order or labels differ")
    difference = float(np.max(np.abs(probability - np.asarray(
        [row["probability"] for row in rows]))))
    if difference > 1e-12:
        raise ValueError("full evidence fails to reconstruct saved probabilities")
    group_map = {**groups_for("mgtab", names),
                 "published_tweet_projections": vocabulary[86:]}
    group_indices = {group: np.asarray([vocabulary.index(term) for term in members])
                     for group, members in group_map.items()}
    active = np.flatnonzero((coefficient[:150] != 0) | (coefficient[150:] != 0))
    output = args.output_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for offset, account_index in enumerate(test):
                    card_terms = []
                    for j in active:
                        card_terms.append({
                            "name": vocabulary[int(j)],
                            "source_locator": locator(int(j), vocabulary[int(j)],
                                                      int(account_index),
                                                      manifest["projection_fingerprint"]),
                            "source_value": float(raw_values[offset, j]),
                            "available": bool(evidence.available[offset, j]),
                            "upper_tail_score": float(evidence.upper[offset, j]),
                            "lower_tail_score": float(evidence.lower[offset, j]),
                            "reliability": float(evidence.reliability[offset, j]),
                            "upper_coefficient": float(coefficient[j]),
                            "lower_coefficient": float(coefficient[j + 150]),
                            "upper_logit_contribution": float(upper_contribution[offset, j]),
                            "lower_logit_contribution": float(lower_contribution[offset, j]),
                            "total_logit_contribution": float(totals[offset, j]),
                        })
                    group_contributions = {group: float(totals[offset, indices].sum())
                                           for group, indices in group_indices.items()}
                    if any(abs(group_contributions[group] - rows[offset]["group_contributions"][group]) > 1e-12
                           for group in group_contributions):
                        raise ValueError("saved evidence groups differ")
                    card = {"account_id": ids[int(account_index)], "label": int(y[account_index]),
                            "model_seed": seed, "reference_version": reference.version,
                            "model_artifact_sha256": manifest["model_artifact_sha256"],
                            "source_features_sha256": manifest["source_features_sha256"],
                            "source_level": "published_feature_and_typed_edge_tensors",
                            "intercept": intercept, "active_terms": card_terms,
                            "group_contributions": group_contributions,
                            "logit": float(logits[offset]),
                            "probability": float(probability[offset]),
                            "threshold": threshold,
                            "prediction": int(probability[offset] >= threshold)}
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
    return {"seed": seed, "test_accounts": len(test),
            "active_terms_per_account": len(active),
            "maximum_saved_probability_difference": difference,
            "cards_path": str(output), "cards_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {"dataset": "mgtab", "scope": "all_fixed_test_account_numeric_source_to_logit",
              "runs": [export_seed(args, seed) for seed in (0, 1, 2)]}
    summary = args.output_dir / "evidence_cards_manifest.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
