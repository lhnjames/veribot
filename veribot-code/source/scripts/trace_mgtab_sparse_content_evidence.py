"""Write one published-source -> evidence -> frozen-logit certificate."""

from __future__ import annotations

import argparse
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
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.model_dir / f"seed{args.seed}"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((folder / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    x, y, ids, names, split, current = load("mgtab", args.data_dir, None, args.seed)
    x, names, current = attach_mgtab_structural(
        x, names, ids, current, args.data_dir, args.structural_facts)
    x, names, supports, current = attach_mgtab_neighbor(
        x, names, ids, current, args.data_dir, args.neighbor_facts)
    if current["files_sha256"] != manifest["files_sha256"] or current["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("source or split differs")
    account = ids.index(args.account_id)
    if account not in set(split["test"].tolist()):
        raise ValueError("account is not in fixed test split")
    with np.load(folder / "model.npz", allow_pickle=False) as artifact:
        vocabulary = artifact["vocabulary"].astype(str).tolist()
        coefficient = artifact["coefficients"].astype(float)
        intercept = float(artifact["intercept"][0])
        threshold = float(artifact["threshold"][0])
        parameters = tuple(artifact[key] for key in
                           ("scaler_mean", "scaler_scale", "pca_mean", "pca_components"))
    if vocabulary[:86] != names or len(vocabulary) != 150:
        raise ValueError("vocabulary differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    scaler = StandardScaler().fit(content[split["train"]])
    normalized = scaler.transform(content)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=args.seed).fit(
        normalized[split["train"]])
    refit = (scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
    if (fingerprint(*refit) != manifest["projection_fingerprint"]
            or any(not np.array_equal(a, b) for a, b in zip(parameters, refit))):
        raise ValueError("saved projection does not refit")
    values = np.concatenate((x[account], pca.transform(normalized[account:account+1])[0]))
    reference = LabelBlindReference.load(folder / "reference.json.gz")
    if reference.version != manifest["reference_version"]:
        raise ValueError("reference differs")
    evidence = transform_with_support(reference, values[None, :], supports[account:account+1])
    upper = np.log1p(evidence.upper[0]) * evidence.reliability[0]
    lower = np.log1p(evidence.lower[0]) * evidence.reliability[0]
    upper_contribution = upper * coefficient[:150]
    lower_contribution = lower * coefficient[150:]
    terms = []
    for j, name in enumerate(vocabulary):
        if j < 20:
            locator = {"file": "features.pt", "row": account, "column": j}
        elif j < 62:
            locator = {"files": ["edge_index.pt", "edge_type.pt"],
                       "account_row": account, "fact_cache": str(args.structural_facts)
                       if j >= 34 else None}
        elif j < 86:
            locator = {"files": ["features.pt", "edge_index.pt", "edge_type.pt"],
                       "account_row": account, "fact_cache": str(args.neighbor_facts)}
        else:
            locator = {"file": "features.pt", "row": account,
                       "content_columns": [20, 787], "pca_component": j - 86,
                       "train_only_projection_sha256": manifest["projection_fingerprint"]}
        terms.append({"name": name, "source_locator": locator,
                      "source_value": float(values[j]),
                      "available": bool(evidence.available[0, j]),
                      "upper_tail_score": float(evidence.upper[0, j]),
                      "lower_tail_score": float(evidence.lower[0, j]),
                      "reliability": float(evidence.reliability[0, j]),
                      "upper_coefficient": float(coefficient[j]),
                      "lower_coefficient": float(coefficient[j + 150]),
                      "upper_logit_contribution": float(upper_contribution[j]),
                      "lower_logit_contribution": float(lower_contribution[j]),
                      "total_logit_contribution": float(upper_contribution[j]
                                                        + lower_contribution[j])})
    logit = float(intercept + np.sum(upper_contribution) + np.sum(lower_contribution))
    probability = float(expit(logit))
    recorded = next(row for row in saved["predictions"] if row["account_id"] == args.account_id)
    if abs(probability - recorded["probability"]) > 1e-12:
        raise ValueError("source-to-evidence logit does not match saved output")
    result = {"dataset": "mgtab", "seed": args.seed, "account_id": args.account_id,
              "label": int(y[account]), "source_level": "published_feature_and_typed_edge_tensors",
              "source_files_sha256": {Path(k).name: v for k, v in current["files_sha256"].items()
                                      if Path(k).parent == args.data_dir},
              "reference_version": reference.version,
              "model_artifact_sha256": sha256_file(folder / "model.npz"),
              "intercept": intercept, "terms": terms, "logit": logit,
              "probability": probability, "threshold": threshold,
              "prediction": int(probability >= threshold),
              "saved_probability_difference": abs(probability - recorded["probability"]),
              "top_absolute_terms": [entry["name"] for entry in
                                     sorted(terms, key=lambda item:
                                            abs(item["total_logit_contribution"]), reverse=True)[:10]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("account_id", "label", "prediction", "probability",
                       "saved_probability_difference", "top_absolute_terms")}, indent=2))


if __name__ == "__main__":
    main()
