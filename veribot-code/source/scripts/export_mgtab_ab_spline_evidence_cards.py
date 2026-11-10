"""Export full per-account source-to-logit cards for selected A/B MGTAB spline."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_mgtab_sparse_evidence_cards import locator
from gate0_shallow_tree import load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def export_seed(args: argparse.Namespace, seed: int) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if (manifest != saved["manifest"] or manifest["seed"] != seed
            or manifest["model"] != "ab_monotone_spline_with_content_projection"):
        raise ValueError("frozen A/B model manifest differs")
    for path, key in ((args.data_dir / "features.pt", "source_features_sha256"),
                      (args.structural_facts, "structural_facts_sha256"),
                      (args.neighbor_facts, "neighbor_facts_sha256"),
                      (directory / "projection.npz", "projection_artifact_sha256"),
                      (directory / "reference.json.gz", "reference_artifact_sha256")):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"source or artifact differs: {path}")
    x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
    x, names, source = attach_mgtab_structural(
        x, names, ids, source, args.data_dir, args.structural_facts)
    x, names, supports, source = attach_mgtab_neighbor(
        x, names, ids, source, args.data_dir, args.neighbor_facts)
    if source["files_sha256"] != manifest["files_sha256"] or source["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("source or split differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    train, test = split["train"], split["test"]
    scaler = StandardScaler().fit(content[train])
    normalized = scaler.transform(content)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
        normalized[train])
    projection_fingerprint = fingerprint(
        scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
    if projection_fingerprint != manifest["projection_fingerprint"]:
        raise ValueError("projection does not refit from published source")
    raw_values = np.column_stack((x[test], pca.transform(normalized[test])))
    reference = LabelBlindReference.load(directory / "reference.json.gz")
    model = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    if (reference.version != manifest["reference_version"]
            or model.reference_version != reference.version
            or reference.names != model.names):
        raise ValueError("reference and reasoner differ")
    evidence = transform_with_support(reference, raw_values, supports[test])
    basis = model._design(evidence)
    per_term = np.column_stack([
        basis[:, model._slices[name]] @ model.coef[model._slices[name]]
        for name in model.names])
    logits = float(model.intercept) + per_term.sum(axis=1)
    probabilities = model.predict_proba(evidence)
    if np.max(np.abs(probabilities - 1 / (1 + np.exp(-logits)))) > 1e-12:
        raise ValueError("per-term spline contributions do not rebuild probabilities")
    rows = saved["predictions"]
    if (len(rows) != len(test) or any(
            row["account_id"] != ids[int(index)] or row["label"] != int(y[index])
            for row, index in zip(rows, test))):
        raise ValueError("saved test population differs")
    difference = float(np.max(np.abs(probabilities - np.asarray(
        [row["probability"] for row in rows]))))
    if difference > 1e-12:
        raise ValueError("frozen test probability differs")
    active = [j for j, name in enumerate(model.names)
              if np.any(model.coef[model._slices[name]] > 0)]
    output = args.output_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for offset, account_index in enumerate(test):
                    terms = []
                    for j in active:
                        name = model.names[j]
                        sl = model._slices[name]
                        terms.append({
                            "name": name,
                            "source_locator": locator(j, name, int(account_index),
                                                      projection_fingerprint),
                            "source_value": float(raw_values[offset, j]),
                            "available": bool(evidence.available[offset, j]),
                            "upper_tail_score": float(evidence.upper[offset, j]),
                            "lower_tail_score": float(evidence.lower[offset, j]),
                            "reliability": float(evidence.reliability[offset, j]),
                            "selected_tail": model.tail[j],
                            "fixed_direction": int(model.sign[j]),
                            "knots": list(model.knots[name]),
                            "nonnegative_slopes": model.coef[sl].tolist(),
                            "logit_contribution": float(per_term[offset, j]),
                        })
                    group_contributions = {
                        group: float(sum(per_term[offset, model.names.index(term)]
                                         for term in members))
                        for group, members in model.groups.items()}
                    if any(abs(group_contributions[group] - rows[offset]["group_contributions"][group]) > 1e-12
                           for group in group_contributions):
                        raise ValueError("saved group contributions differ")
                    card = {"account_id": ids[int(account_index)],
                            "label": int(y[account_index]), "seed": seed,
                            "source_level": "published_feature_and_typed_edge_tensors",
                            "source_features_sha256": manifest["source_features_sha256"],
                            "reference_version": reference.version,
                            "selected_l2": manifest["selected_l2"],
                            "intercept": float(model.intercept), "terms": terms,
                            "group_contributions": group_contributions,
                            "logit": float(logits[offset]),
                            "probability": float(probabilities[offset]),
                            "threshold": float(manifest["threshold"]),
                            "prediction": int(probabilities[offset] >= manifest["threshold"])}
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
    return {"seed": seed, "test_accounts": len(test),
            "active_terms_per_account": len(active),
            "max_saved_probability_difference": difference,
            "cards_path": str(output), "cards_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {"dataset": "mgtab", "model": "ab_monotone_spline_with_content_projection",
              "scope": "all_fixed_test_account_numeric_source_to_logit",
              "runs": [export_seed(args, seed) for seed in (0, 1, 2)]}
    summary = args.output_dir / "evidence_cards_manifest.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
