"""Validation-only signal check for published MGTAB averaged-tweet vectors.

The logistic probes are diagnostics, not the evidence-bottleneck model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("published MGTAB feature tensor changed")
    content = published[:, 20:]
    runs = []
    for seed in (0, 1, 2):
        facts, y, ids, names, split, source = load(
            "mgtab", args.data_dir, None, seed)
        if facts.shape != (10199, 34):
            raise ValueError("MGTAB base facts changed")
        outputs = {}
        for label, x in (("profile_graph_34", facts),
                         ("content_768", content),
                         ("all_published_802", np.column_stack((facts, content)))):
            scaler = StandardScaler().fit(x[split["train"]])
            train = scaler.transform(x[split["train"]])
            val = scaler.transform(x[split["val"]])
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            model.fit(train, y[split["train"]])
            probability = model.predict_proba(val)[:, 1]
            threshold = best_validation_threshold(probability, y[split["val"]])
            outputs[label] = {
                "validation": evaluate(y[split["val"]], probability, threshold),
                "threshold": threshold,
                "converged": bool(model.n_iter_[0] < model.max_iter),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     **outputs})
    result = {
        "dataset": "mgtab", "scope": "validation_only_diagnostic",
        "test_predictions_accessed": False,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "content_source_level": "published_averaged_tweet_vector_tensor",
        "interpretation": "linear signal upper-bound probe; not an interpretable evidence model",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"], "runs": [
        {"seed": row["seed"], **{name: row[name]["validation"] for name in
                                  ("profile_graph_34", "content_768",
                                   "all_published_802")}}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
