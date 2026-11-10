"""Dev-only sensitivity of reference tails to labeled-bot mixture fractions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_mgtab_reference_population import design
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if audit["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("published source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    fractions = (0.0, 0.25, 0.50, 0.75)
    reference_size = 2000
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        train, val = split["train"], split["val"]
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
            normalized[train])
        matrix = np.column_stack((x, pca.transform(normalized)))
        vocabulary = names + [f"published_tweet_projection_{j:03d}" for j in range(64)]
        human = train[y[train] == 0]
        bot = train[y[train] == 1]
        rng = np.random.default_rng(20261103 + seed)
        shuffled_human = rng.permutation(human)
        shuffled_bot = rng.permutation(bot)
        rows = []
        for fraction in fractions:
            bot_count = int(reference_size * fraction)
            human_count = reference_size - bot_count
            if bot_count > len(bot) or human_count > len(human):
                raise ValueError("requested diagnostic reference exceeds train class count")
            selected = np.concatenate((shuffled_human[:human_count],
                                       shuffled_bot[:bot_count]))
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[selected], vocabulary)
            evidence = design(reference, matrix, supports)
            model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                       max_iter=2000, random_state=seed).fit(
                evidence[train], y[train])
            p = expit(float(model.intercept_[0]) + evidence[val] @ model.coef_.ravel())
            threshold = best_validation_threshold(p, y[val])
            rows.append({"reference_bot_fraction": fraction,
                         "reference_accounts": reference_size,
                         "reference_selection_uses_train_labels": True,
                         "reference_version": reference.version,
                         "validation": evaluate(y[val], p, threshold),
                         "threshold": threshold})
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "fractions": rows})
    result = {"dataset": "mgtab", "scope": "validation_only_reference_contamination_diagnostic",
              "test_predictions_accessed": False,
              "fractions_are_diagnostic_not_candidate_model_selection": True,
              "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
              "structural_facts_sha256": sha256_file(args.structural_facts),
              "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
              "source_audit_sha256": sha256_file(args.source_audit),
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{ "seed": run["seed"],
                        "results": [{"bot_fraction": row["reference_bot_fraction"],
                                     "validation": row["validation"]}
                                    for row in run["fractions"]]}
                      for run in runs], indent=2))


if __name__ == "__main__":
    main()
