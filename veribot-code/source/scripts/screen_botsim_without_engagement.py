"""Dev-only BotSim shortcut ablation for source-backed engagement scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import groups_for, rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path, required=True)
    parser.add_argument("--native-graph-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (audit.get("status") != "verified_full_population_against_raw_comments"
            or audit["cache_sha256"] != sha256_file(args.native_graph_facts)
            or audit["input_sha256"]["Users.csv"] != sha256_file(args.account_file)):
        raise ValueError("BotSim native source audit differs")
    variants = {
        "without_avg_post_score": {"avg_post_score"},
        "without_post_engagement_group": {"avg_post_score", "avg_upvote_ratio",
                                          "avg_comment_score"},
    }
    runs = []
    for seed in (0, 1, 2):
        selected = json.loads((args.selected_model_dir / f"seed{seed}" / "report.json")
                              .read_text(encoding="utf-8"))
        x, y, ids, names, split, source = load(
            "botsim", args.data_dir, args.account_file, seed, args.native_graph_facts)
        if (source["split_sha256"] != selected["manifest"]["split_sha256"]
                or source["files_sha256"] != selected["manifest"]["files_sha256"]
                or len(names) != 22):
            raise ValueError("selected model and source population differ")
        train, dev = split["train"], split["val"]
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        groups = groups_for("botsim", names)
        alternatives = {}
        for variant, removed in variants.items():
            keep = [j for j, name in enumerate(names) if name not in removed]
            vocabulary = [names[j] for j in keep]
            reduced_groups = {group: [name for name in members if name not in removed]
                              for group, members in groups.items()}
            reduced_groups = {group: members for group, members in reduced_groups.items()
                              if members}
            reference = LabelBlindReference(min_context_n=30).fit(
                x[train][:, keep], vocabulary)
            evidence = reference.transform(x[:, keep])
            model = SignConstrainedReasoner(reduced_groups, l2=0.01)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, dev))
            threshold = best_validation_threshold(probability, y[dev])
            alternatives[variant] = {
                "omitted_fact_names": sorted(removed),
                "retained_fact_count": len(vocabulary),
                "validation": evaluate(y[dev], probability, threshold),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "complete_validation": selected["validation"],
                     "alternatives": alternatives})
    result = {"dataset": "botsim", "scope": "validation_only_engagement_ablation",
              "test_predictions_accessed": False,
              "source_audit_sha256": sha256_file(args.source_audit),
              "selected_model_dir": str(args.selected_model_dir), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
