"""Fit the same A/B group tree with split-visible sampled graph sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lightgbm import Booster
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import design, fit_tree, score_model, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import sha256
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--full-graph-screen", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed0-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    full_screen = json.loads(args.full_graph_screen.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    if (sha256(args.facts) != manifest["fact_cache_sha256"]
            or manifest["test_accounts_or_features_in_cache"]
            or len(full_screen["rows"]) != 3):
        raise ValueError("Split-visible cache or prior protocol differs")
    with np.load(args.facts, allow_pickle=False) as saved:
        values = saved["values"].astype(float)
        labels = saved["labels"].astype(int)
        split = saved["split"].astype(str)
        contexts = saved["contexts"].astype(str)
        names_all = saved["names"].astype(str).tolist()
    if set(split) != {"train", "dev"}:
        raise ValueError("Split-visible screen must contain train/dev only")
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if len(names) != 20 or set(names) != set().union(*map(set, GROUPS.values())):
        raise ValueError("20-fact evidence vocabulary differs")
    reference = LabelBlindReference(min_context_n=30, support_prior=0).fit(
        values[train], names_all, contexts[train])
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    reference.save(args.reference)
    keep = [names_all.index(name) for name in names]
    batch = evidence_subset(
        reference.transform(values, contexts=contexts), keep, names,
        values[:, names_all.index("tweet_sample_count")], True)
    groups = source_groups(names)
    rows = []
    for seed in (0, 1, 2):
        a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                random_state=1000 + seed)
        tail = DualTailEvidenceReasoner._select_tail_signs(take(batch, a), labels[a])
        signs = np.r_[tail[:, 0], tail[:, 1]]
        xb = design(take(batch, b), signs)
        xd = design(take(batch, dev), signs)
        model = fit_tree(xb, labels[b], groups, signs, True, seed)
        score = score_model(model, xd, labels[dev], groups)
        if seed == 0:
            args.seed0_model.parent.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(args.seed0_model))
            frozen = Booster(model_file=str(args.seed0_model))
            if np.max(np.abs(np.asarray(frozen.predict(xd, raw_score=True))
                             - model.booster_.predict(xd, raw_score=True))) > 1e-12:
                raise ValueError("Split-visible seed0 model reload differs")
        prior = next(row for row in full_screen["rows"] if row["ab_seed"] == seed)
        rows.append({
            "ab_seed": seed, "train_a": len(a), "train_b": len(b),
            "dev_accounts": len(dev), "dev": score["dev"],
            "threshold": score["threshold"],
            "group_additivity_error": score["max_dev_logit_group_reconstruction_error"],
            "tree_counts_by_group": score["tree_counts_by_group"],
            "tail_directions_by_fact": {
                name: {"upper": int(tail[j, 0]), "lower": int(tail[j, 1])}
                for j, name in enumerate(names)},
            "full_graph_transductive_dev": prior["all_provided_sample_graph_dev"],
        })
        print(json.dumps({"seed": seed,
                          "split_visible_macro_f1": score["dev"]["macro_f1"],
                          "transductive_macro_f1": prior["all_provided_sample_graph_dev"]["macro_f1"]}),
              flush=True)
    report = {
        "scope": "TwiBot-20 same 20-fact A/B group-tree development screen with split-visible graph declarations",
        "candidate_test_evaluated": False,
        "split_visible_facts_sha256": sha256(args.facts),
        "split_visible_manifest_sha256": sha256(args.facts.with_suffix(".json")),
        "reference_sha256": sha256(args.reference),
        "seed0_model_sha256": sha256(args.seed0_model),
        "compact_sha256": sha256(args.compact),
        "transductive_comparison_sha256": sha256(args.full_graph_screen),
        "protocol": "train facts use train+support declarations; dev facts use train+support+dev declarations; no test rows or declarations enter this cache; fit label-blind reference on train rows, choose directions on A, fit fixed source-group monotone tree on B, choose and evaluate threshold on official dev",
        "rows": rows,
        "interpretation": "This dev-only comparison isolates graph source visibility but still shares dev accounts between threshold selection and reported metric. It is not an independent future-population estimate or a test score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
