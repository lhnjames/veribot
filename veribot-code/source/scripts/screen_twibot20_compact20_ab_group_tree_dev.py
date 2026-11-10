"""Dev-only fixed group-tree capacity screen over the same 20 TwiBot-20 facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lightgbm import LGBMClassifier
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIG = {"n_estimators": 250, "learning_rate": .05, "num_leaves": 7,
          "max_depth": 3, "min_child_samples": 20, "reg_lambda": 1.0}


def design(batch, signs: np.ndarray) -> np.ndarray:
    x = np.column_stack((np.log1p(batch.upper) * batch.reliability,
                         np.log1p(batch.lower) * batch.reliability))
    return x * (signs != 0)


def source_groups(names: list[str]) -> list[list[int]]:
    index = {name: j for j, name in enumerate(names)}
    groups = []
    for members in GROUPS.values():
        base = [index[name] for name in members]
        groups.append(base + [j + len(names) for j in base])
    if sorted(j for group in groups for j in group) != list(range(2 * len(names))):
        raise ValueError("Source groups do not partition 40 evidence tails")
    return groups


def fit_tree(x_b: np.ndarray, y_b: np.ndarray, groups: list[list[int]],
             signs: np.ndarray, constrained: bool, seed: int) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary", n_jobs=4, random_state=seed,
        deterministic=True, force_col_wise=True, verbosity=-1,
        monotone_constraints=signs.astype(int).tolist() if constrained else [0] * len(signs),
        monotone_constraints_method="advanced",
        interaction_constraints=groups,
        **CONFIG)
    model.fit(x_b, y_b)
    return model


def score_model(model: LGBMClassifier, x_dev: np.ndarray, y_dev: np.ndarray,
                groups: list[list[int]]) -> dict:
    single_group, tree_counts = all_trees_one_group(model, groups)
    if not single_group:
        raise ValueError("A tree crosses source groups")
    raw = np.asarray(model.booster_.predict(x_dev, raw_score=True))
    probability = expit(raw)
    threshold = threshold_on_dev(probability, y_dev)
    zero = np.asarray(model.booster_.predict(np.zeros_like(x_dev), raw_score=True))
    effects = []
    for group in groups:
        masked = x_dev.copy()
        masked[:, group] = 0.0
        without = np.asarray(model.booster_.predict(masked, raw_score=True))
        effects.append(raw - without)
    reconstruction = float(np.max(np.abs(raw - zero - sum(effects))))
    if reconstruction > 1e-9:
        raise ValueError("Group effects do not replay the frozen logits")
    return {"threshold": threshold,
            "dev": metric(y_dev, probability, threshold),
            "all_trees_single_group": single_group,
            "tree_counts_by_group": dict(zip(GROUPS, [tree_counts.get(j, 0)
                                                       for j in range(len(GROUPS))], strict=True)),
            "max_dev_logit_group_reconstruction_error": reconstruction}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--linear-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    linear = json.loads(args.linear_screen.read_text(encoding="utf-8"))
    if sha256(args.facts) != compact["base_facts_sha256"] or \
            sha256(args.reference) != compact["reference_artifact_sha256"] or \
            linear["compact_sha256"] != sha256(args.compact):
        raise ValueError("Source, reference or prior screen differs")
    with np.load(args.facts, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if len(names) != 20 or set(names) != set().union(*map(set, GROUPS.values())):
        raise ValueError("20-fact vocabulary differs")
    sample_count = values[:, all_names.index("tweet_sample_count")]
    if not np.isfinite(sample_count[train]).all() or not np.isfinite(sample_count[dev]).all():
        raise ValueError("Train/dev tweet support is incomplete")
    full = LabelBlindReference.load(args.reference).transform(values, contexts=contexts)
    keep = [all_names.index(name) for name in names]
    groups = source_groups(names)
    rows = []
    for support_aware in (False, True):
        batch = evidence_subset(full, keep, names, sample_count, support_aware)
        for seed in (0, 1, 2):
            a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                    random_state=1000 + seed)
            signs = DualTailEvidenceReasoner._select_tail_signs(take(batch, a), labels[a])
            signs = np.r_[signs[:, 0], signs[:, 1]]
            x_b = design(take(batch, b), signs)
            x_dev = design(take(batch, dev), signs)
            constrained = fit_tree(x_b, labels[b], groups, signs, True, seed)
            free = fit_tree(x_b, labels[b], groups, signs, False, seed)
            row = {"support_aware": support_aware, "ab_seed": seed,
                   "a_n": len(a), "b_n": len(b), "dev_n": len(dev),
                   "a_signs_by_fact": {name: {"upper": int(signs[j]),
                                              "lower": int(signs[j + len(names)])}
                                       for j, name in enumerate(names)},
                   "constrained_group_tree": score_model(
                       constrained, x_dev, labels[dev], groups),
                   "same_b_unconstrained_group_tree": score_model(
                       free, x_dev, labels[dev], groups)}
            rows.append(row)
            print(json.dumps({"support_aware": support_aware, "seed": seed,
                              "constrained_macro_f1": row["constrained_group_tree"]["dev"]["macro_f1"],
                              "free_macro_f1": row["same_b_unconstrained_group_tree"]["dev"]["macro_f1"]}),
                  flush=True)
    report = {
        "scope": "fixed-config TwiBot-20 20-fact A/B group-tree dev-only capacity screen",
        "candidate_test_evaluated": False,
        "facts_sha256": sha256(args.facts), "reference_sha256": sha256(args.reference),
        "compact_sha256": sha256(args.compact),
        "linear_screen_sha256": sha256(args.linear_screen),
        "config_fixed_before_this_screen": CONFIG,
        "method": "A fixes each dual-tail monotone direction; B fits group-restricted nonlinear trees; all tree splits remain inside one source group; three fixed A/B seeds",
        "support_rule": "only six sampled-tweet facts weighted by n/(n+20) when support_aware",
        "comparison_note": "The unconstrained tree sees the same B rows and the same A-selected nonzero tail slots; only monotonicity differs. Threshold selection and reported dev metrics reuse the same dev accounts.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
