"""Train/dev screen: both evidence tails plus a neutral group-additive boundary.

The 16 named facts and source groups remain fixed. Each fact contributes two
explicit tail variables, whose signs are estimated only on direction half A.
The disjoint half B fits the group-restricted monotone model; roles are swapped
for a second fold. No fixed-test labels or predictions enter this screen.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _model, _source_groups
from lock_group_additive_schema_v2_candidates import OUT as LOCK_DIR
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = LOCK_DIR / "selection_lock.json"
OUT = ROOT / "output/gate1_schema_v2_dual_tail_neutral_dev/dev_only.json"
RATIOS = (0.0, 0.02, 0.1, 0.5)


def _context(facts: np.ndarray, names: list[str], run: dict) -> np.ndarray | None:
    if run["reference_context_rule"] == "pooled_unlabeled_training_reference":
        return None
    if run["reference_context_rule"] != \
            "unlabeled_training_tertiles_with_global_fallback":
        raise ValueError("unknown frozen reference rule")
    cuts = np.asarray(run["train_only_context_quantiles"])
    return np.asarray([f"activity_{i}" for i in np.searchsorted(
        cuts, facts[:, names.index(run["reference_context_fact"])], side="right")])


def _design(batch) -> np.ndarray:
    values = np.stack((batch.upper, batch.lower), axis=2).reshape(len(batch.upper), -1)
    reliability = np.repeat(batch.reliability, 2, axis=1)
    return np.log1p(values) * reliability


def _signs(batch, labels: np.ndarray) -> list[int]:
    x = _design(batch)
    bot = x[labels == 1].mean(axis=0)
    human = x[labels == 0].mean(axis=0)
    return np.sign(bot-human).astype(int).tolist()


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA):
        raise ValueError("schema differs from frozen fact vocabulary")
    results = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [row["fact"] for row in spec["concepts"]]
        named_groups, atomic_groups = _source_groups(names, spec)
        groups = [[2*j+k for j in indices for k in (0, 1)]
                  for indices in atomic_groups]
        runs = []
        for seed in (0, 1, 2):
            run = lock["runs"][dataset][seed]
            raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
            if source["split_sha256"] != run["split_sha256"] or \
                    source["files_sha256"] != run["source_files_sha256"] or \
                    names != run["selected_fact_names"]:
                raise ValueError("source or schema differs")
            raw = _missing_means(dataset, raw, fact_names, ids)
            facts = raw[:, [fact_names.index(name) for name in names]]
            train, dev = split["train"], split["val"]
            a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                    random_state=1000 + seed)
            reference_path = ROOT / run["reference_path"]
            if sha256_file(reference_path) != run["reference_sha256"]:
                raise ValueError("reference differs from saved candidate")
            reference = LabelBlindReference.load(reference_path)
            context = _context(facts, names, run)
            e_a = reference.transform(facts[a], None if context is None else context[a])
            e_b = reference.transform(facts[b], None if context is None else context[b])
            e_dev = reference.transform(facts[dev], None if context is None else context[dev])
            folds = ((e_a, y[a], e_b, y[b]),
                     (e_b, y[b], e_a, y[a]))
            ratios = []
            for ratio in RATIOS:
                logits = []
                zeros = []
                fold_effects = []
                fold_without = []
                direction_counts = []
                for direction_batch, direction_y, fit_batch, fit_y in folds:
                    signs = _signs(direction_batch, direction_y)
                    direction_counts.append({
                        "positive": signs.count(1), "negative": signs.count(-1),
                        "zero": signs.count(0)})
                    model = _model(seed, groups, signs)
                    fit_x, labels, weights = neutral_training_rows(
                        _design(fit_batch), fit_y, ratio)
                    model.fit(fit_x, labels, sample_weight=weights)
                    if not all_trees_one_group(model, groups)[0]:
                        raise ValueError("model crosses source group")
                    x_dev = _design(e_dev)
                    full = np.asarray(model.booster_.predict(
                        x_dev, raw_score=True))
                    zero = np.asarray(model.booster_.predict(
                        np.zeros_like(x_dev), raw_score=True))
                    effects = []
                    withouts = []
                    for group in groups:
                        masked = x_dev.copy()
                        masked[:, group] = 0.0
                        without = np.asarray(model.booster_.predict(
                            masked, raw_score=True))
                        effects.append(full - without)
                        withouts.append(without)
                    logits.append(full)
                    zeros.append(zero)
                    fold_effects.append(np.column_stack(effects))
                    fold_without.append(np.column_stack(withouts))
                score = np.mean(logits, axis=0) - np.mean(zeros, axis=0)
                effects = np.mean(fold_effects, axis=0)
                without = np.mean(fold_without, axis=0) - np.mean(zeros, axis=0)[:, None]
                additivity_error = float(np.max(np.abs(score - effects.sum(axis=1))))
                if additivity_error > 1e-9:
                    raise ValueError("signed group effects do not reconstruct score")
                p = expit(score)
                threshold = np.nextafter(.5, 1.0)
                if not np.array_equal(p >= threshold, score > 0):
                    raise ValueError("neutral probability rounding differs")
                metric = evaluate(y[dev], p, threshold)
                old = run["dev_metric_on_same_rows_used_to_calibrate"]
                ratios.append({
                    "anchor_ratio": ratio,
                    "neutral_dev_metric": metric,
                    "prior_one_tail_dev_errors": old["errors"],
                    "zero_score_ties": int(np.count_nonzero(score == 0)),
                    "bot_predictions": int(np.count_nonzero(score > 0)),
                    "human_predictions": int(np.count_nonzero(score < 0)),
                    "zero_input_logit": float(np.mean(zeros, axis=0)[0]),
                    "max_exact_group_additivity_error": additivity_error,
                    "group_class_changes_when_zeroed": {
                        name: int(np.count_nonzero((without[:, j] > 0) !=
                                                    (score > 0)))
                        for j, name in enumerate(named_groups)},
                    "direction_counts_by_fold": direction_counts,
                    "screen_gate_no_more_than_one_percent_acc_loss_and_no_ties":
                        metric["acc"] >= old["acc"] - .01 and
                        not np.any(score == 0),
                })
                print({"dataset": dataset, "seed": seed, "anchor_ratio": ratio,
                       "dev_errors": metric["errors"],
                       "ties": ratios[-1]["zero_score_ties"]}, flush=True)
            runs.append({"seed": seed, "split_sha256": run["split_sha256"],
                         "ratios": ratios})
        results[dataset] = runs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "schema_v2_same_16_facts_dual_tail_group_monotone_neutral_dev_only",
        "ratios_prelisted": RATIOS,
        "selection_lock_sha256": sha256_file(LOCK),
        "test_predictions_or_metrics_accessed": False,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
