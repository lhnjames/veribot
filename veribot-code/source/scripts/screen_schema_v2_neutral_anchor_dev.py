"""Train/dev-only neutral-anchor screen for frozen 16-fact schemas.

The fixed reference and A/B direction choices come from the saved candidate
lock. Only the B/A group-additive reasoners are retrained. No test rows are
used for model choice or metrics.
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
from lock_group_additive_schema_v2_candidates import _design, OUT as LOCK_DIR
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
LOCK = LOCK_DIR / "selection_lock.json"
SCHEMA = ROOT / "research/evidence_schema_v2.json"
OUT = ROOT / "output/gate1_schema_v2_neutral_anchor_dev/dev_only.json"
RATIOS = (0.02, 0.1, 0.5)


def _context(facts: np.ndarray, names: list[str], run: dict) -> np.ndarray | None:
    if run["reference_context_rule"] == "pooled_unlabeled_training_reference":
        return None
    if run["reference_context_rule"] != \
            "unlabeled_training_tertiles_with_global_fallback":
        raise ValueError("unknown frozen reference rule")
    cuts = np.asarray(run["train_only_context_quantiles"])
    return np.asarray([f"activity_{i}" for i in np.searchsorted(
        cuts, facts[:, names.index(run["reference_context_fact"])], side="right")])


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA):
        raise ValueError("schema differs from saved candidate")
    outputs = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [row["fact"] for row in spec["concepts"]]
        _, groups = _source_groups(names, spec)
        runs = []
        for seed in (0, 1, 2):
            run = lock["runs"][dataset][seed]
            raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
            if source["split_sha256"] != run["split_sha256"] or \
                    source["files_sha256"] != run["source_files_sha256"] or \
                    names != run["selected_fact_names"]:
                raise ValueError("source or model vocabulary differs")
            raw = _missing_means(dataset, raw, fact_names, ids)
            facts = raw[:, [fact_names.index(name) for name in names]]
            train, dev = split["train"], split["val"]
            a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                    random_state=1000 + seed)
            reference_path = ROOT / run["reference_path"]
            direction_path = ROOT / run["direction_path"]
            if sha256_file(reference_path) != run["reference_sha256"] or \
                    sha256_file(direction_path) != run["direction_sha256"]:
                raise ValueError("frozen reference or direction differs")
            ref = LabelBlindReference.load(reference_path)
            directions = json.loads(direction_path.read_text(encoding="utf-8"))
            context = _context(facts, names, run)
            e_a = ref.transform(facts[a], None if context is None else context[a])
            e_b = ref.transform(facts[b], None if context is None else context[b])
            e_dev = ref.transform(facts[dev], None if context is None else context[dev])
            train_batches = (e_b, e_a)
            train_labels = (y[b], y[a])
            ratio_rows = []
            for ratio in RATIOS:
                logits = []
                zeros = []
                for fold, (batch, labels, direction) in enumerate(zip(
                        train_batches, train_labels, directions, strict=True)):
                    signs = [int(direction["directions"][name]["sign"])
                             for name in names]
                    model = _model(seed, groups, signs)
                    x_fit = _design(batch, direction, names)
                    fit_x, fit_y, weights = neutral_training_rows(
                        x_fit, labels, ratio)
                    model.fit(fit_x, fit_y, sample_weight=weights)
                    if not all_trees_one_group(model, groups)[0]:
                        raise ValueError("neutral model tree crosses source groups")
                    x_dev = _design(e_dev, direction, names)
                    logits.append(np.asarray(model.booster_.predict(
                        x_dev, raw_score=True)))
                    zeros.append(np.asarray(model.booster_.predict(
                        np.zeros_like(x_dev), raw_score=True)))
                score = np.mean(logits, axis=0) - np.mean(zeros, axis=0)
                probability = expit(score)
                threshold = np.nextafter(.5, 1.0)
                if not np.array_equal(probability >= threshold, score > 0):
                    raise ValueError("neutral score and probability sign differ")
                metric = evaluate(y[dev], probability, threshold)
                old = run["dev_metric_on_same_rows_used_to_calibrate"]
                row = {
                    "anchor_ratio": ratio,
                    "neutral_dev_metric": metric,
                    "prior_dev_errors": old["errors"],
                    "neutral_dev_error_increase": metric["errors"] - old["errors"],
                    "zero_score_ties": int(np.count_nonzero(score == 0)),
                    "bot_predictions": int(np.count_nonzero(score > 0)),
                    "human_predictions": int(np.count_nonzero(score < 0)),
                    "zero_input_logit": float(np.mean(zeros, axis=0)[0]),
                    "screen_gate_no_more_than_one_percent_acc_loss_and_no_ties":
                        metric["acc"] >= old["acc"] - .01 and
                        not np.any(score == 0),
                }
                ratio_rows.append(row)
                print({"dataset": dataset, "seed": seed, "ratio": ratio,
                       "dev_errors": metric["errors"], "ties": row["zero_score_ties"]},
                      flush=True)
            runs.append({"seed": seed, "split_sha256": run["split_sha256"],
                         "ratios": ratio_rows})
        outputs[dataset] = runs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "frozen_schema_and_directions_neutral_anchor_screen_train_dev_only",
        "ratios_prelisted": RATIOS,
        "selection_lock_sha256": sha256_file(LOCK),
        "test_predictions_or_metrics_accessed": False,
        "runs": outputs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
