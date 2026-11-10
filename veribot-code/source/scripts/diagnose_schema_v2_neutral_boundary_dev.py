"""Development-only neutral-boundary check for locked schema-v2 models.

No parameters or thresholds are fitted here. The class boundary is the frozen
model's score on the all-zero evidence vector, fixed before dev evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from lock_group_additive_schema_v2_candidates import _design, OUT as LOCK_DIR
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
LOCK = LOCK_DIR / "selection_lock.json"
SCHEMA = ROOT / "research/evidence_schema_v2.json"
OUT = ROOT / "output/gate1_schema_v2_neutral_boundary_dev/dev_only.json"


def _context(facts: np.ndarray, names: list[str], run: dict) -> np.ndarray | None:
    if run["reference_context_rule"] == "pooled_unlabeled_training_reference":
        return None
    if run["reference_context_rule"] != \
            "unlabeled_training_tertiles_with_global_fallback":
        raise ValueError("unknown frozen reference rule")
    cuts = np.asarray(run["train_only_context_quantiles"])
    values = facts[:, names.index(run["reference_context_fact"])]
    return np.asarray([f"activity_{i}" for i in
                       np.searchsorted(cuts, values, side="right")])


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA):
        raise ValueError("schema differs from candidate lock")
    reports = {}
    for dataset in ("fox8", "botsim"):
        rows = []
        for seed in (0, 1, 2):
            run = lock["runs"][dataset][seed]
            raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
            if source["split_sha256"] != run["split_sha256"] or \
                    source["files_sha256"] != run["source_files_sha256"]:
                raise ValueError("development population or source differs")
            raw = _missing_means(dataset, raw, fact_names, ids)
            names = run["selected_fact_names"]
            facts = raw[:, [fact_names.index(name) for name in names]]
            dev = split["val"]
            reference_path = ROOT / run["reference_path"]
            direction_path = ROOT / run["direction_path"]
            model_paths = [ROOT / p for p in run["model_paths"]]
            if sha256_file(reference_path) != run["reference_sha256"] or \
                    sha256_file(direction_path) != run["direction_sha256"] or \
                    [sha256_file(p) for p in model_paths] != run["model_sha256"]:
                raise ValueError("frozen artifact differs")
            reference = LabelBlindReference.load(reference_path)
            direction = json.loads(direction_path.read_text(encoding="utf-8"))
            evidence = reference.transform(facts[dev], _context(facts[dev], names, run))
            full_logits = []
            zero_logits = []
            for path, manifest in zip(model_paths, direction, strict=True):
                model = joblib.load(path)
                x = _design(evidence, manifest, names)
                full_logits.append(np.asarray(model.booster_.predict(x, raw_score=True)))
                zero_logits.append(np.asarray(model.booster_.predict(
                    np.zeros_like(x), raw_score=True)))
            score = np.mean(full_logits, axis=0) - np.mean(zero_logits, axis=0)
            p_neutral = expit(score)
            strict_threshold = np.nextafter(.5, 1.0)
            if not np.array_equal(p_neutral >= strict_threshold, score > 0):
                raise ValueError("probability rounding changes neutral score rule")
            p_original = expit(np.mean(full_logits, axis=0))
            original = evaluate(y[dev], p_original,
                                float(run["classification_threshold_selected_on_dev"]))
            if original["errors"] != run["dev_metric_on_same_rows_used_to_calibrate"]["errors"]:
                raise ValueError("saved development result does not replay")
            row = {
                "seed": seed, "accounts": len(dev),
                "neutral_boundary_score_rule": "Bot iff full_logit - zero_evidence_logit > 0",
                "zero_effect_ties": int(np.count_nonzero(score == 0)),
                "zero_evidence_logit": float(np.mean(zero_logits, axis=0)[0]),
                "prior_calibrated_dev_metric": original,
                "neutral_boundary_dev_metric": evaluate(y[dev], p_neutral,
                                                         strict_threshold),
                "neutral_bot_predictions": int(np.count_nonzero(score > 0)),
                "neutral_human_predictions": int(np.count_nonzero(score < 0)),
            }
            rows.append(row)
            print({"dataset": dataset, "seed": seed,
                   "prior_errors": original["errors"],
                   "neutral_errors": row["neutral_boundary_dev_metric"]["errors"],
                   "ties": row["zero_effect_ties"]}, flush=True)
        reports[dataset] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "saved_models_neutral_boundary_dev_only",
        "selection_lock_sha256": sha256_file(LOCK),
        "test_predictions_or_metrics_accessed": False,
        "runs": reports,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
