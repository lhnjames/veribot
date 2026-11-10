"""Exact per-concept model dependence of Fox8 pooled-reference evidence GAM."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _lgbm, _missing_means, _source_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
OUT = ROOT / "output/gate1_fox8_schema_v2_decision_dependence/dev_only.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    spec = schema["schema"]["fox8"]
    if (schema["version"] != "candidate-v2" or
            spec["reference_context"]["rule"] !=
            "pooled_unlabeled_training_reference"):
        raise ValueError("Fox8 pooled-reference schema differs")
    names16 = [item["fact"] for item in spec["concepts"]]
    groups = {}
    for item in spec["concepts"]:
        groups.setdefault(item["group"], []).append(item["fact"])
    runs = []
    for seed in (0, 1, 2):
        raw, labels, ids, fact_names, split, source = _source_facts("fox8", seed)
        raw = _missing_means("fox8", raw, fact_names, ids)
        facts = raw[:, [fact_names.index(name) for name in names16]]
        train, dev = split["train"], split["val"]
        a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                random_state=1000 + seed)
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names16)
        ev_a = reference.transform(facts[a])
        ev_b = reference.transform(facts[b])
        ev_dev = reference.transform(facts[dev])
        direction = SignConstrainedReasoner(groups, l2=.01).fit(
            ev_a, labels[a], ev_b, labels[b])

        def design(batch):
            values = np.column_stack([
                batch.upper[:, j] if tail == "upper" else batch.lower[:, j]
                for j, tail in enumerate(direction.tail)])
            return np.log1p(values) * batch.reliability

        model = _lgbm(seed, stump=True,
                      monotone=direction.sign.astype(int).tolist())
        model.fit(design(ev_b), labels[b])
        x = design(ev_dev)
        logit = np.asarray(model.booster_.predict(x, raw_score=True))
        p = expit(logit)
        threshold = .5
        base_class = p >= threshold
        zero = np.zeros_like(x)
        zero_logit = np.asarray(model.booster_.predict(zero, raw_score=True))
        changes = []
        logit_effects = []
        feature_rows = []
        for j, name in enumerate(names16):
            masked = x.copy()
            masked[:, j] = 0.0
            without = np.asarray(model.booster_.predict(masked, raw_score=True))
            effect = logit - without
            logit_effects.append(effect)
            changed = base_class != (expit(without) >= threshold)
            changes.append(changed)
            feature_rows.append({
                "fact": name,
                "source_selector": spec["concepts"][j]["source_selector"],
                "tail_and_class_direction": direction.manifest()["directions"][name],
                "nonzero_evidence_accounts": int(np.count_nonzero(x[:, j])),
                "class_changes_on_zero_evidence_mask": int(np.count_nonzero(changed)),
                "absolute_logit_effect_mean": float(np.mean(np.abs(effect))),
                "absolute_logit_effect_p90": float(np.quantile(np.abs(effect), .9)),
                "gain_split_count": int(model.feature_importances_[j]),
            })
        decomposition_error = float(np.max(np.abs(
            logit - zero_logit - np.sum(logit_effects, axis=0))))
        shap = np.asarray(model.booster_.predict(x, pred_contrib=True))
        shap_error = float(np.max(np.abs(shap.sum(axis=1) - logit)))
        if decomposition_error > 1e-9 or shap_error > 1e-9:
            raise ValueError("feature-masked effects do not exactly reconstruct GAM score")
        change_matrix = np.column_stack(changes)
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": reference.version,
            "dev": evaluate(labels[dev], p, threshold),
            "zero_evidence_class_changes": int(np.count_nonzero(
                base_class != (expit(zero_logit) >= threshold))),
            "accounts_with_any_single_concept_class_change": int(np.count_nonzero(
                change_matrix.any(axis=1))),
            "accounts_with_two_or_more_single_concept_class_changes": int(
                np.count_nonzero(change_matrix.sum(axis=1) >= 2)),
            "max_exact_additive_logit_reconstruction_error": decomposition_error,
            "max_shap_raw_score_reconstruction_error": shap_error,
            "per_concept": feature_rows,
        })
        print({"seed": seed,
               "correct": len(dev) - runs[-1]["dev"]["errors"],
               "any_single_concept_class_change": runs[-1][
                   "accounts_with_any_single_concept_class_change"],
               "top_class_change_facts": sorted(
                   [(item["fact"], item["class_changes_on_zero_evidence_mask"])
                    for item in feature_rows], key=lambda item: -item[1])[:4]},
              flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "Fox8 v2 pooled reference, dev-only exact additive decision dependence",
        "schema_sha256": sha256_file(SCHEMA),
        "test_predictions_or_metrics_accessed": False,
        "single_concept_zero_mask_is_model_function_probe_not_source_intervention": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
