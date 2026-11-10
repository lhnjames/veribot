"""Fox8 train/dev-only diagnosis of reference-stratification information loss."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _contexts, _lgbm, _missing_means, _source_facts
from verifiable_bot.evidence_schema_v1 import SCHEMA
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/gate1_fox8_reference_context/dev_only.json"


def main() -> None:
    spec = SCHEMA["fox8"]
    names16 = [row["fact"] for row in spec["concepts"]]
    groups = {}
    for row in spec["concepts"]:
        groups.setdefault(row["group"], []).append(row["fact"])
    runs = []
    for seed in (0, 1, 2):
        source, y, ids, names, split, manifest = _source_facts("fox8", seed)
        source = _missing_means("fox8", source, names, ids)
        facts = source[:, [names.index(name) for name in names16]]
        train, dev = split["train"], split["val"]
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=1000 + seed)
        context_col = names16.index(spec["reference_context"]["fact"])
        context = np.full(len(y), "", dtype="<U16")
        context[train], cuts = _contexts(facts[train, context_col],
                                         facts[train, context_col])
        context[dev], _ = _contexts(facts[train, context_col],
                                    facts[dev, context_col])
        results = {}
        for setting in ("activity_conditioned", "all_training_accounts_pooled"):
            training_context = context[train] if setting == "activity_conditioned" else None
            a_context = context[a] if setting == "activity_conditioned" else None
            b_context = context[b] if setting == "activity_conditioned" else None
            dev_context = context[dev] if setting == "activity_conditioned" else None
            reference = LabelBlindReference(min_context_n=30).fit(
                facts[train], names16, training_context)
            ev_a = reference.transform(facts[a], a_context)
            ev_b = reference.transform(facts[b], b_context)
            ev_dev = reference.transform(facts[dev], dev_context)
            reasoner = SignConstrainedReasoner(groups, l2=.01).fit(
                ev_a, y[a], ev_b, y[b])

            def design(batch):
                selected = np.column_stack([
                    batch.upper[:, j] if tail == "upper" else batch.lower[:, j]
                    for j, tail in enumerate(reasoner.tail)])
                return np.log1p(selected) * batch.reliability

            model = _lgbm(seed, stump=True,
                          monotone=reasoner.sign.astype(int).tolist())
            model.fit(design(ev_b), y[b])
            results[setting] = {
                "macro_and_accuracy": evaluate(y[dev],
                    model.predict_proba(design(ev_dev))[:, 1], .5),
                "reference_version": reference.version,
                "direction_by_fact": reasoner.manifest()["directions"],
                "split_count_by_fact": dict(zip(names16,
                    map(int, model.feature_importances_), strict=True)),
            }
        raw_b = _lgbm(seed, stump=True)
        raw_b.fit(facts[b], y[b])
        results["same_16_raw_facts_B_only"] = {
            "macro_and_accuracy": evaluate(y[dev],
                raw_b.predict_proba(facts[dev])[:, 1], .5),
            "split_count_by_fact": dict(zip(names16,
                map(int, raw_b.feature_importances_), strict=True)),
        }
        runs.append({"seed": seed, "split_sha256": manifest["split_sha256"],
                     "training_activity_quantiles": cuts,
                     "A_accounts": len(a), "B_accounts": len(b),
                     "dev_accounts": len(dev), "results": results})
        print({"seed": seed, **{key: round(item["macro_and_accuracy"]["macro_f1"], 5)
                                 for key, item in results.items()}}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "Fox8 same accounts and fields, conditional versus pooled label-free reference",
        "schema_sha256": sha256_file(ROOT / "research/evidence_schema_v1.json"),
        "test_predictions_or_metrics_accessed": False,
        "dev_only_diagnostic_not_model_selection_lock": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
