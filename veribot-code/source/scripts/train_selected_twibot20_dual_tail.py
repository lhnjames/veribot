"""Train the dev-locked TwiBot-20 evidence model and evaluate official test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import groups, metric, select, sha256, threshold_on_dev
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if lock["selected_model"] != "dual_tail_22" or \
            lock["test_predictions_accessed"] is not False or \
            lock["screen_sha256"] != sha256(args.screen) or \
            lock["base_facts_sha256"] != sha256(args.facts) or \
            screen["test_predictions_accessed"] is not False or \
            audit["status"] != "verified_all_labeled_account_fact_cells_against_official_zip" or \
            audit["fact_cache_sha256"] != sha256(args.facts) or \
            audit["source_archive_sha256"] != lock["source_archive_sha256"]:
        raise ValueError("dev selection, source facts or population audit differs")
    with np.load(args.facts, allow_pickle=False) as cache:
        x = cache["values"].astype(float)
        y = cache["labels"].astype(int)
        ids = cache["ids"].astype(str)
        split = cache["split"].astype(str)
        contexts = cache["contexts"].astype(str)
        names = cache["names"].astype(str).tolist()
    if x.shape != (11826, 22):
        raise ValueError("selected TwiBot-20 fact population differs")
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    test = np.flatnonzero(split == "test")
    if (len(train), len(dev), len(test)) != (8278, 2365, 1183):
        raise ValueError("official TwiBot-20 split differs")
    results = []
    for seed in (0, 1, 2):
        reference = LabelBlindReference(min_context_n=30).fit(
            x[train], names, contexts=contexts[train])
        evidence = reference.transform(x, contexts=contexts)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        model = DualTailEvidenceReasoner(groups(names), l2=0.01)
        model.fit(select(evidence, a), y[a], select(evidence, b), y[b])
        dev_probability = model.predict_proba(select(evidence, dev))
        threshold = threshold_on_dev(dev_probability, y[dev])
        dev_metric = metric(y[dev], dev_probability, threshold)
        screened = screen["runs"][seed]["dual_tail_22"]
        if screen["runs"][seed]["seed"] != seed or \
                reference.version != screened["reference_version"] or \
                not np.isclose(threshold, screened["threshold"], rtol=0, atol=1e-12) or \
                any(not np.isclose(dev_metric[key], screened["validation"][key],
                                   rtol=0, atol=1e-12)
                    for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("frozen dev model differs from locked selection")
        test_evidence = select(evidence, test)
        probability = model.predict_proba(test_evidence)
        contributions = model.contributions(test_evidence)
        out = args.output / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        reference_path = out / "reference.json.gz"
        reference.save(reference_path)
        manifest = {
            "dataset": "twibot20", "seed": seed,
            "model": "computable_evidence_dual_tail",
            "official_split": True,
            "facts_sha256": sha256(args.facts),
            "source_archive_sha256": lock["source_archive_sha256"],
            "source_audit_sha256": sha256(args.source_audit),
            "selection_lock_sha256": sha256(args.selection_lock),
            "screen_sha256": lock["screen_sha256"],
            "reference_version": reference.version,
            "reference_population": "official_train_mixture_by_domain",
            "reference_used_labels": False,
            "reference_artifact_sha256": sha256(reference_path),
            "direction_split_size": len(a),
            "coefficient_split_size": len(b),
            "threshold": threshold,
            "threshold_selected_on": "official_dev_macro_f1",
        }
        report = {"manifest": manifest, "reasoner": model.manifest(),
                  "dev": dev_metric, "test": metric(y[test], probability, threshold)}
        (out / "evidence_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                   encoding="utf-8")
        (out / "test_predictions.json").write_text(json.dumps({
            "manifest": manifest,
            "predictions": [
                {"account_id": ids[int(i)], "label": int(y[i]),
                 "probability": float(probability[k]),
                 "group_contributions": {name: float(value[k])
                                         for name, value in contributions.items()}}
                for k, i in enumerate(test)]}, indent=2) + "\n", encoding="utf-8")
        results.append({"seed": seed, "dev": dev_metric, "test": report["test"]})
    print(json.dumps({"selected_model": lock["selected_model"],
                      "runs": results}, indent=2))


if __name__ == "__main__":
    main()
