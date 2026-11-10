"""Evaluate dev-locked TwiBot-20 token evidence on official test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import groups as base_groups, metric, select, sha256, threshold_on_dev
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--token-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    token_manifest = json.loads(args.token_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    base_manifest = json.loads(args.base_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (lock["selected_model"] != "dual_tail_token128"
            or lock["candidate_test_predictions_accessed"] is not False
            or screen["test_predictions_accessed"] is not False
            or lock["screen_sha256"] != sha256(args.screen)
            or lock["source_audit_sha256"] != sha256(args.source_audit)
            or lock["base_facts_sha256"] != sha256(args.base_facts)
            or lock["token_facts_sha256"] != sha256(args.token_facts)
            or audit["status"] !=
            "verified_all_token_fact_cells_and_train_only_vocabulary"
            or audit["fact_cache_sha256"] != token_manifest["facts_sha256"]
            or token_manifest["source_archive_sha256"] != sha256(args.zip)
            or base_manifest["archive_sha256"] != sha256(args.zip)):
        raise ValueError("source, dev lock, or full audit differs")
    with np.load(args.base_facts, allow_pickle=False) as base:
        x = base["values"].astype(float)
        y = base["labels"].astype(int)
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        positions = base["positions"].astype(int)
        contexts = base["contexts"].astype(str)
        names = base["names"].astype(str).tolist()
    with np.load(args.token_facts, allow_pickle=False) as cache:
        if (not np.array_equal(ids, cache["ids"].astype(str))
                or not np.array_equal(parts, cache["split"].astype(str))
                or not np.array_equal(positions, cache["positions"].astype(int))):
            raise ValueError("token fact account order differs")
        token_names = cache["names"].astype(str).tolist()
        added = cache["values"].astype(float)
    if len(names) != 22 or len(token_names) != 128:
        raise ValueError("fixed vocabulary differs")
    matrix = np.column_stack((x, added))
    vocabulary = names + token_names
    groups = {**base_groups(names), "tweet_token_incidence": token_names}
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    test = np.flatnonzero(parts == "test")
    if (len(train), len(dev), len(test)) != (8278, 2365, 1183):
        raise ValueError("official split population differs")
    results = []
    for seed in (0, 1, 2):
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary, contexts=contexts[train])
        evidence = reference.transform(matrix, contexts=contexts)
        a, b = train_test_split(train, test_size=0.5,
                                stratify=y[train], random_state=seed)
        reasoner = DualTailEvidenceReasoner(groups, l2=0.01)
        reasoner.fit(select(evidence, a), y[a], select(evidence, b), y[b])
        dev_p = reasoner.predict_proba(select(evidence, dev))
        threshold = threshold_on_dev(dev_p, y[dev])
        dev_metric = metric(y[dev], dev_p, threshold)
        selected = screen["runs"][seed]["dual_tail_token128"]
        if (reference.version != selected["reference_version"]
                or not np.isclose(threshold, selected["threshold"],
                                  rtol=0, atol=1e-12)
                or any(not np.isclose(dev_metric[key], selected["validation"][key],
                                       rtol=0, atol=1e-12)
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("dev model differs from fixed selection")
        out = args.output / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        reference_path = out / "reference.json.gz"
        reference.save(reference_path)
        test_evidence = select(evidence, test)
        test_p = reasoner.predict_proba(test_evidence)
        contributions = reasoner.contributions(test_evidence)
        manifest = {
            "dataset": "twibot20", "seed": seed,
            "official_split": True, "support_labels_used": False,
            "source_archive_sha256": sha256(args.zip),
            "base_facts_sha256": sha256(args.base_facts),
            "token_facts_sha256": sha256(args.token_facts),
            "token_source_audit_sha256": sha256(args.source_audit),
            "selection_lock_sha256": sha256(args.selection_lock),
            "reference_version": reference.version,
            "reference_artifact_sha256": sha256(reference_path),
            "reference_used_bot_human_labels": False,
            "threshold": threshold,
            "threshold_selected_on": "official_dev_macro_f1",
            "direction_split_size": len(a),
            "coefficient_split_size": len(b),
            "fact_names": vocabulary,
        }
        report = {"manifest": manifest, "reasoner": reasoner.manifest(),
                  "dev": dev_metric,
                  "test": metric(y[test], test_p, threshold)}
        (out / "evidence_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out / "test_predictions.json").write_text(json.dumps({
            "manifest": manifest,
            "predictions": [
                {"account_id": ids[int(i)], "label": int(y[i]),
                 "probability": float(test_p[k]),
                 "group_contributions": {name: float(values[k])
                                         for name, values in contributions.items()}}
                for k, i in enumerate(test)]}, indent=2) + "\n",
            encoding="utf-8")
        results.append({"seed": seed, "dev": dev_metric,
                        "test": report["test"]})
    print(json.dumps({"selected_model": lock["selected_model"],
                      "runs": results}, indent=2))


if __name__ == "__main__":
    main()
