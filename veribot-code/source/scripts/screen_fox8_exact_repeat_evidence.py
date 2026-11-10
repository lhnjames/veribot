"""Dev-only Fox8 exact-repeat concept added to the locked evidence bottleneck."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def joined(original: EvidenceBatch, added: EvidenceBatch) -> EvidenceBatch:
    version = hashlib.sha256(json.dumps(
        {"original": original.reference_version, "added": added.reference_version,
         "names": list(original.names + added.names)},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return EvidenceBatch(
        np.column_stack((original.upper, added.upper)),
        np.column_stack((original.lower, added.lower)),
        np.column_stack((original.reliability, added.reliability)),
        np.column_stack((original.available, added.available)),
        version, original.names + added.names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-facts", type=Path, required=True)
    parser.add_argument("--repeat-facts", type=Path, required=True)
    parser.add_argument("--full-model-dir", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.repeat_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (audit["status"] != "all_accounts_values_and_tweet_locators_recomputed"
            or audit["facts_sha256"] != sha256_file(args.repeat_facts)
            or manifest["facts_sha256"] != audit["facts_sha256"]):
        raise ValueError("repeat fact source audit differs")
    with np.load(args.repeat_facts, allow_pickle=False) as cache:
        repeat_ids = cache["ids"].astype(str).tolist()
        repeat_names = cache["names"].astype(str).tolist()
        repeat = cache["values"].astype(float)
        support = cache["supports"].astype(float)
    if repeat_names != [manifest["fact_name"]] or repeat.shape != (2279, 1):
        raise ValueError("repeat fact vocabulary differs")
    groups = json.loads(args.groups_json.read_text(encoding="utf-8"))
    groups["sample_text"].append(manifest["fact_name"])
    result_runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir,
                                               None, seed)
        if ids != repeat_ids:
            raise ValueError("repeat fact account order differs")
        with np.load(args.native_facts, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("native fact account order differs")
            x = np.column_stack((x, cache["values"].astype(float)))
            names += cache["names"].astype(str).tolist()
        full = json.loads((args.full_model_dir / f"seed{seed}" / "report.json")
                          .read_text(encoding="utf-8"))
        fixed = json.loads((args.selected_model_dir / f"seed{seed}" / "report.json")
                           .read_text(encoding="utf-8"))
        if (source["split_sha256"] != fixed["manifest"]["split_sha256"]
                or sorted(source["files_sha256"].values()) !=
                sorted(fixed["manifest"]["source_files_sha256"].values())
                or tuple(names) != tuple(full["reasoner"]["directions"])
                or sha256_file(args.native_facts) != fixed["manifest"]["native_facts_sha256"]):
            raise ValueError("fixed Fox8 input or split differs")
        reference = LabelBlindReference.load(
            args.full_model_dir / f"seed{seed}" / "reference.json.gz")
        if reference.version != full["manifest"]["reference_version"]:
            raise ValueError("fixed label-blind reference differs")
        train, dev = split["train"], split["val"]
        old_train, old_dev = reference.transform(x[train]), reference.transform(x[dev])
        baseline_groups = json.loads(args.groups_json.read_text(encoding="utf-8"))
        old_compact, _ = collapse_group_tails(old_dev, baseline_groups)
        old_model = MonotoneSplineReasoner.from_manifest(fixed["reasoner"])
        old_probability = old_model.predict_proba(old_compact)
        old_threshold = fixed["manifest"]["threshold"]
        baseline = evaluate(y[dev], old_probability, old_threshold)
        if any(abs(baseline[key] - fixed["validation"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("baseline validation does not replay")
        repeat_ref = LabelBlindReference(support_prior=5.0).fit(
            repeat[train], repeat_names)
        added_train = repeat_ref.transform(repeat[train],
                                           support_counts=support[train])
        added_dev = repeat_ref.transform(repeat[dev],
                                         support_counts=support[dev])
        compact_train, _ = collapse_group_tails(joined(old_train, added_train), groups)
        joined_dev = joined(old_dev, added_dev)
        compact_dev, winners = collapse_group_tails(joined_dev, groups)
        if len(compact_train.names) != 20:
            raise ValueError("evidence bottleneck changed width")
        concept_groups = {group: [f"{group}:high_tail_max",
                                  f"{group}:low_tail_max"]
                          for group in groups}
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        model = MonotoneSplineReasoner(concept_groups, l2=0.0001)
        model.fit(rows(compact_train, a), y[train][a],
                  rows(compact_train, b), y[train][b])
        probability = model.predict_proba(compact_dev)
        threshold = best_validation_threshold(probability, y[dev])
        validation = evaluate(y[dev], probability, threshold)
        missing_upper = joined_dev.upper.copy()
        missing_lower = joined_dev.lower.copy()
        missing_available = joined_dev.available.copy()
        missing_upper[:, -1] = 0.0
        missing_lower[:, -1] = 0.0
        missing_available[:, -1] = False
        without_repeat = EvidenceBatch(
            missing_upper, missing_lower, joined_dev.reliability,
            missing_available, joined_dev.reference_version, joined_dev.names)
        compact_without, _ = collapse_group_tails(without_repeat, groups)
        without_probability = model.predict_proba(compact_without)
        before = old_probability >= old_threshold
        after = probability >= threshold
        repeat_index = len(names)
        winner_counts = {name: int(np.count_nonzero(index == repeat_index))
                         for name, index in winners.items()
                         if name.startswith("sample_text:")}
        result_runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "original_reference_version": reference.version,
            "repeat_reference_version": repeat_ref.version,
            "new_concept_version": compact_train.reference_version,
            "repeated_accounts_in_dev": int(np.count_nonzero(repeat[dev, 0] > 0)),
            "repeat_wins_concept_on_dev": winner_counts,
            "repeat_removal_changes_probability_count": int(np.count_nonzero(
                np.abs(probability - without_probability) > 1e-12)),
            "repeat_removal_changes_class_count": int(np.count_nonzero(
                (probability >= threshold) != (without_probability >= threshold))),
            "repeat_removal_max_probability_change": float(np.max(
                np.abs(probability - without_probability))),
            "baseline_validation": baseline,
            "new_validation": validation,
            "new_threshold": threshold,
            "corrected_vs_baseline": int(np.count_nonzero((after == y[dev]) &
                                                          (before != y[dev]))),
            "new_errors_vs_baseline": int(np.count_nonzero((after != y[dev]) &
                                                            (before == y[dev]))),
        })
    result = {
        "dataset": "fox8", "scope": "official_train_validation_only",
        "test_predictions_accessed": False,
        "fact": manifest["fact_name"],
        "fact_source_audit_sha256": sha256_file(args.source_audit),
        "fact_archive_sha256": manifest["facts_sha256"],
        "bottleneck_concept_count": 20,
        "insertion_group": "sample_text",
        "locked_l2": 0.0001,
        "new_fact_support_prior": 5.0,
        "runs": result_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [
        {"seed": row["seed"], "baseline": row["baseline_validation"],
         "new": row["new_validation"], "winner": row["repeat_wins_concept_on_dev"]}
        for row in result_runs]}, indent=2))


if __name__ == "__main__":
    main()
