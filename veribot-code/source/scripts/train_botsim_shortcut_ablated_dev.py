"""Freeze the widest BotSim shortcut ablation selected on official dev only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_botsim_observed_means_dev import ACTIVITY, DATA, NATIVE, ROOT, USERS, corrected_facts
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


SCREEN = ROOT / "output/v3_botsim_simulator_shortcut_ablation/dev_only.json"
FOLDER = ROOT / "output/v3_botsim_shortcut_ablated_10"
OUT = FOLDER / "dev_only.json"


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    selected = screen["selected_for_next_stage"]
    if selected != "drop_presentation_and_score_zero_cues" or screen["test_predictions_or_metrics_accessed"]:
        raise ValueError("shortcut candidate has not been selected by dev-only screen")
    excluded = set(screen["configurations"][selected])
    full_groups = json.loads((ROOT / "output/v3_botsim_observed_means_12/dev_only.json").read_text(encoding="utf-8"))["atomic_groups"]
    groups = {group: [name for name in members if name not in excluded]
              for group, members in full_groups.items()}
    groups = {group: members for group, members in groups.items() if members}
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    FOLDER.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        raw, labels, ids, names, split, source = load("botsim", DATA, USERS, seed, NATIVE)
        facts, _ = corrected_facts(raw, names, ids, activity)
        keep = [i for i, name in enumerate(names) if name not in excluded]
        kept_names = [names[i] for i in keep]
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(facts[train][:, keep], kept_names)
        concept_train, _ = collapse_group_tails(reference.transform(facts[train][:, keep]), groups)
        concept_dev, _ = collapse_group_tails(reference.transform(facts[dev][:, keep]), groups)
        concept_groups = {group: [f"{group}:high_tail_max", f"{group}:low_tail_max"]
                          for group in groups}
        a, b = train_test_split(np.arange(len(train)), test_size=.5,
                                stratify=labels[train], random_state=seed)
        reasoner = MonotoneSplineReasoner(concept_groups, l2=.01)
        reasoner.fit(rows(concept_train, a), labels[train][a],
                     rows(concept_train, b), labels[train][b])
        probability = reasoner.predict_proba(concept_dev)
        threshold = best_validation_threshold(probability, labels[dev])
        metrics = evaluate(labels[dev], probability, threshold)
        prior = screen["runs"][seed]["configs"][selected]
        if (source["split_sha256"] != screen["runs"][seed]["split_sha256"] or
                threshold != prior["threshold"] or metrics != prior["validation"] or
                reference.version != prior["reference_version"] or
                concept_dev.reference_version != reasoner.reference_version):
            raise ValueError("selected candidate differs from dev-only screen")
        directory = FOLDER / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / "reference.json.gz"
        reasoner_path = directory / "reasoner.json"
        reference.save(reference_path)
        reasoner_path.write_text(json.dumps(reasoner.manifest(), indent=2) + "\n", encoding="utf-8")
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "users_sha256": sha256_file(USERS), "activity_sha256": sha256_file(ACTIVITY),
            "native_graph_facts_sha256": sha256_file(NATIVE),
            "reference_version": reference.version,
            "reference_artifact_sha256": sha256_file(reference_path),
            "reasoner_artifact_sha256": sha256_file(reasoner_path),
            "threshold": threshold, "validation": metrics,
            "A_and_B_disjoint": bool(not len(np.intersect1d(a, b))),
        })
        print({"seed": seed, "dev_errors": metrics["errors"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_shortcut_ablated_observed_means_ten_concepts_dev_lock",
        "screen_sha256": sha256_file(SCREEN),
        "selected_configuration": selected,
        "excluded_source_facts": sorted(excluded),
        "retained_source_facts": kept_names,
        "atomic_groups": groups,
        "preprocessing": "literal zero preserved; post means missing without posts; comment means missing without first-party comments",
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
