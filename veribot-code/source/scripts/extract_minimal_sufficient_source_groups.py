"""Enumerate source-group subsets under the current frozen classifiers.

Sufficiency here means preserving the classifier's thresholded output after
zeroing encoded inputs of omitted groups. It is not a claim about behavior,
causality, or an independent semantic ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime

from audit_best_source_group_effects import GROUPS


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "output/v3_best_source_group_effects"
OUTPUT = ROOT / "output/v3_minimal_sufficient_source_groups"
CERT = ROOT / "research/certificates/minimal_sufficient_source_groups.json"


def _load_group_effects(dataset: str, seed: int) -> dict:
    path = INPUT / f"{dataset}_seed{seed}.npz"
    with np.load(path, allow_pickle=False) as z:
        out = {key: z[key].copy() for key in z.files}
    out["path"] = path
    return out


def _all_subsets_botsim(seed: int, saved: dict) -> np.ndarray:
    runtime = BotSimObservedMeanRuntime(ROOT, seed)
    ids = saved["ids"].astype(str)
    rows = runtime.predict_many(ids.tolist())
    full_logit = np.asarray([row["logit"] for row in rows], dtype=float)
    exact_path = ROOT / f"output/v3_best_exact_fact_effects/botsim_observed_seed{seed}.npz"
    with np.load(exact_path, allow_pickle=False) as z:
        concept_names = z["concept_names"].astype(str)
        contribution = z["logit_difference"].astype(float)
    groups = saved["group_names"].astype(str).tolist()
    if concept_names.tolist() != list(runtime.reasoner.names):
        raise ValueError("BotSim concept vocabulary changed")
    group_contribution = np.column_stack([
        contribution[:, [j for j, name in enumerate(concept_names)
                         if name.startswith(group + ":")]].sum(axis=1)
        for group in groups
    ])
    if abs(float(saved["threshold"]) - runtime.threshold) > 0:
        raise ValueError("BotSim threshold changed")
    masks = np.arange(1 << len(groups), dtype=np.uint32)
    included = ((masks[:, None] >> np.arange(len(groups))) & 1).astype(float)
    # Starting from the actual frozen full logit preserves its operation order.
    removed = 1.0 - included
    return expit(full_logit[:, None] - group_contribution @ removed.T)


def _all_subsets_tree(dataset: str, seed: int, saved: dict) -> np.ndarray:
    runtime = (Fox8NeighborAtomicRuntime(ROOT, seed) if dataset == "fox8"
               else MGTABFullAtomicRuntime(ROOT, seed,
                                            variant="ab_group_additive"))
    ids = saved["ids"].astype(str)
    rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
    design = evidence_design(runtime, rows)
    groups = GROUPS[dataset]
    if [x[0] for x in groups] != saved["group_names"].astype(str).tolist():
        raise ValueError("Source-group vocabulary changed")
    if abs(float(saved["threshold"]) - runtime.threshold) > 0:
        raise ValueError("Tree threshold changed")
    d = design.shape[1] // 2
    probabilities = np.empty((len(ids), 1 << len(groups)), dtype=float)
    for mask in range(1 << len(groups)):
        modified = design.copy()
        for j, (_, start, stop) in enumerate(groups):
            if not (mask & (1 << j)):
                modified[:, start:stop] = 0.0
                modified[:, start+d:stop+d] = 0.0
        probabilities[:, mask] = expit(np.asarray(
            runtime.model.booster_.predict(modified, raw_score=True),
            dtype=float))
    return probabilities


def _check_one_group_replays(scores: np.ndarray, saved: dict) -> float:
    g = len(saved["group_names"])
    full_mask = (1 << g) - 1
    observed = [np.max(np.abs(scores[:, full_mask] -
                              saved["full_probability"]))]
    observed.append(np.max(np.abs(scores[:, 0] -
                                  saved["empty_evidence_probability"])))
    for j in range(g):
        observed.append(np.max(np.abs(
            scores[:, full_mask ^ (1 << j)] -
            saved["without_group_probability"][:, j])))
    error = float(max(observed))
    if error > 1e-12:
        raise ValueError(f"Existing exact group-effect audit differs: {error}")
    return error


def run(dataset: str, seed: int) -> dict:
    saved = _load_group_effects(dataset, seed)
    scores = (_all_subsets_botsim(seed, saved) if dataset == "botsim"
              else _all_subsets_tree(dataset, seed, saved))
    replay_error = _check_one_group_replays(scores, saved)
    group_names = saved["group_names"].astype(str)
    g = len(group_names)
    full_mask = (1 << g) - 1
    masks = np.arange(1 << g, dtype=np.int32)
    cardinality = np.asarray([int(mask).bit_count() for mask in masks])
    full_p = scores[:, full_mask]
    empty_p = scores[:, 0]
    threshold = float(saved["threshold"])
    full_class = full_p >= threshold
    sufficient = (scores >= threshold) == full_class[:, None]
    if not sufficient[:, full_mask].all():
        raise ValueError("Full group set did not preserve its own decision")
    min_size = np.min(np.where(sufficient, cardinality[None, :], g+1), axis=1)
    min_candidate = sufficient & (cardinality[None, :] == min_size[:, None])
    distance = np.abs(scores - full_p[:, None])
    selected = np.argmin(np.where(min_candidate, distance, np.inf), axis=1)
    selected_p = scores[np.arange(len(scores)), selected]
    necessary = np.empty(len(scores), dtype=np.int32)
    count_min = min_candidate.sum(axis=1)
    for i in range(len(scores)):
        intersection = full_mask
        for mask in masks[sufficient[i]]:
            intersection &= int(mask)
        necessary[i] = intersection
    best_distance = np.column_stack([
        distance[:, cardinality == k].min(axis=1) for k in range(g+1)])
    best_sufficient_distance = np.column_stack([
        np.where(sufficient[:, cardinality == k],
                 distance[:, cardinality == k], np.inf).min(axis=1)
        for k in range(g+1)])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{dataset}_seed{seed}.npz"
    np.savez_compressed(
        path, ids=saved["ids"], group_names=group_names,
        subset_mask=masks, subset_probability=scores,
        full_probability=full_p, threshold=np.asarray(threshold),
        minimum_sufficient_group_count=min_size,
        chosen_minimum_mask=selected, chosen_minimum_probability=selected_p,
        minimum_sufficient_set_count=count_min,
        necessary_group_mask=necessary,
        best_probability_difference_by_group_count=best_distance,
        best_class_preserving_difference_by_group_count=
            best_sufficient_distance)
    size_histogram = {str(k): int((min_size == k).sum()) for k in range(g+1)}
    predicted_class = {
        name: {str(k): int(np.count_nonzero((full_class == value) &
                                              (min_size == k)))
               for k in range(g+1)}
        for name, value in (("human", False), ("bot", True))}
    directional_support = np.where(full_class,
                                   full_p > empty_p + 1e-10,
                                   full_p < empty_p - 1e-10)
    directional_opposition = np.where(full_class,
                                      full_p < empty_p - 1e-10,
                                      full_p > empty_p + 1e-10)
    return {
        "dataset": dataset, "seed": seed, "accounts": len(scores),
        "group_names": group_names.tolist(), "model_probability_replay_error":
            replay_error,
        "minimum_sufficient_size_histogram": size_histogram,
        "minimum_sufficient_size_by_predicted_class": predicted_class,
        "empty_input_same_class_count": int((min_size == 0).sum()),
        "empty_input_bot_probability": float(empty_p[0]),
        "score_direction_relative_to_empty_input": {
            name: {
                "supports_predicted_class": int(np.count_nonzero(
                    (full_class == value) & directional_support)),
                "opposes_predicted_class": int(np.count_nonzero(
                    (full_class == value) & directional_opposition)),
                "neutral_within_1e-10": int(np.count_nonzero(
                    (full_class == value) & ~directional_support &
                    ~directional_opposition))}
            for name, value in (("human", False), ("bot", True))},
        "accounts_requiring_all_groups": int((min_size == g).sum()),
        "accounts_with_any_necessary_group": int(np.count_nonzero(necessary)),
        "bot_predicted_group_membership": {
            str(group_names[j]): {
                "in_chosen_minimum": int(np.count_nonzero(
                    full_class & ((selected & (1 << j)) != 0))),
                "necessary_in_every_class_preserving_subset": int(
                    np.count_nonzero(full_class &
                                     ((necessary & (1 << j)) != 0)))}
            for j in range(g)},
        "median_probability_difference_of_chosen_minimum": float(
            np.median(np.abs(selected_p - full_p))),
        "mean_best_absolute_probability_difference_by_group_count": [
            float(x) for x in best_distance.mean(axis=0)],
        "full_input_probability_sha256": sha256_file(saved["path"]),
        "subset_matrix_path": str(path), "subset_matrix_sha256":
            sha256_file(path),
    }


def main() -> None:
    rows = []
    for dataset in ("botsim", "fox8", "mgtab"):
        for seed in (0, 1, 2):
            row = run(dataset, seed)
            rows.append(row)
            print({"dataset": dataset, "seed": seed,
                   "minimum_set_histogram": row[
                       "minimum_sufficient_size_histogram"],
                   "empty": row["empty_input_same_class_count"],
                   "necessary": row["accounts_with_any_necessary_group"]},
                  flush=True)
    CERT.parent.mkdir(parents=True, exist_ok=True)
    CERT.write_text(json.dumps({
        "status": "exact_exhaustive_frozen_classifier_source_group_subsets",
        "intervention": "omit a source group by zeroing both encoded evidence tails, then evaluate the same frozen classifier",
        "sufficiency_definition": "smallest cardinality subset whose thresholded class equals the complete-input class",
        "scope_limit": "This measures classifier response to encoded-input withholding. It does not establish behavioral causality, independent semantic validity, or whether input zero is a natural account.",
        "fixed_test_labels_read": False,
        "runs": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
