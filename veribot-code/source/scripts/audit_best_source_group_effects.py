"""Exact group-level source-input withholding in frozen Fox8/MGTAB models."""

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
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output/v3_best_source_group_effects"
CERT = ROOT / "research/certificates/best_source_group_effects.json"
GROUPS = {
    "fox8": [("account_and_profile", 0, 23),
             ("sampled_tweet_activity", 23, 31),
             ("native_interaction", 31, 43),
             ("native_neighbor_means", 43, 74)],
    "mgtab": [("published_profile", 0, 20),
              ("published_typed_relations", 20, 86),
              ("published_averaged_tweet_projection", 86, 150)],
}


def run_botsim(seed: int) -> dict:
    runtime = BotSimObservedMeanRuntime(ROOT, seed)
    exact_path = ROOT / f"output/v3_best_exact_fact_effects/botsim_observed_seed{seed}.npz"
    with np.load(exact_path, allow_pickle=False) as exact:
        ids = exact["ids"].astype(str)
        expected = exact["full_probability"].astype(float)
        without_fact = exact["without_probability"].astype(float)
        contribution = exact["logit_difference"].astype(float)
        names = exact["concept_names"].astype(str)
        threshold = float(exact["threshold"])
    decisions = runtime.predict_many(ids.tolist())
    probability = np.asarray([row["probability"] for row in decisions])
    logit = np.asarray([row["logit"] for row in decisions])
    if (len(ids) != 437 or threshold != runtime.threshold or
            names.tolist() != list(runtime.reasoner.names) or
            np.max(np.abs(probability-expected)) > 1e-12):
        raise ValueError("frozen BotSim concept decisions differ")
    rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
    atomic = runtime.reference.transform(runtime.facts[rows])
    concepts, _ = collapse_group_tails(atomic, runtime.atomic_groups)
    full_class = probability >= threshold
    single_fact_pivotal = np.any((without_fact >= threshold) !=
                                 full_class[:, None], axis=1)
    group_names = list(dict.fromkeys(name.split(":")[0] for name in names))
    without_groups = []
    summaries = {}
    for group in group_names:
        selected = [j for j, name in enumerate(names)
                    if name.startswith(group + ":")]
        after = expit(logit-contribution[:, selected].sum(axis=1))
        high = concepts.upper.copy()
        low = concepts.lower.copy()
        high[:, selected] = 0.0
        low[:, selected] = 0.0
        masked = EvidenceBatch(high, low, concepts.reliability,
            concepts.available, concepts.reference_version, concepts.names)
        direct = runtime.reasoner.predict_proba(masked)
        if np.max(np.abs(after-direct)) > 1e-12:
            raise ValueError("BotSim group subtraction differs from direct mask")
        changed = (after >= threshold) != full_class
        without_groups.append(after)
        summaries[group] = {"concept_count": len(selected),
            "score_changed_accounts": int(np.count_nonzero(
                np.abs(after-probability) > 1e-10)),
            "class_changed_accounts": int(np.count_nonzero(changed)),
            "class_changed_without_any_single_fact_pivotal": int(
                np.count_nonzero(changed & ~single_fact_pivotal))}
    without = np.stack(without_groups, axis=1)
    group_pivotal = np.any((without >= threshold) != full_class[:, None], axis=1)
    empty_p = expit(logit-contribution.sum(axis=1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"botsim_seed{seed}.npz"
    np.savez_compressed(out, ids=ids, fact_names=names,
        group_names=np.asarray(group_names), full_probability=probability,
        without_group_probability=without,
        empty_evidence_probability=empty_p, threshold=np.asarray(threshold))
    return {"dataset": "botsim", "seed": seed, "accounts": len(ids),
        "fact_count": len(names), "source_groups": summaries,
        "accounts_with_any_single_fact_class_flip": int(single_fact_pivotal.sum()),
        "accounts_with_any_source_group_class_flip": int(group_pivotal.sum()),
        "accounts_with_group_flip_but_no_single_fact_flip": int(
            np.count_nonzero(group_pivotal & ~single_fact_pivotal)),
        "accounts_with_all_evidence_zero_class_flip": int(np.count_nonzero(
            (empty_p >= threshold) != full_class)),
        "accounts_with_all_evidence_flip_but_no_group_flip": int(np.count_nonzero(
            ((empty_p >= threshold) != full_class) & ~group_pivotal)),
        "source_model_sha256": runtime.source_binding["reasoner_sha256"],
        "single_fact_effect_sha256": sha256_file(exact_path),
        "group_effect_path": str(out), "group_effect_sha256": sha256_file(out),
        "maximum_full_probability_replay_error": float(np.max(
            np.abs(probability-expected)))}


def run(dataset: str, seed: int) -> dict:
    runtime = (Fox8NeighborAtomicRuntime(ROOT, seed) if dataset == "fox8"
               else MGTABFullAtomicRuntime(ROOT, seed,
                                            variant="ab_group_additive"))
    exact_path = ROOT / f"output/v3_best_exact_fact_effects/{dataset}_seed{seed}.npz"
    with np.load(exact_path, allow_pickle=False) as exact:
        ids = exact["ids"].astype(str)
        expected = exact["full_probability"].astype(float)
        without_fact = exact["without_probability"].astype(float)
        threshold = float(exact["threshold"])
        fact_names = exact["fact_names"].astype(str)
    if (fact_names.tolist() != list(runtime.reference.names) or
            threshold != runtime.threshold or
            len(ids) != (343 if dataset == "fox8" else 1020)):
        raise ValueError("frozen source fact vocabulary or test population differs")
    rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
    matrix = evidence_design(runtime, rows)
    d = len(fact_names)
    if matrix.shape != (len(ids), 2*d):
        raise ValueError("frozen evidence design shape differs")
    probability = expit(np.asarray(runtime.model.booster_.predict(
        matrix, raw_score=True), dtype=float))
    if np.max(np.abs(probability - expected)) > 1e-12:
        raise ValueError("full classifier does not replay exact-fact audit")
    full_class = probability >= threshold
    single_fact_pivotal = np.any((without_fact >= threshold) !=
                                 full_class[:, None], axis=1)
    group_names = []
    altered_probability = []
    summaries = {}
    for name, first, stop in GROUPS[dataset]:
        modified = matrix.copy()
        modified[:, first:stop] = 0.0
        modified[:, first+d:stop+d] = 0.0
        after = expit(np.asarray(runtime.model.booster_.predict(
            modified, raw_score=True), dtype=float))
        changed = (after >= threshold) != full_class
        group_names.append(name)
        altered_probability.append(after)
        summaries[name] = {"fact_count": stop-first,
            "score_changed_accounts": int(np.count_nonzero(
                np.abs(after-probability) > 1e-10)),
            "class_changed_accounts": int(np.count_nonzero(changed)),
            "class_changed_without_any_single_fact_pivotal": int(
                np.count_nonzero(changed & ~single_fact_pivotal))}
    without = np.stack(altered_probability, axis=1)
    group_pivotal = np.any((without >= threshold) != full_class[:, None], axis=1)
    empty = np.zeros_like(matrix)
    empty_p = expit(np.asarray(runtime.model.booster_.predict(
        empty, raw_score=True), dtype=float))
    all_pivotal = (empty_p >= threshold) != full_class
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dataset}_seed{seed}.npz"
    np.savez_compressed(out, ids=ids, fact_names=fact_names,
        group_names=np.asarray(group_names), full_probability=probability,
        without_group_probability=without,
        empty_evidence_probability=empty_p, threshold=np.asarray(threshold))
    return {"dataset": dataset, "seed": seed, "accounts": len(ids),
        "fact_count": d, "source_groups": summaries,
        "accounts_with_any_single_fact_class_flip": int(single_fact_pivotal.sum()),
        "accounts_with_any_source_group_class_flip": int(group_pivotal.sum()),
        "accounts_with_group_flip_but_no_single_fact_flip": int(
            np.count_nonzero(group_pivotal & ~single_fact_pivotal)),
        "accounts_with_all_evidence_zero_class_flip": int(all_pivotal.sum()),
        "accounts_with_all_evidence_flip_but_no_group_flip": int(
            np.count_nonzero(all_pivotal & ~group_pivotal)),
        "source_model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
        "single_fact_effect_sha256": sha256_file(exact_path),
        "group_effect_path": str(out), "group_effect_sha256": sha256_file(out),
        "maximum_full_probability_replay_error": float(np.max(
            np.abs(probability-expected)))}


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        row = run_botsim(seed)
        runs.append(row)
        print({"dataset": "botsim", "seed": seed,
            "single_fact": row["accounts_with_any_single_fact_class_flip"],
            "source_group": row["accounts_with_any_source_group_class_flip"],
            "distributed_group": row[
                "accounts_with_group_flip_but_no_single_fact_flip"]}, flush=True)
    for dataset in ("fox8", "mgtab"):
        for seed in (0, 1, 2):
            row = run(dataset, seed)
            runs.append(row)
            print({"dataset": dataset, "seed": seed,
                "single_fact": row["accounts_with_any_single_fact_class_flip"],
                "source_group": row["accounts_with_any_source_group_class_flip"],
                "distributed_group": row[
                    "accounts_with_group_flip_but_no_single_fact_flip"]}, flush=True)
    CERT.write_text(json.dumps({
        "status": "frozen_source_group_input_effects_bound_to_full_fact_effects",
        "scope": "botsim_fox8_and_mgtab_three_fixed_test_splits",
        "intervention": "zero both encoded evidence tails of one predeclared source group; preserve all original records and model parameters",
        "semantic_or_causal_ground_truth_claimed": False,
        "runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
