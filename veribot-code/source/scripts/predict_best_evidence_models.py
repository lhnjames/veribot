"""One account-ID entry for the three selected source-evidence models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_botsim_observed_means_from_sources import predict as predict_botsim
from verifiable_bot.compact_concept_effects import concept_effects
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime
from verifiable_bot.high_performance_fact_effects import account_fact_effects
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.mgtab_ab_positive_human_runtime import MGTABABPositiveHumanRuntime
from verifiable_bot.mgtab_bounded_projection_report import bounded_projection_report
from verifiable_bot.mgtab_reduced_projection_runtime import MGTABReducedProjectionRuntime


MODELS = {
    "botsim": ("source_group_tail_12_observed_means", 12),
    "fox8": ("raw_neighbor_atomic_74", 74),
    "mgtab": ("ab_group_additive_150", 150),
}


def predict(root: Path, dataset: str, seed: int, account_id: str,
            *, positive_human: bool = False,
            withholding_effects: bool = False,
            bounded_projection_residual: bool = False,
            fox8_twenty_concepts: bool = False,
            mgtab_eight_projection: bool = False) -> dict:
    if dataset not in MODELS or seed not in (0, 1, 2):
        raise ValueError("choose a frozen dataset and seed")
    if positive_human and dataset != "mgtab":
        raise ValueError("positive Human evidence variant is available for MGTAB")
    if positive_human and withholding_effects:
        raise ValueError("withholding effects are for the three selected main models")
    if bounded_projection_residual and (dataset != "mgtab" or positive_human):
        raise ValueError("bounded projection residual needs the MGTAB main model")
    if bounded_projection_residual and withholding_effects:
        raise ValueError("single-fact effects refer to the original uncapped model")
    if mgtab_eight_projection:
        if (dataset != "mgtab" or positive_human or withholding_effects
                or bounded_projection_residual or fox8_twenty_concepts):
            raise ValueError("8-coordinate model needs the MGTAB selected variant alone")
        compact = MGTABReducedProjectionRuntime(root, seed)
        compact_decision = compact.predict(str(account_id))
        reconstructed = compact_decision["zero_evidence_logit"] + math.fsum(
            float(group["logit_effect"]) for group in compact_decision["groups"])
        if (sum(len(group["source_facts"]) for group in compact_decision["groups"]) != 94
                or abs(reconstructed - compact_decision["logit"]) > 1e-9):
            raise ValueError("Reduced source-group evidence does not reconstruct decision")
        return {
            "dataset": dataset, "seed": seed,
            "model_id": "ab_source_group_tree_with_eight_capped_projection_coordinates",
            "account_id": str(account_id),
            "decision": compact_decision,
            "evidence_term_count": 94,
            "classification_input": "86 published profile/relation facts and 8 content projection coordinates",
            "source_scope": "published tensors and typed edges; averaged-tweet coordinates lack raw tweet text",
            "source_binding": {
                "inference_manifest": str(root /
                    "output/v3_mgtab_reduced_projection_ab_group/inference_manifest.json"),
            },
        }
    if fox8_twenty_concepts:
        if dataset != "fox8" or positive_human or bounded_projection_residual or withholding_effects:
            raise ValueError("20-concept A/B option needs the Fox8 selected variant alone")
        compact = Fox8TargetedTwentyRuntime(root, seed)
        compact_decision = compact.predict(str(account_id))
        reconstructed = compact_decision["zero_evidence_logit"] + math.fsum(
            float(group["logit_effect"])
            for group in compact_decision["source_group_effects"])
        if (len(compact_decision["concepts"]) != 20 or
                abs(reconstructed - compact_decision["logit"]) > 1e-9 or
                int(compact_decision["probability"] >= compact_decision["threshold"])
                != compact_decision["prediction"]):
            raise ValueError("20-concept source-group evidence does not reconstruct decision")
        return {
            "dataset": dataset, "seed": seed,
            "model_id": "targeted_20_ab_source_group_tree",
            "account_id": str(account_id),
            "decision": compact_decision,
            "evidence_term_count": 20,
            "classification_input": "only twenty source-derived scalar concepts",
            "source_scope": "raw account and sampled tweet records plus native reply/retweet graph",
            "source_binding": {
                "raw_and_neighbor_audit": str(root /
                    "research/certificates/fox8_neighbor_atomic_evidence_full_replay.json"),
                "native_event_index": str(compact.source.event_path),
            },
        }
    model_id, expected_terms = MODELS[dataset]
    if dataset == "botsim":
        sourced = predict_botsim(root, "botsim", seed, [str(account_id)])
        decision = sourced["decisions"][0]
        evidence = decision["terms"]
        intercept = float(decision["intercept"])
        source_binding = sourced["source_binding"]
    else:
        runtime = (Fox8NeighborAtomicRuntime(root, seed) if dataset == "fox8"
                   else MGTABABPositiveHumanRuntime(root, seed)
                   if positive_human else MGTABFullAtomicRuntime(
                       root, seed, variant="ab_group_additive"))
        if positive_human:
            model_id = "ab_group_additive_positive_human_309"
            expected_terms = 159
        decision = runtime.predict(str(account_id))
        evidence = decision["evidence"]
        intercept = float(decision["intercept_logit"])
        decision["threshold"] = runtime.threshold
        manifest = (root / ("output/v3_fox8_neighbor_atomic_evidence_capacity"
                            if dataset == "fox8" else
                            "output/v3_mgtab_ab_group_positive_human"
                            if positive_human else
                            "output/v3_mgtab_ab_group_additive_tree") /
                    "inference_manifest.json")
        source_binding = {"inference_manifest": str(manifest)}
    human_contributions = (decision["positive_human_logit_attributions"]
                           if positive_human else [])
    if (len(evidence) + len(human_contributions) != expected_terms or
            abs(intercept + math.fsum(
                [float(term["logit_contribution"]) for term in evidence] +
                [float(value) for value in human_contributions]) -
                float(decision["logit"])) > 1e-9 or
            int(float(decision["probability"]) >= float(decision["threshold"]))
            != int(decision["prediction"])):
        raise ValueError("evidence does not reconstruct frozen decision")
    result = {
        "dataset": dataset, "seed": seed, "model_id": model_id,
        "account_id": str(account_id),
        "source_binding": source_binding,
        "decision": decision,
        "evidence_term_count": expected_terms,
        "positive_human_evidence_channel_count": len(human_contributions),
        "classification_input": "only source-derived evidence values",
        "source_scope": (
            "published tensors and typed relation edges; tweet content is an average tensor"
            if dataset == "mgtab" else
            "published account, sampled posts and native relation records"),
    }
    if withholding_effects:
        result["evidence_effects"] = (concept_effects(decision)
                                      if dataset == "botsim" else
                                      account_fact_effects(runtime, str(account_id)))
    if bounded_projection_residual:
        result["bounded_projection_residual"] = bounded_projection_report(
            runtime, decision)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--positive-human", action="store_true",
                        help="MGTAB 309-input source-present Human evidence variant")
    parser.add_argument("--withholding-effects", action="store_true",
                        help="include exact one-concept or one-fact frozen-model effects")
    parser.add_argument("--bounded-projection-residual", action="store_true",
                        help="MGTAB validation-locked cap on the numeric projection group")
    parser.add_argument("--fox8-twenty-concepts", action="store_true",
                        help="Fox8 20-concept A/B grouped model with source-bound inference")
    parser.add_argument("--mgtab-eight-projection", action="store_true",
                        help="MGTAB A/B grouped model with eight bounded content coordinates")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = predict(Path(__file__).resolve().parents[2], args.dataset,
                     args.seed, args.account_id,
                     positive_human=args.positive_human,
                     withholding_effects=args.withholding_effects,
                     bounded_projection_residual=args.bounded_projection_residual,
                     fox8_twenty_concepts=args.fox8_twenty_concepts,
                     mgtab_eight_projection=args.mgtab_eight_projection)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(args.out)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
