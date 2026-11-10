"""Audit whether concise fixed-decision cards omit contrary model effects.

The effects are exact frozen-model input-withholding responses, not causal
effects of editing a raw social-media record.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
REASON_CERT = ROOT / "research/certificates/best_decision_reasons_full_test.json"
OUT = ROOT / "research/certificates/best_reason_direction_balance.json"
ROWS = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
TOL = 1e-10


def _effect_path(dataset: str, seed: int) -> Path:
    name = (f"botsim_observed_seed{seed}.npz" if dataset == "botsim"
            else f"{dataset}_seed{seed}.npz")
    return ROOT / "output/v3_best_exact_fact_effects" / name


def _item(name: str, term: dict, delta: float, without: float) -> dict:
    return {
        "name": name,
        "source_fact": term["source_fact"],
        "source_level": term["source_level"],
        "source_value": term["source_value"],
        "source_query": term["source_query"],
        "logit_difference_when_withheld": delta,
        "probability_without_fact": without,
    }


def main() -> None:
    manifest = json.loads(REASON_CERT.read_text(encoding="utf-8"))
    if manifest["total_accounts"] != 5400 or len(manifest["runs"]) != 9:
        raise ValueError("unexpected fixed-decision population")
    summaries = []
    with ROWS.open("wb") as sink, gzip.GzipFile(
            filename="", mode="wb", fileobj=sink, mtime=0) as target:
        for run in manifest["runs"]:
            dataset, seed = run["dataset"], int(run["seed"])
            reason_path = Path(run["reason_cards_file"])
            source_path = ROOT / ("output/v3_best_verified_evidence_full_test/"
                                  f"{dataset}_seed{seed}.jsonl.gz")
            effect_path = _effect_path(dataset, seed)
            if (sha256_file(reason_path) != run["reason_cards_sha256"] or
                    sha256_file(source_path) != run["source_cards_sha256"] or
                    sha256_file(effect_path) != run["effect_matrix_sha256"]):
                raise ValueError("bound reason, source or effect file changed")
            with np.load(effect_path, allow_pickle=False) as matrix:
                ids = matrix["ids"].astype(str).tolist()
                names = matrix["concept_names" if dataset == "botsim"
                               else "fact_names"].astype(str).tolist()
                full = matrix["full_probability"].astype(float)
                without = matrix["without_probability"].astype(float)
                threshold = float(matrix["threshold"])
                delta = (matrix["logit_difference"].astype(float)
                         if dataset == "botsim" else
                         matrix["full_logit"].astype(float)[:, None] -
                         matrix["without_logit"].astype(float))
            if (len(ids) != run["accounts"] or
                    without.shape != (len(ids), len(names)) or
                    delta.shape != without.shape):
                raise ValueError("effect matrix shape changed")
            counts = Counter()
            by_prediction = defaultdict(Counter)
            by_reason = defaultdict(Counter)
            examples = []
            with gzip.open(reason_path, "rt", encoding="utf-8") as reason_stream, \
                    gzip.open(source_path, "rt", encoding="utf-8") as source_stream:
                for i, (reason_line, source_line) in enumerate(zip(
                        reason_stream, source_stream, strict=True)):
                    reason = json.loads(reason_line)
                    source = json.loads(source_line)
                    if (i >= len(ids) or reason["account_id"] != ids[i] or
                            source["account_id"] != ids[i] or
                            abs(reason["probability"] - full[i]) > TOL or
                            abs(source["probability"] - full[i]) > TOL or
                            reason["threshold"] != threshold or
                            source["threshold"] != threshold):
                        raise ValueError("reason/source/effect decision mismatch")
                    terms = {term["name"]: term
                             for term in source["evidence_terms"]}
                    if len(terms) != len(names) or set(terms) != set(names):
                        raise ValueError("source vocabulary mismatch")
                    prediction = int(reason["prediction"])
                    if prediction != int(full[i] >= threshold):
                        raise ValueError("prediction mismatch")
                    sign = 1 if prediction == 1 else -1
                    supporting = []
                    opposing = []
                    for j, name in enumerate(names):
                        effect = float(delta[i, j])
                        if abs(effect) <= TOL:
                            continue
                        term = terms[name]
                        if (term["source_query"] is None or
                                not term["source_fact_observed"]):
                            raise ValueError("score-relevant item lacks source")
                        item = _item(name, term, effect,
                                     float(without[i, j]))
                        (supporting if effect * sign > 0 else opposing).append(item)
                    supporting.sort(key=lambda x: (-abs(x[
                        "logit_difference_when_withheld"]), x["name"]))
                    opposing.sort(key=lambda x: (-abs(x[
                        "logit_difference_when_withheld"]), x["name"]))
                    displayed = reason["displayed_items"]
                    dependency = source["exact_frozen_model_dependency"]
                    group_masks = dependency.get("source_group_masks", {})
                    class_critical_group_masks = {
                        name: effect for name, effect in group_masks.items()
                        if effect["class_changes_without_group"]
                    }
                    group_critical_despite_no_single_or_all_change = bool(
                        reason["reason_category"] ==
                        "no_class_change_from_all_evidence" and
                        class_critical_group_masks)
                    if len(displayed) > 5 or len({x["name"] for x in displayed}) != len(displayed):
                        raise ValueError("display cap or duplicate changed")
                    by_name = {name: j for j, name in enumerate(names)}
                    displayed_oppose = 0
                    for item in displayed:
                        j = by_name[item["name"]]
                        if not math.isclose(float(item[
                                "logit_difference_when_withheld"]),
                                float(delta[i, j]), abs_tol=TOL, rel_tol=0):
                            raise ValueError("displayed effect mismatch")
                        displayed_oppose += int(float(delta[i, j]) * sign < -TOL)
                    flags = {
                        "has_supporting": bool(supporting),
                        "has_opposing": bool(opposing),
                        "has_both_directions": bool(supporting and opposing),
                        "opposing_omitted_from_display": bool(
                            opposing and displayed_oppose == 0),
                        "five_critical_and_opposition_omitted": bool(
                            opposing and displayed_oppose == 0 and
                            len(displayed) == 5 and
                            all(x["role"] == "class_critical"
                                for x in displayed)),
                        "group_critical_despite_no_single_or_all_change":
                            group_critical_despite_no_single_or_all_change,
                    }
                    for key, value in flags.items():
                        counts[key] += value
                        by_prediction[str(prediction)][key] += value
                        by_reason[reason["reason_category"]][key] += value
                    counts["accounts"] += 1
                    counts["displayed_opposing_items"] += displayed_oppose
                    balanced_items = list(displayed)
                    if flags["opposing_omitted_from_display"]:
                        balanced_items.append({
                            **opposing[0],
                            "role": "contrary_score_relevant",
                            "class_changes_when_withheld": False,
                        })
                    if (len(balanced_items) > 6 or
                            (opposing and not any(float(x[
                                "logit_difference_when_withheld"]) * sign < -TOL
                                for x in balanced_items))):
                        raise ValueError("balanced display still omits opposition")
                    counts["six_item_balanced_cards"] += len(balanced_items) == 6
                    if flags["opposing_omitted_from_display"] and len(examples) < 3:
                        examples.append(ids[i])
                    row = {
                        "dataset": dataset, "seed": seed,
                        "account_id": ids[i], "prediction": prediction,
                        "reason_category": reason["reason_category"],
                        "all_evidence_zero_probability": dependency[
                            "all_evidence_zero_probability"],
                        "source_group_masks_available": "source_group_masks" in
                            dependency,
                        "class_critical_source_group_masks":
                            class_critical_group_masks,
                        "source_card_sha256": run["source_cards_sha256"],
                        "reason_card_sha256": run["reason_cards_sha256"],
                        "supporting_count": len(supporting),
                        "opposing_count": len(opposing),
                        "displayed_item_count": len(displayed),
                        "class_critical_displayed_count": sum(
                            x["role"] == "class_critical" for x in displayed),
                        "displayed_opposing_count": displayed_oppose,
                        "opposing_omitted_from_display": flags[
                            "opposing_omitted_from_display"],
                        "group_critical_despite_no_single_or_all_change":
                            group_critical_despite_no_single_or_all_change,
                        "strongest_supporting_input_effect": (
                            supporting[0] if supporting else None),
                        "strongest_opposing_input_effect": (
                            opposing[0] if opposing else None),
                        "additional_opposing_item_for_balanced_card": (
                            opposing[0] if flags[
                                "opposing_omitted_from_display"] else None),
                        "balanced_displayed_items": balanced_items,
                    }
                    target.write((json.dumps(row, ensure_ascii=False,
                                             separators=(",", ":"), allow_nan=False)
                                  + "\n").encode("utf-8"))
            if counts["accounts"] != run["accounts"]:
                raise ValueError("run population mismatch")
            summaries.append({
                "dataset": dataset, "seed": seed,
                **dict(counts),
                "by_prediction": {k: dict(v) for k, v in by_prediction.items()},
                "by_reason_category": {k: dict(v) for k, v in by_reason.items()},
                "first_omission_account_ids": examples,
                "reason_cards_sha256": run["reason_cards_sha256"],
                "source_cards_sha256": run["source_cards_sha256"],
                "effect_matrix_sha256": run["effect_matrix_sha256"],
            })
            print(dataset, seed, dict(counts), flush=True)
    OUT.write_text(json.dumps({
        "status": "all_current_reason_cards_direction_display_audited",
        "scope": "nine_fixed_test_runs_5400_account_decisions",
        "effect_definition": "full_model_logit_minus_exact_single_input_withheld_logit",
        "direction_definition": "positive_effect_supports_predicted_bot_negative_supports_predicted_human",
        "not_a_raw_source_edit_or_behavioral_causality": True,
        "reason_certificate_sha256": sha256_file(REASON_CERT),
        "account_sidecar_sha256": sha256_file(ROWS),
        "runs": summaries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
