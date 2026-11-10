"""Independently replay all release-source MGTAB LightGBM predictions and cards."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.mgtab_profile_schema import PROFILE_SCHEMA
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_named_evidence_lightgbm"
    lock_path = folder / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary_path = folder / "selected/paired_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["selection_lock_sha256"] != sha256_file(lock_path)
            or summary["published_content_projection_used"] is not False):
        raise ValueError("locked selected LightGBM summary differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    if (lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(structural)
            or lock["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("published MGTAB source hash differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor,
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    checks = []
    for seed in (0, 1, 2):
        run_dir = folder / "selected" / f"seed{seed}"
        prediction_path = run_dir / "test_predictions.json"
        cards_path = run_dir / "evidence_cards.jsonl"
        model_path = run_dir / "model.txt"
        reference_path = run_dir / "reference.json.gz"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        cards = [json.loads(line) for line in cards_path.read_text(
            encoding="utf-8").splitlines()]
        run = summary["runs"][seed]
        if (prediction["manifest"]["selection_lock_sha256"] != sha256_file(lock_path)
                or prediction["manifest"]["model_sha256"] != sha256_file(model_path)
                or prediction["manifest"]["reference_sha256"] !=
                sha256_file(reference_path)
                or run["predictions_sha256"] != sha256_file(prediction_path)
                or run["evidence_cards_sha256"] != sha256_file(cards_path)):
            raise ValueError("locked MGTAB evidence artifacts differ")
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        test = split["test"]
        if (old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(test) != len(cards) or len(test) != len(prediction["predictions"])):
            raise ValueError("official test split or card population differs")
        reference = LabelBlindReference.load(reference_path)
        if (reference.version != lock["reference_version_by_seed"][str(seed)]
                or list(reference.names) != names[:86]):
            raise ValueError("label-blind named reference differs")
        evidence = transform_with_support(
            reference, x[test, :86], split["supports"][test])
        matrix = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                  np.log1p(evidence.lower) * evidence.reliability))
        model = lightgbm.Booster(model_file=str(model_path))
        probability = np.asarray(model.predict(matrix))
        raw = np.asarray(model.predict(matrix, raw_score=True))
        contrib = np.asarray(model.predict(matrix, pred_contrib=True))
        max_probability_difference = 0.0
        max_card_contribution_difference = 0.0
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        h_threshold = baseline["manifest"]["threshold"]
        threshold = prediction["manifest"]["threshold"]
        new_correct = old_correct = only_new = only_old = 0
        for position, index in enumerate(test):
            saved = prediction["predictions"][position]
            card = cards[position]
            other = baseline["predictions"][position]
            if (saved["account_id"] != str(int(index))
                    or saved["label"] != int(y[index])
                    or card["account_id"] != saved["account_id"]
                    or other["account_id"] != saved["account_id"]
                    or other["label"] != saved["label"]):
                raise ValueError("paired fixed test population differs")
            max_probability_difference = max(
                max_probability_difference, abs(probability[position] -
                                                saved["probability"]))
            if (abs(raw[position] - saved["raw_logit"]) > 1e-12
                    or abs(contrib[position].sum() - saved["contribution_sum"]) > 1e-12
                    or abs(contrib[position, -1] - card["contribution_bias"]) > 1e-12
                    or abs(raw[position] - card["raw_logit"]) > 1e-12):
                raise ValueError("saved test logit or evidence card differs")
            predicted = int(probability[position] >= threshold)
            if predicted != saved["prediction"] or predicted != card["prediction"]:
                raise ValueError("saved test class differs")
            expected_top = np.argsort(np.abs(contrib[position, :-1]))[-8:][::-1]
            if len(card["top_absolute_contributions"]) != 8:
                raise ValueError("top evidence card length differs")
            for item, j in zip(card["top_absolute_contributions"], expected_top):
                value = x[index, j % 86]
                expected_raw = float(value) if np.isfinite(value) else None
                if (item["fact_name"] != names[j % 86]
                        or item["tail"] != ("upper" if j < 86 else "lower")
                        or item["raw_fact_value"] != expected_raw
                        or item["published_node_row"] != int(index)
                        or item["published_source"] !=
                        ("features.pt profile column" if j % 86 < 20
                         else "typed edge derived fact cache")
                        or item["paper_table_a5_profile_alias"] !=
                        (PROFILE_SCHEMA[j % 86][0] if j % 86 < 20 else None)):
                    raise ValueError("top evidence source fact differs")
                max_card_contribution_difference = max(
                    max_card_contribution_difference,
                    abs(item["evidence_input"] - matrix[position, j]),
                    abs(item["logit_contribution"] - contrib[position, j]))
            is_new = predicted == int(y[index])
            is_old = int(other["probability"] >= h_threshold) == int(y[index])
            new_correct += is_new
            old_correct += is_old
            only_new += is_new and not is_old
            only_old += is_old and not is_new
        if (max_probability_difference > 1e-12
                or max_card_contribution_difference > 1e-12
                or new_correct != run["candidate_correct"]
                or old_correct != run["hypbot_correct"]
                or only_new != run["candidate_only_correct"]
                or only_old != run["hypbot_only_correct"]):
            raise ValueError("replayed probabilities, cards, or paired counts differ")
        checks.append({
            "seed": seed, "accounts_checked": len(test),
            "top_evidence_items_checked": len(test) * 8,
            "max_saved_probability_difference": max_probability_difference,
            "max_card_contribution_difference": max_card_contribution_difference,
            "candidate_correct": new_correct, "hypbot_correct": old_correct,
            "candidate_only_correct": only_new, "hypbot_only_correct": only_old,
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(reference_path),
            "cards_sha256": sha256_file(cards_path),
        })
        print(seed, len(test), max_probability_difference,
              max_card_contribution_difference, flush=True)
    output = {
        "scope": "independent_3060_mgtab_named_source_lightgbm_probability_card_and_paired_replay",
        "selection_lock_sha256": sha256_file(lock_path),
        "paired_summary_sha256": sha256_file(summary_path),
        "runs": checks,
    }
    destination = root / "research/mgtab_locked_named_evidence_lightgbm_replay_2026-11-03.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
