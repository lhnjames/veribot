"""Independently replay compact MGTAB evidence models and all fixed test cards."""

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
    folder = root / "output/v3_mgtab_named_top20_lightgbm"
    lock_path = folder / "selection_lock.json"
    summary_path = folder / "selected/paired_summary.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["selection_lock_sha256"] != sha256_file(lock_path)
            or summary["published_content_projection_used"] is not False):
        raise ValueError("compact selection lock or source level differs")
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    checks = []
    for seed in (0, 1, 2):
        run_dir = folder / "selected" / f"seed{seed}"
        model_path = run_dir / "model.txt"
        predictions_path = run_dir / "test_predictions.json"
        cards_path = run_dir / "evidence_cards.jsonl"
        saved = json.loads(predictions_path.read_text(encoding="utf-8"))
        cards = [json.loads(line) for line in cards_path.read_text(
            encoding="utf-8").splitlines()]
        run = summary["runs"][seed]
        parent_reference_path = (root /
            f"output/v3_mgtab_named_evidence_lightgbm/selected/seed{seed}/reference.json.gz")
        if (run["model_sha256"] != sha256_file(model_path)
                or run["predictions_sha256"] != sha256_file(predictions_path)
                or run["evidence_cards_sha256"] != sha256_file(cards_path)
                or run["reference_sha256"] != sha256_file(parent_reference_path)
                or saved["manifest"]["selection_lock_sha256"] != sha256_file(lock_path)):
            raise ValueError("compact selected model, cards, or reference differs")
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        test = split["test"]
        selected = np.asarray(lock["selected_fact_indices_by_seed"][str(seed)],
                              dtype=int)
        if (len(selected) != 20 or [names[j] for j in selected] !=
                lock["selected_fact_names_by_seed"][str(seed)]
                or old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(cards) != len(test) or len(saved["predictions"]) != len(test)):
            raise ValueError("compact fact selection or fixed population differs")
        reference = LabelBlindReference.load(parent_reference_path)
        evidence = transform_with_support(reference, x[test, :86],
                                          split["supports"][test])
        full = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                np.log1p(evidence.lower) * evidence.reliability))
        matrix = full[:, np.r_[selected, selected + 86]]
        model = lightgbm.Booster(model_file=str(model_path))
        probability = np.asarray(model.predict(matrix))
        raw = np.asarray(model.predict(matrix, raw_score=True))
        contribution = np.asarray(model.predict(matrix, pred_contrib=True))
        if contribution.shape != (len(test), 41):
            raise ValueError("compact contribution dimension differs")
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        threshold = saved["manifest"]["threshold"]
        h_threshold = baseline["manifest"]["threshold"]
        candidate_correct = hypbot_correct = candidate_only = hypbot_only = 0
        max_probability_error = max_card_error = 0.0
        for position, index in enumerate(test):
            row = saved["predictions"][position]
            card = cards[position]
            other = baseline["predictions"][position]
            if (row["account_id"] != str(int(index))
                    or other["account_id"] != row["account_id"]
                    or card["account_id"] != row["account_id"]
                    or row["label"] != int(y[index])
                    or other["label"] != row["label"]):
                raise ValueError("compact paired test account differs")
            max_probability_error = max(max_probability_error,
                                        abs(probability[position] - row["probability"]))
            if (abs(raw[position] - row["raw_logit"]) > 1e-12
                    or abs(raw[position] - card["raw_logit"]) > 1e-12
                    or abs(contribution[position].sum() -
                           card["contribution_sum"]) > 1e-12
                    or abs(contribution[position, -1] -
                           card["contribution_bias"]) > 1e-12):
                raise ValueError("compact evidence logit or bias differs")
            predicted = int(probability[position] >= threshold)
            if predicted != row["prediction"] or predicted != card["prediction"]:
                raise ValueError("compact fixed class differs")
            top = np.argsort(np.abs(contribution[position, :-1]))[-8:][::-1]
            if len(card["top_absolute_contributions"]) != 8:
                raise ValueError("compact card top evidence count differs")
            for item, j in zip(card["top_absolute_contributions"], top):
                fact_index = selected[j % 20]
                raw_fact = x[index, fact_index]
                expected = float(raw_fact) if np.isfinite(raw_fact) else None
                if (item["fact_name"] != names[fact_index]
                        or item["tail"] != ("upper" if j < 20 else "lower")
                        or item["raw_fact_value"] != expected
                        or item["published_node_row"] != int(index)
                        or item["published_source"] !=
                        ("features.pt profile column" if fact_index < 20
                         else "typed edge derived fact cache")
                        or item["paper_table_a5_profile_alias"] !=
                        (PROFILE_SCHEMA[fact_index][0]
                         if fact_index < 20 else None)):
                    raise ValueError("compact top item source locator differs")
                max_card_error = max(max_card_error,
                                     abs(item["evidence_input"] - matrix[position, j]),
                                     abs(item["logit_contribution"] -
                                         contribution[position, j]))
            new_ok = predicted == int(y[index])
            old_ok = int(other["probability"] >= h_threshold) == int(y[index])
            candidate_correct += new_ok
            hypbot_correct += old_ok
            candidate_only += new_ok and not old_ok
            hypbot_only += old_ok and not new_ok
        if (max_probability_error > 1e-12 or max_card_error > 1e-12
                or candidate_correct != run["candidate_correct"]
                or hypbot_correct != run["hypbot_correct"]
                or candidate_only != run["candidate_only_correct"]
                or hypbot_only != run["hypbot_only_correct"]):
            raise ValueError("compact probability, card, or paired count differs")
        checks.append({
            "seed": seed, "accounts_checked": len(test),
            "top_evidence_items_checked": len(test) * 8,
            "max_saved_probability_difference": max_probability_error,
            "max_card_contribution_difference": max_card_error,
            "candidate_correct": candidate_correct,
            "hypbot_correct": hypbot_correct,
            "candidate_only_correct": candidate_only,
            "hypbot_only_correct": hypbot_only,
            "model_sha256": sha256_file(model_path),
            "cards_sha256": sha256_file(cards_path),
        })
        print(seed, len(test), max_probability_error, max_card_error,
              flush=True)
    output = {
        "scope": "independent_3060_mgtab_compact_named_evidence_probability_card_and_paired_replay",
        "selection_lock_sha256": sha256_file(lock_path),
        "paired_summary_sha256": sha256_file(summary_path),
        "runs": checks,
    }
    destination = root / "research/mgtab_locked_named_top20_lightgbm_replay_2026-11-03.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
