"""Replay all selected fixed-test predictions with the label-free loaded runtime."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.compact_evidence_runtime import CompactEvidenceRuntime
from verifiable_bot.compact_source_population import load_source_population


CONFIG = {
    "botsim": {
        "data": "D:/Code/SAHG/SAHG/code/data/botsim_24",
        "account": "data/raw/botsim_24/Users.csv",
        "native": "data/processed/botsim_v3/native_graph_facts.npz",
        "selected": "output/v3_computable/botsim_group_tail_12",
        "full": "output/v3_computable/botsim_native_graph",
        "cards": "concept_evidence_cards.jsonl.gz", "terms": "concepts",
    },
    "fox8": {
        "data": "D:/Code/SAHG/SAHG/code/data/fox8_23",
        "native": "data/processed/fox8_v3/native_interaction_facts.npz",
        "selected": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
        "full": "output/v3_fox8_native_spline_l2/selected",
        "groups": "research/fox8_20_concept_groups.json",
        "cards": "concept_evidence_cards.jsonl.gz", "terms": "concepts",
    },
    "mgtab": {
        "data": "D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB",
        "structural": "data/processed/mgtab_v3/structural_facts.npz",
        "neighbor": "data/processed/mgtab_v3/neighbor_facts.npz",
        "selected": "output/v3_mgtab_ab_spline_content/direct_selected_20",
        "full": "output/v3_mgtab_ab_spline_content/selected",
        "cards": "evidence_cards.jsonl.gz", "terms": "terms",
    },
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_seed(root: Path, name: str, config: dict, seed: int) -> dict:
    selected_dir = root / config["selected"]
    full_dir = root / config["full"]
    runtime = CompactEvidenceRuntime(
        name, selected_dir, full_dir, seed,
        root / config["groups"] if config.get("groups") else None)
    saved = read(selected_dir / f"seed{seed}" / "test_predictions.json")["predictions"]
    original_csv = pd.read_csv
    original_torch_load = torch.load

    def checked_csv(path, *args, **kwargs):
        if Path(path).name == "labels.csv":
            raise AssertionError("classification labels entered the source loader")
        return original_csv(path, *args, **kwargs)

    def checked_torch_load(path, *args, **kwargs):
        if Path(path).name == "labels_bot.pt":
            raise AssertionError("classification labels entered the source loader")
        return original_torch_load(path, *args, **kwargs)

    with patch.object(pd, "read_csv", checked_csv), patch.object(torch, "load", checked_torch_load):
        x, names, ids = load_source_population(
            name, Path(config["data"]), full_dir / f"seed{seed}",
            account_file=root / config["account"] if config.get("account") else None,
            native_facts=root / config["native"] if config.get("native") else None,
            structural_facts=root / config["structural"] if config.get("structural") else None,
            neighbor_facts=root / config["neighbor"] if config.get("neighbor") else None,
            raw_source=root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
            if name == "fox8" else None,
        )
    index_by_id = {account: i for i, account in enumerate(ids)}
    account_ids = [row["account_id"] for row in saved]
    if len(set(account_ids)) != len(account_ids) or not set(account_ids) <= set(index_by_id):
        raise ValueError("saved accounts are not in the unlabeled source population")
    test = np.asarray([index_by_id[account] for account in account_ids], dtype=int)
    predictions = runtime.predict(x[test], names, account_ids)
    card_path = selected_dir / f"seed{seed}" / config["cards"]
    maximum_probability = maximum_contribution = maximum_source_value = 0.0
    with gzip.open(card_path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= len(predictions) or index >= len(saved):
                raise ValueError("saved prediction population differs")
            current = predictions[index]
            prior = saved[index]
            card = json.loads(line)
            if (current["account_id"] != prior["account_id"]
                    or current["account_id"] != card["account_id"]
                    or current["prediction"] != card["prediction"]
                    or current["reference_version"] != card.get(
                        "concept_version", card.get("reference_version"))
                    or len(current["terms"]) != len(card[config["terms"]])):
                raise ValueError("loaded runtime changes fixed account or decision")
            maximum_probability = max(maximum_probability,
                                      abs(current["probability"] - prior["probability"]),
                                      abs(current["probability"] - card["probability"]))
            for actual, expected in zip(current["terms"], card[config["terms"]], strict=True):
                if actual["name"] != expected["name"]:
                    raise ValueError("runtime concept name differs")
                if name != "mgtab" and actual["source_fact_name"] != expected["winning_atomic_fact"]:
                    raise ValueError("runtime selected a different source fact")
                if name == "mgtab" and actual["source_fact_name"] != expected["name"]:
                    raise ValueError("MGTAB direct source fact differs")
                source_value = expected.get("raw_source_value", expected.get("source_value"))
                if source_value is not None:
                    maximum_source_value = max(maximum_source_value,
                                               abs(actual["source_fact_value"] - source_value))
                maximum_contribution = max(maximum_contribution,
                                           abs(actual["logit_contribution"] -
                                               expected["logit_contribution"]))
        if index + 1 != len(predictions) or index + 1 != len(saved):
            raise ValueError("runtime did not cover every saved account")
    if max(maximum_probability, maximum_contribution, maximum_source_value) > 1e-10:
        raise ValueError("loaded runtime does not replay the selected model")
    return {"seed": seed, "test_accounts": len(predictions),
            "classification_label_file_access_guard_passed": True,
            "max_probability_difference": maximum_probability,
            "max_contribution_difference": maximum_contribution,
            "max_source_fact_difference": maximum_source_value,
            "status": "frozen_inference_runtime_replays_all_saved_test_predictions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": "three_dataset_label_free_runtime_replay",
              "datasets": [
                  {"dataset": name, "runs": [
                      verify_seed(args.project_root, name, config, seed)
                      for seed in (0, 1, 2)]}
                  for name, config in CONFIG.items()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
