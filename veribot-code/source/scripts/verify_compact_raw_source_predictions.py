"""Replay compact decisions from raw BotSim/Fox8 records and MGTAB tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_botsim_full_evidence_cards import raw_matrix as botsim_raw_matrix
from verifiable_bot.compact_evidence_runtime import CompactEvidenceRuntime
from verifiable_bot.compact_source_population import load_source_population
from verifiable_bot.fox8_adapter import index_sources, raw_feature_matrix
from verifiable_bot.mgtab_full_source_audit import audit as audit_mgtab_sources


CONFIG = {
    "botsim": {
        "data": "D:/Code/SAHG/SAHG/code/data/botsim_24",
        "account": "data/raw/botsim_24/Users.csv",
        "activity": "data/raw/botsim_24/user_post_comment.json",
        "native": "data/processed/botsim_v3/native_graph_facts.npz",
        "selected": "output/v3_computable/botsim_group_tail_12",
        "full": "output/v3_computable/botsim_native_graph",
    },
    "fox8": {
        "data": "D:/Code/SAHG/SAHG/code/data/fox8_23",
        "raw": "data/raw/fox8_23/fox8_23_dataset.ndjson.gz",
        "native": "data/processed/fox8_v3/native_interaction_facts.npz",
        "groups": "research/fox8_20_concept_groups.json",
        "selected": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
        "full": "output/v3_fox8_native_spline_l2/selected",
    },
    "mgtab": {
        "data": "D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB",
        "structural": "data/processed/mgtab_v3/structural_facts.npz",
        "neighbor": "data/processed/mgtab_v3/neighbor_facts.npz",
        "selected": "output/v3_mgtab_ab_spline_content/direct_selected_20",
        "full": "output/v3_mgtab_ab_spline_content/selected",
    },
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_inputs(root: Path, dataset: str, config: dict,
                  test_ids: list[str], seed: int = 0,
                  inference_metadata: Path | None = None
                  ) -> tuple[list[str], np.ndarray, float, str]:
    full_dir = root / config["full"]
    data_dir = Path(config["data"])
    x, names, ids = load_source_population(
        dataset, data_dir, full_dir / f"seed{seed}",
        account_file=root / config["account"] if dataset == "botsim" else None,
        native_facts=root / config["native"] if dataset != "mgtab" else None,
        structural_facts=root / config["structural"] if dataset == "mgtab" else None,
        neighbor_facts=root / config["neighbor"] if dataset == "mgtab" else None,
        raw_source=root / config["raw"] if dataset == "fox8" else None,
        inference_metadata=inference_metadata,
    )
    if dataset == "botsim":
        raw_ids, raw_x = botsim_raw_matrix(SimpleNamespace(
            account_file=root / config["account"],
            activity=root / config["activity"],
            native_graph_facts=root / config["native"],
        ))
        if raw_ids != ids or raw_x.shape != x.shape:
            raise ValueError("BotSim raw source account population differs")
        difference = float(np.max(np.abs(raw_x.astype(float) - x.astype(float))))
        if difference > 1e-5:
            raise ValueError("BotSim raw user, post or reply facts differ")
        return names, raw_x, difference, "raw_user_post_comment_and_reply_graph"
    if dataset == "fox8":
        if not set(test_ids) <= set(ids):
            raise ValueError("Fox8 test accounts absent from published feature population")
        selected_ids = [uid for uid in ids if uid in set(test_ids)]
        sources = index_sources(root / config["raw"], selected_ids)
        base = raw_feature_matrix(sources, selected_ids)
        source_index = {uid: i for i, uid in enumerate(ids)}
        raw_x = x.copy()
        positions = [source_index[uid] for uid in selected_ids]
        difference = float(np.max(np.abs(base.astype(float) - raw_x[positions, :31].astype(float))))
        if difference > 1e-5:
            raise ValueError("Fox8 raw profile or tweet facts differ")
        raw_x[positions, :31] = base
        return names, raw_x, difference, "raw_profile_tweets_and_native_interaction_graph"
    if dataset == "mgtab":
        source_audit = audit_mgtab_sources(
            data_dir, root / config["structural"], root / config["neighbor"])
        if (source_audit["status"] != "verified_all_published_tensor_derived_fact_cells"
                or source_audit["account_count"] != len(ids)):
            raise ValueError("MGTAB published tensor source audit differs")
        return names, x, 0.0, "published_profile_edge_and_mean_tweet_tensors"
    raise ValueError(dataset)


def verify_dataset(root: Path, dataset: str, config: dict) -> dict:
    selected_dir = root / config["selected"]
    fixed = {seed: read(selected_dir / f"seed{seed}" / "test_predictions.json")["predictions"]
             for seed in (0, 1, 2)}
    all_test_ids = list(dict.fromkeys(row["account_id"] for rows in fixed.values() for row in rows))
    names, x, source_difference, level = source_inputs(root, dataset, config, all_test_ids)
    # All selected-model IDs use the same released population order; the loader
    # returns that order without any class labels.
    if dataset == "botsim":
        import pandas as pd
        ids = pd.read_csv(root / config["account"])["user_id"].astype(str).tolist()
    elif dataset == "fox8":
        import pandas as pd
        ids = pd.read_csv(Path(config["data"]) / "features.csv")["user_id"].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(x))]
    index = {account: i for i, account in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("source account IDs are not unique")
    runs = []
    for seed in (0, 1, 2):
        if dataset == "mgtab" and seed != 0:
            seed_x, seed_names, seed_ids = load_source_population(
                dataset, Path(config["data"]), root / config["full"] / f"seed{seed}",
                structural_facts=root / config["structural"],
                neighbor_facts=root / config["neighbor"])
            if seed_ids != ids:
                raise ValueError("MGTAB seed-specific projection population differs")
        else:
            seed_x, seed_names = x, names
        runtime = CompactEvidenceRuntime(
            dataset, selected_dir, root / config["full"], seed,
            root / config["groups"] if config.get("groups") else None)
        accounts = [row["account_id"] for row in fixed[seed]]
        rows = np.asarray([index[account] for account in accounts], dtype=int)
        predicted = runtime.predict(seed_x[rows], seed_names, accounts)
        maximum = max(abs(row["probability"] - saved["probability"])
                      for row, saved in zip(predicted, fixed[seed], strict=True))
        if maximum > 1e-10:
            raise ValueError(f"{dataset} seed{seed} raw source changes saved probability")
        runs.append({"seed": seed, "test_accounts": len(accounts),
                     "max_raw_source_to_saved_probability_difference": float(maximum),
                     "decision_mismatches": sum(
                         row["prediction"] != int(saved["probability"] >= runtime.threshold)
                         for row, saved in zip(predicted, fixed[seed], strict=True))})
    return {"dataset": dataset, "source_level": level,
            "max_raw_to_released_fact_difference": source_difference,
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": "raw_or_published_source_to_frozen_compact_prediction",
              "classification_labels_used": False,
              "datasets": [verify_dataset(args.project_root, name, config)
                           for name, config in CONFIG.items()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
