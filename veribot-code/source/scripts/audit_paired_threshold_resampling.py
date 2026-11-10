"""Fixed-weight paired ACC sensitivity to validation-threshold resampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v2_detector import file_sha256, load_data
from verify_compact_inference_runtime import CONFIG
from verifiable_bot.compact_evidence_runtime import CompactEvidenceRuntime
from verifiable_bot.compact_source_population import load_source_population


def choose_threshold(probability: np.ndarray, labels: np.ndarray,
                     *, allow_all_negative: bool) -> float:
    """Vectorized equivalent of each model's saved validation Macro-F1 rule."""
    values = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=int)
    candidates = np.unique(np.r_[0.5, values,
                                  [np.nextafter(1.0, 2.0)] if allow_all_negative else []])
    order = np.argsort(values)
    sorted_values = values[order]
    prefix_positive = np.r_[0, np.cumsum(y[order])]
    below = np.searchsorted(sorted_values, candidates, side="left")
    false_negative = prefix_positive[below]
    true_positive = int(y.sum()) - false_negative
    false_positive = len(y) - below - true_positive
    true_negative = below - false_negative
    bot_f1 = np.divide(2 * true_positive,
                       2 * true_positive + false_positive + false_negative,
                       out=np.zeros_like(candidates),
                       where=(2 * true_positive + false_positive + false_negative) > 0)
    human_f1 = np.divide(2 * true_negative,
                         2 * true_negative + false_positive + false_negative,
                         out=np.zeros_like(candidates),
                         where=(2 * true_negative + false_positive + false_negative) > 0)
    scores = (bot_f1 + human_f1) / 2
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(candidates[tied[np.argmin(abs(candidates[tied] - 0.5))]])


def _hypbot_probabilities(dataset: str, config: dict, seed: int,
                          root: Path, saved: dict
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    data_dir = Path(config["data"])
    account_file = root / config["account"] if config.get("account") else None
    placeholder = data_dir / ("features.pt" if dataset == "mgtab" else "features.csv")
    args = SimpleNamespace(dataset=dataset, data_dir=data_dir,
                           account_file=account_file, knn_k=10,
                           raw_source=[placeholder])
    x, labels, split, ids, edge, _, hashes, _ = load_data(args, seed)
    manifest = saved["manifest"]
    checkpoint_path = root / f"output/hypbot_{dataset}_paired/seed{seed}/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (manifest["dataset"] != dataset or manifest["seed"] != seed
            or file_sha256(checkpoint_path) != manifest["model_sha256"]
            or any(hashes[key] != manifest[key]
                   for key in ("feature_sha256", "graph_sha256", "split_sha256"))):
        raise ValueError("HypBot checkpoint, graph, features or split differ")
    repo = Path(manifest["official_repo"])
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    from hypbot.data import knn_graph, mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS
    if dataset == "mgtab":
        adjacency = mean_adjacency(edge, len(x))
        extra = {"graph_width": x.shape[1], "sparse_adj": True}
    else:
        adjacency = knn_graph(x, k=10)
        if not torch.equal(adjacency, edge):
            raise ValueError("HypBot kNN graph differs")
        extra = {}
    model = HypBot(x.shape[1], K=checkpoint["num_sectors"], gated=False,
                   **PRESETS[dataset]["model"], **extra)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        probability = model.predict_prob(x, adjacency).cpu().numpy()
    return probability, labels, split, ids, hashes


def audit_seed(root: Path, dataset: str, config: dict, seed: int,
               draws: int) -> dict:
    selected_dir = root / config["selected"]
    full_dir = root / config["full"]
    candidate_path = selected_dir / f"seed{seed}" / "test_predictions.json"
    baseline_path = root / f"output/hypbot_{dataset}_paired/seed{seed}/test_predictions.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if candidate["manifest"]["split_sha256"] != baseline["manifest"]["split_sha256"]:
        raise ValueError("paired split differs")
    baseline_prob, labels, split, ids, hashes = _hypbot_probabilities(
        dataset, config, seed, root, baseline)
    runtime = CompactEvidenceRuntime(dataset, selected_dir, full_dir, seed,
                                     root / config["groups"] if config.get("groups") else None)
    facts, fact_names, source_ids = load_source_population(
        dataset, Path(config["data"]), full_dir / f"seed{seed}",
        account_file=root / config["account"] if config.get("account") else None,
        native_facts=root / config["native"] if config.get("native") else None,
        structural_facts=root / config["structural"] if config.get("structural") else None,
        neighbor_facts=root / config["neighbor"] if config.get("neighbor") else None,
        raw_source=root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
        if dataset == "fox8" else None)
    source_by_id = {account_id: i for i, account_id in enumerate(source_ids)}
    if len(source_by_id) != len(source_ids) or set(ids) != set(source_ids):
        raise ValueError("source and HypBot account populations differ")
    all_indices = np.r_[split["val"], split["test"]]
    account_ids = [ids[int(i)] for i in all_indices]
    selected_rows = np.asarray([source_by_id[account] for account in account_ids], dtype=int)
    candidate_rows = runtime.predict(facts[selected_rows], fact_names, account_ids)
    candidate_prob = np.asarray([row["probability"] for row in candidate_rows])
    n_val = len(split["val"])
    candidate_val, candidate_test = candidate_prob[:n_val], candidate_prob[n_val:]
    baseline_val, baseline_test = baseline_prob[split["val"]], baseline_prob[split["test"]]
    val_labels, test_labels = labels[split["val"]], labels[split["test"]]
    candidate_saved = {row["account_id"]: row for row in candidate["predictions"]}
    baseline_saved = {row["account_id"]: row for row in baseline["predictions"]}
    if (len(candidate_saved) != len(split["test"]) or
            len(baseline_saved) != len(split["test"])):
        raise ValueError("saved test population differs")
    for offset, index in enumerate(split["test"]):
        account_id = ids[int(index)]
        if (int(candidate_saved[account_id]["label"]) != int(test_labels[offset])
                or int(baseline_saved[account_id]["label"]) != int(test_labels[offset])
                or abs(candidate_test[offset] - candidate_saved[account_id]["probability"]) > 1e-10
                or abs(baseline_test[offset] - baseline_saved[account_id]["probability"]) > 1e-5):
            raise ValueError("frozen test probability or labels did not replay")
    candidate_locked = choose_threshold(candidate_val, val_labels,
                                        allow_all_negative=True)
    baseline_locked = choose_threshold(baseline_val, val_labels,
                                       allow_all_negative=False)
    if (abs(candidate_locked - candidate["manifest"]["threshold"]) > 1e-10
            or abs(baseline_locked - baseline["manifest"]["threshold"]) > 1e-5):
        raise ValueError("saved validation threshold did not replay")
    rng = np.random.default_rng(20261103 + seed)
    candidate_thresholds = np.empty(draws)
    baseline_thresholds = np.empty(draws)
    for draw in range(draws):
        sampled = rng.integers(0, n_val, n_val)
        candidate_thresholds[draw] = choose_threshold(
            candidate_val[sampled], val_labels[sampled], allow_all_negative=True)
        baseline_thresholds[draw] = choose_threshold(
            baseline_val[sampled], val_labels[sampled], allow_all_negative=False)
    candidate_acc = np.mean((candidate_test[None, :] >= candidate_thresholds[:, None])
                            == test_labels[None, :], axis=1)
    baseline_acc = np.mean((baseline_test[None, :] >= baseline_thresholds[:, None])
                           == test_labels[None, :], axis=1)
    gap = candidate_acc - baseline_acc
    locked_gap = float(np.mean((candidate_test >= candidate_locked) == test_labels) -
                       np.mean((baseline_test >= baseline_locked) == test_labels))
    return {
        "seed": seed, "validation_accounts": n_val, "test_accounts": len(test_labels),
        "split_sha256": hashes["split_sha256"],
        "candidate_test_predictions_sha256": file_sha256(candidate_path),
        "hypbot_test_predictions_sha256": file_sha256(baseline_path),
        "candidate_locked_threshold": candidate_locked,
        "hypbot_locked_threshold": baseline_locked,
        "locked_paired_acc_gap": locked_gap,
        "candidate_threshold_bootstrap_5_50_95": [float(v) for v in
            np.quantile(candidate_thresholds, [0.05, 0.5, 0.95])],
        "hypbot_threshold_bootstrap_5_50_95": [float(v) for v in
            np.quantile(baseline_thresholds, [0.05, 0.5, 0.95])],
        "paired_acc_gap_bootstrap_2_5_50_97_5": [float(v) for v in
            np.quantile(gap, [0.025, 0.5, 0.975])],
        "fraction_resampled_thresholds_candidate_better": float(np.mean(gap > 0)),
        "fraction_resampled_thresholds_tied": float(np.mean(gap == 0)),
        "fraction_resampled_thresholds_candidate_worse": float(np.mean(gap < 0)),
        "candidate_acc_at_0_5": float(np.mean((candidate_test >= 0.5) == test_labels)),
        "hypbot_acc_at_0_5": float(np.mean((baseline_test >= 0.5) == test_labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("at least 100 validation resamples required")
    datasets = []
    for dataset, config in CONFIG.items():
        runs = [audit_seed(args.project_root, dataset, config, seed, args.draws)
                for seed in (0, 1, 2)]
        datasets.append({"dataset": dataset, "runs": runs})
        print(dataset, [{"seed": row["seed"],
                         "locked_gap": row["locked_paired_acc_gap"],
                         "fraction_better": row[
                             "fraction_resampled_thresholds_candidate_better"]}
                        for row in runs], flush=True)
    result = {"scope": "fixed_weight_validation_threshold_resampling",
              "draws_per_seed": args.draws,
              "test_rows_used_only_for_evaluation_not_threshold_selection": True,
              "training_randomness_or_prior_model_search_covered": False,
              "datasets": datasets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
