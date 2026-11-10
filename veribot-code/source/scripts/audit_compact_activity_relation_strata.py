"""Audit fixed compact evidence models across train-defined source strata.

This is a descriptive subgroup analysis. It does not select a new model,
threshold, evidence rule, or test population.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import load


CONFIG = {
    "botsim": {
        "data": "D:/Code/SAHG/SAHG/code/data/botsim_24",
        "account_file": "data/raw/botsim_24/Users.csv",
        "native_graph": "data/processed/botsim_v3/native_graph_facts.npz",
        "candidate": "output/v3_computable/botsim_group_tail_12",
        "baseline": "output/hypbot_botsim_paired",
    },
    "fox8": {
        "data": "D:/Code/SAHG/SAHG/code/data/fox8_23",
        "native_graph": "data/processed/fox8_v3/native_interaction_facts.npz",
        "candidate": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
        "baseline": "output/hypbot_fox8_paired",
    },
    "mgtab": {
        "data": "D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB",
        "candidate": "output/v3_mgtab_ab_spline_content/direct_selected_20",
        "baseline": "output/hypbot_mgtab_paired",
    },
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_axes(root: Path, dataset: str, x: np.ndarray, ids: list[str],
             names: list[str], config: dict) -> dict[str, np.ndarray]:
    col = {name: x[:, index] for index, name in enumerate(names)}
    if dataset == "botsim":
        return {
            "recorded_activity": col["n_posts"] + col["n_comments_made"],
            "native_reply_unique_neighbors": np.expm1(col["native_reply_out_unique_log1p"])
                                            + np.expm1(col["native_reply_in_unique_log1p"]),
        }
    if dataset == "fox8":
        cache_path = root / config["native_graph"]
        with np.load(cache_path, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("Fox8 native interaction cache account order differs")
            graph = {name: cache["values"][:, i].astype(float)
                     for i, name in enumerate(cache["names"].astype(str))}
        return {
            "sampled_tweet_count": col["n_tweets"],
            "native_reply_retweet_events": sum(
                np.expm1(graph[name]) for name in (
                    "native_reply_out_tweet_count", "native_reply_in_tweet_count",
                    "native_retweet_out_tweet_count", "native_retweet_in_tweet_count"
                )
            ),
        }
    if dataset == "mgtab":
        return {
            "all_published_relation_edges": sum(
                np.expm1(col[name]) for name in names if name.endswith("degree_log1p")
            ),
            "published_reply_edges": np.expm1(col["published_reply_out_degree_log1p"])
                                   + np.expm1(col["published_reply_in_degree_log1p"]),
        }
    raise ValueError(dataset)


def ordered_predictions(path: Path, ids: list[str], labels: np.ndarray,
                        test_indices: np.ndarray, split_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    artifact = read(path)
    manifest = artifact["manifest"]
    if manifest["split_sha256"] != split_sha256:
        raise ValueError(f"split mismatch: {path}")
    rows = artifact["predictions"]
    mapping = {row["account_id"]: row for row in rows}
    expected_ids = [ids[int(i)] for i in test_indices]
    if len(mapping) != len(rows) or set(mapping) != set(expected_ids):
        raise ValueError(f"account population mismatch: {path}")
    if any(mapping[account]["label"] != int(labels[int(index)])
           for account, index in zip(expected_ids, test_indices)):
        raise ValueError(f"account label mismatch: {path}")
    probability = np.array([mapping[account]["probability"] for account in expected_ids], float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"invalid probability: {path}")
    return probability, probability >= float(manifest["threshold"])


def evaluate(mask: np.ndarray, labels: np.ndarray, candidate: np.ndarray,
             baseline: np.ndarray, seed: int, stratum_index: int) -> dict:
    y = labels[mask]
    c = candidate[mask]
    b = baseline[mask]
    n = len(y)
    if n == 0:
        return {"n": 0, "bot_n": 0, "human_n": 0,
                "candidate_acc": None, "hypbot_acc": None,
                "paired_acc_difference": None, "paired_bootstrap_95pct_ci": None,
                "candidate_macro_f1": None, "hypbot_macro_f1": None}
    cc = c == y
    bb = b == y
    diff = cc.astype(float) - bb.astype(float)
    rng = np.random.default_rng(20261104 + 100 * seed + stratum_index)
    bootstrap = np.mean(diff[rng.integers(n, size=(4000, n))], axis=1)
    return {
        "n": n, "bot_n": int(y.sum()), "human_n": int(n - y.sum()),
        "candidate_acc": float(cc.mean()), "hypbot_acc": float(bb.mean()),
        "paired_acc_difference": float(diff.mean()),
        "paired_bootstrap_95pct_ci": [float(v) for v in np.quantile(bootstrap, [0.025, 0.975])],
        "candidate_macro_f1": float(f1_score(y, c, labels=[0, 1], average="macro", zero_division=0)),
        "hypbot_macro_f1": float(f1_score(y, b, labels=[0, 1], average="macro", zero_division=0)),
        "candidate_only_correct": int(np.sum(cc & ~bb)),
        "hypbot_only_correct": int(np.sum(bb & ~cc)),
    }


def audit_dataset(root: Path, dataset: str, config: dict) -> dict:
    runs = []
    for seed in (0, 1, 2):
        x, labels, ids, names, split, manifest = load(
            dataset, Path(config["data"]),
            root / config["account_file"] if config.get("account_file") else None,
            seed,
            root / config["native_graph"] if dataset == "botsim" else None,
        )
        axes = raw_axes(root, dataset, x, ids, names, config)
        test = split["test"]
        y = labels[test]
        candidate_prob, candidate_pred = ordered_predictions(
            root / config["candidate"] / f"seed{seed}" / "test_predictions.json",
            ids, labels, test, manifest["split_sha256"])
        baseline_prob, baseline_pred = ordered_predictions(
            root / config["baseline"] / f"seed{seed}" / "test_predictions.json",
            ids, labels, test, manifest["split_sha256"])
        axis_rows = []
        low_masks = []
        for axis_index, (name, values) in enumerate(axes.items()):
            threshold = float(np.median(values[split["train"]]))
            low = values[test] <= threshold
            low_masks.append(low)
            axis_rows.append({
                "axis": name, "train_median_threshold": threshold,
                "low": evaluate(low, y, candidate_pred, baseline_pred, seed, axis_index * 2 + 1),
                "high": evaluate(~low, y, candidate_pred, baseline_pred, seed, axis_index * 2 + 2),
            })
        runs.append({
            "seed": seed, "split_sha256": manifest["split_sha256"],
            "test_n": len(test),
            "overall": evaluate(np.ones(len(test), bool), y, candidate_pred, baseline_pred, seed, 0),
            "axes": axis_rows,
            "both_axes_low": evaluate(low_masks[0] & low_masks[1], y,
                                      candidate_pred, baseline_pred, seed, 5),
            "both_axes_high": evaluate(~low_masks[0] & ~low_masks[1], y,
                                       candidate_pred, baseline_pred, seed, 6),
        })
    return {"dataset": dataset, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "fixed_model_train_median_source_strata",
        "interpretation": "descriptive in-distribution subgroup audit; overlapping axes; not OOD or causal proof",
        "selection": "models and probability thresholds fixed; source-axis thresholds use train only",
        "datasets": [audit_dataset(args.project_root, name, config)
                     for name, config in CONFIG.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for dataset in result["datasets"]:
        print(dataset["dataset"])
        for run in dataset["runs"]:
            print(run["seed"], [(row["axis"], row["low"]["n"],
                                 round(row["low"]["paired_acc_difference"] or 0, 4),
                                 row["high"]["n"],
                                 round(row["high"]["paired_acc_difference"] or 0, 4))
                                for row in run["axes"]])


if __name__ == "__main__":
    main()
