"""Audit explicit zero upvote ratios and frozen BotSim dev decisions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.compact_evidence_runtime import CompactEvidenceRuntime
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
USERS = ROOT / "data/raw/botsim_24/Users.csv"
ACTIVITY = ROOT / "data/raw/botsim_24/user_post_comment.json"
DATA = Path("D:/Code/SAHG/SAHG/code/data/botsim_24")
NATIVE = ROOT / "data/processed/botsim_v3/native_graph_facts.npz"
SELECTED = ROOT / "output/v3_computable/botsim_group_tail_12"
FULL = ROOT / "output/v3_computable/botsim_native_graph"
DEPLOYMENT = ROOT / "output/v3_current_compact_deployment"
OUT = ROOT / "research/certificates/botsim_literal_upvote_zero_dev.json"


def literal_ratios(ids: list[str]) -> tuple[np.ndarray, dict]:
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    if set(activity) != set(ids):
        raise ValueError("raw activity population differs")
    means = []
    zero = missing = total = 0
    for uid in ids:
        posts = activity[uid].get("posts", [])
        values = []
        for post in posts:
            total += 1
            value = post.get("upvote_ratio")
            if value is None:
                missing += 1
                value = .5
            elif float(value) == 0:
                zero += 1
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid raw post upvote ratio")
            values.append(value)
        means.append(float(np.float32(sum(values) / len(values)))
                     if values else float(np.float32(.5)))
    return np.asarray(means, dtype=np.float32), {
        "raw_posts": total, "explicit_zero_posts": zero,
        "missing_or_null_posts": missing,
    }


def main() -> None:
    with USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_ids = [row["user_id"] for row in csv.DictReader(stream)]
    literal, raw_stats = literal_ratios(raw_ids)
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        if ids != raw_ids:
            raise ValueError("source order differs from raw Users.csv")
        col = names.index("avg_upvote_ratio")
        actual = x[:, col]
        difference = np.abs(actual - literal)
        if not np.any(difference):
            raise ValueError("expected explicit zero recoding was not found")
        runtime = CompactEvidenceRuntime(
            "botsim", SELECTED, FULL, seed,
            deployment_metadata=DEPLOYMENT / f"botsim_runtime_seed{seed}.json")
        dev = split["val"]
        original = x[dev]
        corrected = original.copy()
        corrected[:, col] = literal[dev]
        old_concepts, old_winners = collapse_group_tails(
            runtime.reference.transform(original), runtime.atomic_groups)
        new_concepts, new_winners = collapse_group_tails(
            runtime.reference.transform(corrected), runtime.atomic_groups)
        old_p = runtime.reasoner.predict_proba(old_concepts)
        new_p = runtime.reasoner.predict_proba(new_concepts)
        report = json.loads((SELECTED / f"seed{seed}/report.json")
                            .read_text(encoding="utf-8"))
        old_metric = evaluate(y[dev], old_p, runtime.threshold)
        if (source["split_sha256"] != report["manifest"]["split_sha256"]
                or old_metric["errors"] != report["validation"]["errors"]
                or abs(old_metric["acc"] - report["validation"]["acc"]) > 1e-12):
            raise ValueError("frozen dev baseline differs")
        old_class = old_p >= runtime.threshold
        new_class = new_p >= runtime.threshold
        winner_names = ("post_engagement:high_tail_max",
                        "post_engagement:low_tail_max")
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "frozen_model_reference_version": runtime.reference.version,
            "dev_accounts": len(dev),
            "all_accounts_with_literal_vs_materialized_difference":
                int(np.count_nonzero(difference)),
            "dev_accounts_with_difference":
                int(np.count_nonzero(difference[dev])),
            "maximum_literal_vs_materialized_difference":
                float(np.max(difference)),
            "post_engagement_winner_changes": {
                name: int(np.count_nonzero(old_winners[name] !=
                                           new_winners[name]))
                for name in winner_names},
            "dev_probability_changed_accounts":
                int(np.count_nonzero(np.abs(old_p - new_p) > 1e-12)),
            "dev_class_changed_accounts":
                int(np.count_nonzero(old_class != new_class)),
            "old_dev": old_metric,
            "literal_zero_dev_with_old_frozen_model": evaluate(
                y[dev], new_p, runtime.threshold),
        })
        print({"seed": seed, "affected_dev": runs[-1][
            "dev_accounts_with_difference"], "class_changes": runs[-1][
            "dev_class_changed_accounts"], "old_errors": old_metric["errors"],
            "literal_errors": runs[-1][
                "literal_zero_dev_with_old_frozen_model"]["errors"]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "explicit_raw_upvote_zero_recode_and_frozen_dev_effect_audited",
        "users_sha256": sha256_file(USERS),
        "activity_sha256": sha256_file(ACTIVITY),
        "raw_post_statistics": raw_stats,
        "current_materialization_rule": "post.get('upvote_ratio', .5) or .5",
        "literal_rule": "preserve explicit zero; use .5 only for missing/null or no posts",
        "test_predictions_or_metrics_accessed": False,
        "frozen_model_retrained_for_literal_rule": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
