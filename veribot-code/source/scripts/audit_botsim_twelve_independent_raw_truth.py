"""Recompute BotSim twelve-concept source facts without training feature functions."""

from __future__ import annotations

import ast
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_source_audit import audit_all


ROOT = Path(__file__).resolve().parents[2]
USERS = ROOT / "data/raw/botsim_24/Users.csv"
ACTIVITY = ROOT / "data/raw/botsim_24/user_post_comment.json"
NATIVE_DIR = ROOT / "data/processed/botsim_v3"
FOLDER = ROOT / "output/v3_computable/botsim_group_tail_12"
OUT = ROOT / "research/certificates/botsim_twelve_independent_raw_truth.json"
BASE_NAMES = (
    "submission_num", "comment_num", "comment_num_1", "comment_num_2",
    "has_description", "description_len", "n_subreddits", "n_posts",
    "n_comments_received", "n_comments_made", "avg_post_score",
    "avg_upvote_ratio", "avg_post_len", "avg_comment_score",
    "avg_comment_len", "n_unique_subs", "is_active")
NATIVE_NAMES = (
    "native_reply_out_unique_log1p", "native_reply_in_unique_log1p",
    "native_reply_mutual_unique_log1p", "native_reply_out_comments_log1p",
    "native_reply_in_comments_log1p")
NAMES = BASE_NAMES + NATIVE_NAMES


def number(value: object) -> float:
    if value is None or str(value).strip().lower() in {"", "nan"}:
        return 0.0
    return float(value)


def raw_account(row: dict, record: dict) -> list[float]:
    posts = record.get("posts", [])
    comments1 = record.get("comment_1", [])
    comments2 = record.get("comment_2", [])
    description = row.get("description", "")
    if description is None or str(description).lower() == "nan":
        description = ""
    subreddits = row.get("subreddit", "")
    try:
        parsed = ast.literal_eval(subreddits) if subreddits else []
    except (SyntaxError, ValueError):
        parsed = []
    scores = [post.get("score", 0) or 0 for post in posts]
    received = [post.get("num_comments", 0) or 0 for post in posts]
    upvotes = [post.get("upvote_ratio", .5) or .5 for post in posts]
    post_lengths = [len(post.get("posts", "") or "") for post in posts]
    comment_scores = [item.get("comment_score", 0) or 0 for item in comments1]
    comment_lengths = [len(item.get("comment_body", "") or "")
                       for item in comments1]
    values = (
        number(row.get("submission_num")), number(row.get("comment_num")),
        number(row.get("comment_num_1")), number(row.get("comment_num_2")),
        float(bool(description)), float(len(description)), float(len(parsed)),
        float(len(posts)), float(int(sum(received))) if received else 0.0,
        float(len(comments1) + len(comments2)),
        float(sum(scores)/len(scores)) if scores else 0.0,
        float(sum(upvotes)/len(upvotes)) if upvotes else .5,
        float(sum(post_lengths)/len(post_lengths)) if post_lengths else 0.0,
        float(sum(comment_scores)/len(comment_scores)) if comment_scores else 0.0,
        float(sum(comment_lengths)/len(comment_lengths)) if comment_lengths else 0.0,
        float(len({post.get("subreddit", "") for post in posts})),
        float(bool(posts or comments1 or comments2)),
    )
    return [float(np.float32(value)) for value in values]


def main() -> None:
    native_audit = audit_all(USERS, ACTIVITY, NATIVE_DIR)
    if (native_audit["status"] != "verified_full_population_against_raw_comments"
            or native_audit["verified_fact_cells"] != 2907 * 5):
        raise ValueError("BotSim independent raw reply audit differs")
    with USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    ids = [row["user_id"] for row in users]
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    if len(ids) != 2907 or len(set(ids)) != len(ids) or set(activity) != set(ids):
        raise ValueError("BotSim source population differs")
    base = np.asarray([raw_account(row, activity[row["user_id"]])
                       for row in users], dtype=np.float32)
    with np.load(NATIVE_DIR / "native_graph_facts.npz", allow_pickle=False) as cache:
        if (cache["ids"].astype(str).tolist() != ids
                or cache["names"].astype(str).tolist() != list(NATIVE_NAMES)):
            raise ValueError("BotSim verified native fact vocabulary differs")
        native = cache["values"].copy()
    truth = np.column_stack((base, native))
    report0 = json.loads((FOLDER / "seed0/report.json").read_text(encoding="utf-8"))
    feature_paths = [Path(path) for path in report0["manifest"]["source_files_sha256"]
                     if Path(path).name == "features.csv"]
    if len(feature_paths) != 1:
        raise ValueError("BotSim published feature path ambiguous")
    with feature_paths[0].open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or []) != BASE_NAMES:
            raise ValueError("BotSim published feature columns differ")
        cached = np.asarray([[float(np.float32(row[name])) for name in BASE_NAMES]
                             for row in reader], dtype=np.float32)
    if cached.shape != base.shape:
        raise ValueError("BotSim feature population size differs")
    max_base_difference = float(np.max(np.abs(base - cached)))
    if max_base_difference > 1e-6:
        raise ValueError("BotSim model-used base facts differ from raw source")
    runs = []
    index = {uid: i for i, uid in enumerate(ids)}
    for seed in (0, 1, 2):
        card_path = FOLDER / f"seed{seed}/concept_evidence_cards.jsonl.gz"
        pred_path = FOLDER / f"seed{seed}/test_predictions.json"
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))["predictions"]
        available = 0
        max_winner_difference = max_probability_difference = 0.0
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            cards = [json.loads(line) for line in stream]
        if len(cards) != len(predictions) or len(cards) != 437:
            raise ValueError("BotSim fixed test account count differs")
        for card, prediction in zip(cards, predictions, strict=True):
            uid = card["account_id"]
            if (uid != prediction["account_id"]
                    or len(card["concepts"]) != 12):
                raise ValueError("BotSim 12-concept card order differs")
            max_probability_difference = max(max_probability_difference,
                abs(card["probability"] - prediction["probability"]))
            for concept in card["concepts"]:
                name = concept["winning_atomic_fact"]
                if name is None:
                    continue
                if concept["source_locator"] is None:
                    raise ValueError("BotSim winning fact lacks locator")
                expected = truth[index[uid], NAMES.index(name)]
                if not np.isfinite(expected):
                    raise ValueError("BotSim winning fact absent from raw")
                max_winner_difference = max(max_winner_difference,
                    abs(float(concept["raw_source_value"]) - float(expected)))
                available += 1
        if max_winner_difference > 1e-6 or max_probability_difference > 1e-12:
            raise ValueError("BotSim raw fact-to-concept-to-decision chain differs")
        runs.append({
            "seed": seed,
            "raw_accounts_checked": len(ids),
            "raw_model_fact_cells_checked": int(truth.size),
            "test_accounts_checked": len(cards),
            "test_concept_slots_checked": len(cards) * 12,
            "available_winners_checked": available,
            "max_base_fact_difference": max_base_difference,
            "max_winner_difference": max_winner_difference,
            "max_card_probability_difference": max_probability_difference,
            "cards_sha256": sha256_file(card_path),
            "test_predictions_sha256": sha256_file(pred_path),
        })
        print(runs[-1], flush=True)
    OUT.write_text(json.dumps({
        "status": "botsim_twelve_concepts_recomputed_from_raw_users_activity_and_replies",
        "users_sha256": sha256_file(USERS),
        "activity_sha256": sha256_file(ACTIVITY),
        "native_source_audit": native_audit,
        "feature_preprocessing_functions_imported": False,
        "semantic_bot_behavior_validity_inferred": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
