"""Classify BotSim displayed reasons by literal summary or object-backed source."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/botsim_24"
CARDS = ROOT / "output/v3_best_decision_reasons"
CERT = ROOT / "research/certificates/botsim_summary_evidence_scope.json"
SUMMARY_FIELDS = ("submission_num", "comment_num", "comment_num_1", "comment_num_2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    card_cert = ROOT / "research/certificates/best_decision_reasons_full_test.json"
    manifest = json.loads(card_cert.read_text(encoding="utf-8"))
    with (RAW / "Users.csv").open(encoding="utf-8-sig", newline="") as stream:
        users = {row["user_id"]: row for row in csv.DictReader(stream)}
    activity = json.loads((RAW / "user_post_comment.json").read_text(encoding="utf-8"))
    if set(users) != set(activity) or len(users) != 2907:
        raise ValueError("source account population differs")
    mismatched = {}
    for uid, row in users.items():
        record = activity[uid]
        observed = (len(record["posts"]),
                    len(record["comment_1"]) + len(record["comment_2"]),
                    len(record["comment_1"]), len(record["comment_2"]))
        mismatched[uid] = {name for name, count in zip(SUMMARY_FIELDS, observed)
                           if float(row[name]) != count}
    runs = []
    for seed in (0, 1, 2):
        run = next(row for row in manifest["runs"] if row["dataset"] == "botsim"
                   and int(row["seed"]) == seed)
        path = CARDS / f"botsim_seed{seed}.jsonl.gz"
        if sha256(path) != run["reason_cards_sha256"]:
            raise ValueError("reason-card source hash differs")
        levels = Counter()
        summary_facts = Counter()
        reasons_with_mismatched_summary = 0
        decisions_with_mismatched_summary = 0
        critical_mismatched_summary = 0
        affected_source_accounts = 0
        accounts = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                card = json.loads(line)
                uid = card["account_id"]
                if uid not in users:
                    raise ValueError("card account lacks raw source")
                affected_source_accounts += bool(mismatched[uid])
                account_has_mismatch_reason = False
                for item in card["displayed_items"]:
                    level, fact = item["source_level"], item["source_fact"]
                    levels[level] += 1
                    if level == "published_user_summary":
                        if fact not in SUMMARY_FIELDS or item["source_query"][
                                "operation"] != "read_declared_user_summary_count":
                            raise ValueError("summary fact has incorrect source contract")
                        record = activity[uid]
                        object_count = {
                            "submission_num": len(record["posts"]),
                            "comment_num": len(record["comment_1"]) + len(record["comment_2"]),
                            "comment_num_1": len(record["comment_1"]),
                            "comment_num_2": len(record["comment_2"]),
                        }[fact]
                        expected_status = (
                            "object_count_differs_from_declared_summary"
                            if fact in mismatched[uid] else
                            "object_count_matches_declared_summary")
                        if (item["source_query"]["activity_object_count"] != object_count
                                or item["source_query"]["cross_source_status"] !=
                                expected_status):
                            raise ValueError("displayed summary cross-source status differs")
                        summary_facts[fact] += 1
                        if fact in mismatched[uid]:
                            reasons_with_mismatched_summary += 1
                            critical_mismatched_summary += item["role"] == "class_critical"
                            account_has_mismatch_reason = True
                    elif fact in SUMMARY_FIELDS:
                        raise ValueError("summary fact incorrectly claims object backing")
                decisions_with_mismatched_summary += account_has_mismatch_reason
                accounts += 1
        if accounts != run["accounts"]:
            raise ValueError("card population differs")
        runs.append({
            "seed": seed, "accounts": accounts,
            "accounts_with_any_summary_activity_disagreement": affected_source_accounts,
            "displayed_source_level_counts": dict(levels),
            "displayed_summary_fact_counts": dict(summary_facts),
            "displayed_summary_facts_with_activity_count_disagreement":
                reasons_with_mismatched_summary,
            "decisions_displaying_disagreed_summary_count":
                decisions_with_mismatched_summary,
            "class_critical_disagreed_summary_counts": critical_mismatched_summary,
        })
    result = {
        "status": "summary_counts_distinguished_from_recountable_activity_objects",
        "source_sha256": {name: sha256(RAW / name)
                          for name in ("Users.csv", "user_post_comment.json")},
        "reason_card_certificate_sha256": sha256(card_cert),
        "interpretation": (
            "A disagreement identifies different published quantities, not a proven "
            "mistake: the activity JSON may omit events counted in Users.csv. "
            "Summary facts are verifiable as CSV declarations, not independently "
            "recounted complete histories."),
        "runs": runs,
    }
    CERT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps(runs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
