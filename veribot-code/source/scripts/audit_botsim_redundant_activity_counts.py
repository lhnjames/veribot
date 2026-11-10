"""Check BotSim summary counts against the independently released activity objects."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw = ROOT / "data/raw/botsim_24"
    with (raw / "Users.csv").open(encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    activity = json.loads((raw / "user_post_comment.json").read_text(encoding="utf-8"))
    if len(users) != len(activity) or {r["user_id"] for r in users} != set(activity):
        raise ValueError("activity and user populations differ")
    mismatches = {name: [] for name in (
        "submission_num", "comment_num", "comment_num_1", "comment_num_2")}
    distribution = {name: Counter() for name in mismatches}
    nonintegral = {name: 0 for name in mismatches}
    internal_comment_disagreements = []
    cohort_mismatch = {"first_1907": Counter(), "last_1000": Counter()}
    for index, row in enumerate(users):
        uid = row["user_id"]
        record = activity[uid]
        actual = {
            "submission_num": len(record["posts"]),
            "comment_num": len(record["comment_1"]) + len(record["comment_2"]),
            "comment_num_1": len(record["comment_1"]),
            "comment_num_2": len(record["comment_2"]),
        }
        if float(row["comment_num"]) != (float(row["comment_num_1"]) +
                                          float(row["comment_num_2"])):
            internal_comment_disagreements.append(uid)
        for name, count in actual.items():
            declared = float(row[name])
            if not declared.is_integer():
                nonintegral[name] += 1
            delta = int(declared - count) if declared.is_integer() else declared - count
            distribution[name][delta] += 1
            if delta:
                mismatches[name].append({
                    "user_id": uid, "row_index": index, "declared": declared,
                    "activity_count": count, "delta": delta,
                })
                cohort_mismatch["first_1907" if index < 1907 else "last_1000"][name] += 1
    report = {
        "population": len(users),
        "source_files": [str(raw / "Users.csv"), str(raw / "user_post_comment.json")],
        "source_sha256": {name: sha256(raw / name) for name in
                          ("Users.csv", "user_post_comment.json")},
        "mismatch_counts": {name: len(rows) for name, rows in mismatches.items()},
        "internal_comment_num_ne_sum_count": len(internal_comment_disagreements),
        "internal_comment_num_ne_sum_examples": internal_comment_disagreements[:10],
        "mismatch_counts_by_published_row_cohort": {
            key: dict(value) for key, value in cohort_mismatch.items()},
        "nonintegral_counts": nonintegral,
        "delta_distributions": {name: dict(counter.most_common(12))
                                for name, counter in distribution.items()},
        "mismatch_examples": {name: rows[:10] for name, rows in mismatches.items()},
        "all_mismatch_user_ids": sorted({row["user_id"] for rows in mismatches.values()
                                         for row in rows}),
    }
    path = ROOT / "research/certificates/botsim_redundant_activity_counts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "population", "mismatch_counts", "nonintegral_counts",
        "internal_comment_num_ne_sum_count",
        "mismatch_counts_by_published_row_cohort")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
