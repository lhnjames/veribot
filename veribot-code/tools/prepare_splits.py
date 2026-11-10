"""Freeze reproducible, account-level 70/10/20 splits for the public datasets.

This follows the supplied preprocessing scripts' seed-0 per-class shuffle and
records IDs and source checksums so all methods use identical populations.
No statistics or thresholds are fitted here; those must use TRAIN only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_split(name: str, ids: list[str], labels: list[int], source: Path) -> None:
    if len(ids) != len(labels):
        raise ValueError(f"{name}: label mismatch")
    id_labels = {}
    for account_id, label in zip(ids, labels):
        if account_id in id_labels and id_labels[account_id] != label:
            raise ValueError(f"{name}: conflicting labels for account {account_id}")
        id_labels[account_id] = label
    y = np.asarray(labels)
    rng = np.random.default_rng(0)
    split_indices = {"train": [], "dev": [], "test": []}
    for cls in (0, 1):
        index = np.where(y == cls)[0]
        rng.shuffle(index)
        a, b = int(0.7 * len(index)), int(0.8 * len(index))
        for split, part in (("train", index[:a]), ("dev", index[a:b]), ("test", index[b:])):
            split_indices[split].extend(part.tolist())
    # The official Fox8 release repeats one human account in two source subsets.
    # Keep both records but never let that identity cross split boundaries.
    assignment = {i: split for split, indices in split_indices.items() for i in indices}
    first_by_id = {}
    id_counts = Counter(ids)
    for i, account_id in enumerate(ids):
        if account_id in first_by_id:
            assignment[i] = assignment[first_by_id[account_id]]
        else:
            first_by_id[account_id] = i
    target_sizes = {split: len(indices) for split, indices in split_indices.items()}
    for split in ("train", "dev", "test"):
        while sum(assignment[i] == split for i in range(len(ids))) > target_sizes[split]:
            receiver = next(s for s in ("train", "dev", "test")
                            if sum(assignment[i] == s for i in range(len(ids))) < target_sizes[s])
            candidate = next(i for i in range(len(ids) - 1, -1, -1)
                             if assignment[i] == split and id_counts[ids[i]] == 1 and
                             sum(labels[j] == labels[i] and assignment[j] == receiver
                                 for j in range(len(ids))) <
                             sum(labels[j] == labels[i] and j in split_indices[receiver]
                                 for j in range(len(ids))))
            assignment[candidate] = receiver
    split_indices = {split: [i for i in range(len(ids)) if assignment[i] == split]
                     for split in ("train", "dev", "test")}
    if len(set(sum(split_indices.values(), []))) != len(ids):
        raise ValueError(f"{name}: split overlap or omission")
    id_sets = {split: {ids[i] for i in index} for split, index in split_indices.items()}
    if any(id_sets[a] & id_sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError(f"{name}: an account ID occurs in more than one split")
    output = {
        "dataset": name, "source_sha256": sha256(source), "seed": 0,
        "rule": "numpy.default_rng(0), shuffle class 0 then class 1, floor 70/80 percent cutoffs; duplicate IDs grouped and class-balanced sizes restored",
        "label_convention": {"human": 0, "bot": 1},
        "splits": {
            split: {"indices": index, "account_ids": [ids[i] for i in index],
                    "counts": {"human": sum(labels[i] == 0 for i in index),
                               "bot": sum(labels[i] == 1 for i in index)}}
            for split, index in split_indices.items()
        },
    }
    target = ROOT / "splits" / f"{name}_seed0.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(name, {k: len(v) for k, v in split_indices.items()}, "->", target)


def main() -> None:
    fox = ROOT / "raw" / "fox8_23" / "fox8_23_dataset.ndjson"
    fox_ids, fox_labels = [], []
    with fox.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            fox_ids.append(str(row["user_id"]))
            fox_labels.append(int(row["label"] == "bot"))
    write_split("fox8_23", fox_ids, fox_labels, fox)

    botsim = ROOT / "raw" / "botsim_24" / "Users.csv"
    with botsim.open(encoding="utf-8-sig", newline="") as source:
        users = list(csv.DictReader(source))
    if len(users) != 2907:
        raise ValueError(f"Expected 2907 BotSim users, got {len(users)}")
    write_split("botsim_24", [u["user_id"] for u in users], [0] * 1907 + [1] * 1000, botsim)


if __name__ == "__main__":
    main()
