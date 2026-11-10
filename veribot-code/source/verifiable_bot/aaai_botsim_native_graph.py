"""Raw Users.csv subreddit memberships for VeriBot HGT and RelNet."""

from __future__ import annotations

import ast
import csv
from hashlib import sha256
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_memberships(users_path: Path) -> tuple[list[str], list[str],
                                                 dict[tuple[str, str], dict]]:
    with users_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    account_ids = [str(row["user_id"]) for row in rows]
    if len(set(account_ids)) != len(account_ids):
        raise ValueError("BotSim user IDs are not unique")
    locations = {}
    for row_number, row in enumerate(rows, 2):
        encoded = row.get("subreddit") or "[]"
        parsed = ast.literal_eval(encoded)
        if not isinstance(parsed, list) or any(not isinstance(value, str) for value in parsed):
            raise ValueError(f"BotSim subreddit list is invalid at CSV row {row_number}")
        for position, subreddit in enumerate(parsed):
            if not subreddit:
                raise ValueError("empty subreddit is not a graph entity")
            key = (str(row["user_id"]), subreddit)
            if key in locations:
                raise ValueError("duplicate user-subreddit membership")
            locations[key] = {"csv_row": row_number,
                              "subreddit_list_index": position,
                              "raw_location": (f"{users_path.resolve()}#row={row_number};"
                                               f"subreddit[{position}]")}
    subreddits = sorted({subreddit for _, subreddit in locations})
    return account_ids, subreddits, locations


def build_native_graph(
    users_path: Path, account_ids: list[str],
    excluded_memberships: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[dict[tuple[str, str, str], torch.Tensor],
           dict[str, torch.Tensor], dict]:
    """Build a bipartite graph and exact binary co-subreddit mean operator."""
    raw_ids, subreddits, locations = read_memberships(users_path)
    if raw_ids != account_ids:
        raise ValueError("BotSim raw and model user inventories differ")
    if not excluded_memberships <= locations.keys():
        raise ValueError("membership edit points outside the raw graph")
    user_index = {uid: i for i, uid in enumerate(account_ids)}
    sub_index = {subreddit: i for i, subreddit in enumerate(subreddits)}
    membership = torch.zeros((len(account_ids), len(subreddits)), dtype=torch.float32)
    pairs = []
    for uid, subreddit in locations:
        if (uid, subreddit) in excluded_memberships:
            continue
        i, j = user_index[uid], sub_index[subreddit]
        membership[i, j] = 1.0
        pairs.append((i, j))
    pairs.sort()
    user_sub = (torch.tensor(pairs, dtype=torch.long).T.contiguous()
                if pairs else torch.empty((2, 0), dtype=torch.long))
    if not pairs:
        raise ValueError("BotSim native membership graph is empty")
    # A user pair is adjacent if and only if it shares one or more source
    # subreddits. No subreddit or edge is removed for computational reasons.
    shared = (membership @ membership.T) > 0
    shared.fill_diagonal_(False)
    adjacency = shared.to(torch.float32)
    degree = adjacency.sum(dim=1, keepdim=True)
    adjacency = adjacency / degree.clamp(min=1)
    graph = {
        ("user", "posted_in", "subreddit"): user_sub,
        ("subreddit", "rev_posted_in", "user"): user_sub.flip(0).contiguous(),
    }
    return graph, {"shared_subreddit": adjacency}, {
        "raw_source_sha256": sha256_file(users_path),
        "account_count": len(account_ids),
        "subreddit_names": subreddits,
        "subreddit_count": len(subreddits),
        "membership_count": len(pairs),
        "projected_directed_user_pairs": int(shared.sum()),
        "projected_users_with_neighbors": int((degree[:, 0] > 0).sum()),
        "operation": "raw user-subreddit memberships and exact any-shared-subreddit user projection",
    }
