"""BotSim-24 preprocessing. IMPORTANT: this is a REDDIT dataset (submissions/comments/subreddits),
not Twitter -- structurally different from TwiBot-20/22/Fox8. No followers/friends/retweets exist;
"profile" only has description + submission/comment counts + subreddit list.

Source files (already on js02 at /path/to/data_root/data/botsim/):
  Users.csv             2907 rows: user_id,name,description,submission_num,comment_num,
                         character_setting,comment_num_1,comment_num_2,subreddit
                         FIRST 1907 rows = human, LAST 1000 rows = bot (per official Readme, NOT shuffled)
  user_post_comment.json  per-user {posts:[...], comment_1:[...], comment_2:[...]} raw Reddit text

Graph: not shipped for the full dataset (only mini-sample has edge_index/type/weight). We DERIVE a
reply graph from comment_1/comment_2 parent_id/link_id chains (user A replies to user B's post/comment
-> edge A->B), which is the direct Reddit analogue of TwiBot-20's user-object edges (object = the
post/comment thread), giving us a real, traceable, auditable graph -- closer to TB20 T1/T2 spirit than
MGTAB's opaque edges.

Output tensors: mirrors the TB20-style bundle but domain-appropriate.
"""
import json, os, csv
import numpy as np, torch

D = "/path/to/data_root/data/botsim"
OUT = "/path/to/data_root/data/botsim_split"
os.makedirs(OUT, exist_ok=True)


def load_users():
    rows = list(csv.DictReader(open(f"{D}/Users.csv")))
    # official Readme: first 1907 = human, last 1000 = bot, IN FILE ORDER (do not shuffle before labeling!)
    labels = [0] * 1907 + [1] * 1000
    assert len(rows) == len(labels), f"expected 2907 rows, got {len(rows)}"
    uid2idx = {r["user_id"]: i for i, r in enumerate(rows)}
    return rows, labels, uid2idx


def build_reply_graph(uid2idx):
    """user -> submission/comment id -> author edges, derived from parent_id/link_id chains.
    This is the closest BotSim-24 analogue to TwiBot-20's user->object structure: the 'object' here
    is a submission_id / comment thread, and co-reply structure can seed a T1-style cluster."""
    data = json.load(open(f"{D}/user_post_comment.json"))
    # map submission_id/comment_id -> author uid, to resolve reply edges to (replier, replied-to) pairs
    id2author = {}
    for uid, rec in data.items():
        for p in rec.get("posts", []) or []:
            id2author[p["submission_id"]] = uid
        for lvl in ("comment_1", "comment_2"):
            for c in rec.get(lvl, []) or []:
                id2author[c["comment_id"]] = uid
    src, dst = [], []
    for uid, rec in data.items():
        if uid not in uid2idx:
            continue
        for lvl in ("comment_1", "comment_2"):
            for c in rec.get(lvl, []) or []:
                parent = c.get("parent_id", "") or c.get("link_id", "")
                parent = parent.split("_", 1)[-1] if "_" in parent else parent
                other = id2author.get(parent)
                if other and other in uid2idx and other != uid:
                    src.append(uid2idx[uid]); dst.append(uid2idx[other])   # replier -> replied-to
    return torch.tensor([src, dst], dtype=torch.long)


def main():
    rows, labels, uid2idx = load_users()
    y = torch.tensor(labels, dtype=torch.long)
    num = torch.tensor([[float(r["submission_num"] or 0), float(r["comment_num"] or 0),
                          float(r["comment_num_1"] or 0), float(r["comment_num_2"] or 0)]
                         for r in rows], dtype=torch.float32)
    num = (num - num.mean(0)) / (num.std(0) + 1e-6)
    desc = [r["description"] or "" for r in rows]
    print(f"[botsim] N={len(rows)} bot_rate={float(y.float().mean()):.3f}  (Reddit domain, "
          f"NOT Twitter -- no followers/friends/retweets)")

    print("[botsim] deriving reply graph from user_post_comment.json (this is slow: ~76MB json)...")
    try:
        ei = build_reply_graph(uid2idx)
        print(f"[botsim] reply edges: {ei.shape[1]}")
    except json.JSONDecodeError as e:
        print(f"[botsim] WARNING: user_post_comment.json is truncated/incomplete ({e}) -- "
              f"skipping reply-graph derivation. Not needed for the 'gated' (graph-less) path anyway; "
              f"only affects a future dedicated reply-graph HGT variant.")
        ei = None

    rng = np.random.default_rng(0); yb = y.numpy(); tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(yb == cls)[0]; rng.shuffle(idx)
        n = len(idx); a, b = int(0.7 * n), int(0.8 * n)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    perm = np.array(tr + va + te); inv = np.argsort(perm)
    torch.save(num[perm], f"{OUT}/num_properties_tensor.pt")
    torch.save(y[perm], f"{OUT}/label.pt")
    torch.save([desc[i] for i in perm], f"{OUT}/description.pkl" if False else f"{OUT}/description_list.pt")
    inv_t = torch.tensor(inv)
    if ei is not None:
        torch.save(inv_t[ei], f"{OUT}/edge_index.pt")
    json.dump({"train_boundary": len(tr), "val_end": len(tr) + len(va), "test_end": len(rows),
               "domain": "reddit", "note": "no followers/friends/retweets; num=[submission_num,"
               "comment_num,comment_num_1,comment_num_2]; graph=derived reply edges (traceable to "
               "submission/comment id, closest analogue of TB20 object edges)"},
              open(f"{OUT}/boundaries.json", "w"))
    print(f"[botsim] split train={len(tr)} val={len(va)} test={len(te)} -> {OUT}")


if __name__ == "__main__":
    main()
