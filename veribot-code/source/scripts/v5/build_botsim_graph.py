"""Build a REAL entity graph for BotSim-24 from Reddit subreddit membership (Users.csv 'subreddit'
column, a complete field for all 2907 users -- unlike user_post_comment.json which is truncated).
subreddit is structurally identical to TB20/22's hashtag entity: a shared topic tag many users attach
to, extracted from data we actually have in full. This enables the REAL train_hgt (entity-typed) and
train_relnet (social-graph relational conv) paths, not just the graph-less 'gated' fallback.

Outputs (data_dir = /path/to/data_root/data/botsim_graph):
  hetero_data.pt          -- for train_hgt: user/subreddit nodes, user<->subreddit edges
  des_tensor.pt, tweets_tensor.pt, num_properties_tensor.pt, cat_properties_tensor.pt,
  edge_index.pt, edge_type.pt, label.pt, boundaries.json  -- for train_relnet

Honest limitations documented, not hidden:
  * No separate description-only vs tweets-only text exists (raw post/comment JSON is truncated
    everywhere we've checked) -- both RelNet text branches receive the SAME RoBERTa embedding of
    the user's profile description (from the complete SAHG features bundle).
  * num_properties: 17-dim engineered features from /path/to/SAHG/data/botsim_24/features.csv (richer
    than the 4-dim [submission_num,...] used in the graph-less gated-only run).
  * RelNet's social-graph relational conv needs a flat user-user edge_index -- derived here as a
    co-subreddit PROJECTION (edge if two users share >=1 subreddit), single relation.
"""
import ast, csv, json, os
import numpy as np, torch
from torch_geometric.data import HeteroData

USERS_CSV = "/path/to/data_root/data/botsim/Users.csv"
SAHG_DIR = "/path/to/SAHG/data/botsim_24"
OUT = "/path/to/data_root/data/botsim_graph"
os.makedirs(OUT, exist_ok=True)


def parse_sr(s):
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def main():
    rows = list(csv.DictReader(open(USERS_CSV)))
    N = len(rows)
    labels = [0] * 1907 + [1] * 1000
    assert N == len(labels) == 2907
    y = torch.tensor(labels, dtype=torch.long)

    sr_lists = [parse_sr(r.get("subreddit", "")) for r in rows]
    all_subs = sorted({s for lst in sr_lists for s in lst})
    sub2idx = {s: i for i, s in enumerate(all_subs)}
    print(f"[botsim-graph] N={N} unique_subreddits={len(all_subs)}")

    u_src, sr_dst = [], []
    for u, lst in enumerate(sr_lists):
        for s in lst:
            u_src.append(u); sr_dst.append(sub2idx[s])
    user_sub_ei = torch.tensor([u_src, sr_dst], dtype=torch.long)
    print(f"[botsim-graph] user->subreddit edges: {user_sub_ei.shape[1]}")

    # SAHG's richer 17-dim engineered features (submission_num, comment_num, has_description,
    # description_len, n_subreddits, n_posts, n_comments_received, ..., is_active)
    sahg_feat_rows = list(csv.DictReader(open(f"{SAHG_DIR}/features.csv")))
    sahg_labels = [int(r["label"]) for r in csv.DictReader(open(f"{SAHG_DIR}/labels.csv"))]
    assert len(sahg_feat_rows) == N, f"SAHG bundle has {len(sahg_feat_rows)} rows, expected {N}"
    assert sahg_labels == labels, "SAHG label order doesn't match Users.csv order -- cannot safely reuse"
    feat_cols = list(sahg_feat_rows[0].keys())
    num = torch.tensor([[float(r[c]) for c in feat_cols] for r in sahg_feat_rows], dtype=torch.float32)

    text = torch.load(f"{SAHG_DIR}/roberta_features.pt", map_location="cpu", weights_only=False).float()
    assert text.shape[0] == N

    rng = np.random.default_rng(0); yb = y.numpy(); tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(yb == cls)[0]; rng.shuffle(idx)
        n = len(idx); a, b = int(0.7 * n), int(0.8 * n)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    perm = np.array(tr + va + te)
    inv = np.argsort(perm)
    inv_t = torch.tensor(inv, dtype=torch.long)

    num_p = num[perm]
    mu, std = num_p[:len(tr)].mean(0), num_p[:len(tr)].std(0).clamp(min=1e-6)
    num_p = (num_p - mu) / std
    text_p = text[perm]
    y_p = y[perm]

    boundaries = {"train_boundary": len(tr), "val_end": len(tr) + len(va), "test_end": N,
                  "note": f"BotSim-24 REAL entity graph (subreddit membership, {len(all_subs)} subreddits, "
                          f"{user_sub_ei.shape[1]} user-subreddit edges) + co-subreddit user-user projection "
                          f"for RelNet. num={len(feat_cols)}-d (SAHG engineered features). des/tweets text "
                          f"branches share the SAME profile-description RoBERTa embedding (no separate "
                          f"post/comment text available -- source JSON truncated)."}
    json.dump(boundaries, open(f"{OUT}/boundaries.json", "w"))

    # ---- RelNet tensors ----
    torch.save(text_p, f"{OUT}/des_tensor.pt")
    torch.save(text_p, f"{OUT}/tweets_tensor.pt")
    torch.save(num_p, f"{OUT}/num_properties_tensor.pt")
    torch.save(torch.zeros(N, 1), f"{OUT}/cat_properties_tensor.pt")  # no real categorical block available
    torch.save(y_p, f"{OUT}/label.pt")

    # co-subreddit user-user projection (single relation) for RelNet's relational conv
    from collections import defaultdict
    sub_users = defaultdict(list)
    for u, lst in enumerate(sr_lists):
        for s in lst:
            sub_users[s].append(u)
    edge_set = set()
    for s, users in sub_users.items():
        if len(users) > 500:   # skip mega-subreddits (e.g. 'worldnews') -- O(n^2) blowup, low info edges
            continue
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                a, b = users[i], users[j]
                edge_set.add((a, b)); edge_set.add((b, a))
    src = [inv[a] for a, b in edge_set]; dst = [inv[b] for a, b in edge_set]
    proj_ei = torch.tensor([src, dst], dtype=torch.long)
    proj_et = torch.zeros(proj_ei.shape[1], dtype=torch.long)
    print(f"[botsim-graph] co-subreddit user-user projection edges: {proj_ei.shape[1]}")
    torch.save(proj_ei, f"{OUT}/edge_index.pt")
    torch.save(proj_et, f"{OUT}/edge_type.pt")

    # ---- HGT hetero_data.pt ----
    hd = HeteroData()
    user_x = torch.cat([text_p, text_p, num_p, torch.zeros(N, 1)], dim=-1)  # des+tweet+num+cat, mirrors TB20 layout
    hd["user"].x = user_x
    hd["user"].y = y_p
    hd["subreddit"].x = torch.ones(len(all_subs), 1)
    user_sub_ei_perm = torch.stack([inv_t[user_sub_ei[0]], user_sub_ei[1]])  # remap USER idx only, subreddit idx unchanged
    hd["user", "posted_in", "subreddit"].edge_index = user_sub_ei_perm
    hd["subreddit", "rev_posted_in", "user"].edge_index = user_sub_ei_perm.flip(0)
    hd["user", "cosubreddit", "user"].edge_index = proj_ei
    torch.save(hd, f"{OUT}/hetero_data.pt")
    print(f"[botsim-graph] saved -> {OUT}  node_types={hd.node_types}")


if __name__ == "__main__":
    main()
