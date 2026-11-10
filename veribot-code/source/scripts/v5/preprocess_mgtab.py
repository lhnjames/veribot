"""MGTAB preprocessing for a from-scratch retrain (no raw text -> SemVerifier/T2 content-object evidence
are NOT applicable; this dataset only supports Layer-1 HGT-style retrain + T3 behavioral(feature) +
T1-lite coordinated cluster on the co-hashtag/co-URL relation types + T4 embedding kNN).

MGTAB provides (already downloaded to /path/to/data_root/data/mgtab/):
  features.pt      [10199, 788]  float32   precomputed user features (opaque: numeric+LaBSE tweet emb concat)
  edge_index.pt     [2, 1700108] int64     multi-relational USER-USER graph
  edge_type.pt      [1700108]    int64     7 relation types (0..6): follower/friend/mention/reply/quoted/url/hashtag
  edge_weight.pt    [1700108]    float32
  labels_bot.pt     [10199]      int64     0=human(7451) 1=bot(2748)  (~27% bot rate)
  labels_stance.pt  [10199]      int64     (unused for bot detection)

This script only builds the SPLIT + a homogeneous-graph HGT-compatible tensor bundle (single node type
"user", 7 relation types as separate edge_index_dict entries) -- NOT a full hetero object graph (MGTAB
doesn't expose hashtag/url as distinct nodes, only as user-user edge types), so:
  - T1 (coordinated cluster) degrades to "cluster via relation-type 5/6 (url/hashtag) user-user edges"
    -- interpretable as co-sharing but not traceable to WHICH object.
  - T2 (strong single object) is NOT APPLICABLE (no object identity).
  - T3 (behavioral) IS applicable if we treat contiguous slices of the 788-d feature as "features"
    (MGTAB doesn't name them, so this tier degrades to unnamed feature-bin Wilson, less interpretable
    than TwiBot-20's 33 NAMED features).
"""
import json, os
import numpy as np, torch

D = "/path/to/data_root/data/mgtab"
OUT = "/path/to/data_root/data/mgtab_split"
os.makedirs(OUT, exist_ok=True)
REL_NAMES = ["follower", "friend", "mention", "reply", "quoted", "url", "hashtag"]  # per MGTAB README order


def main():
    feat = torch.load(f"{D}/features.pt", weights_only=False)
    y = torch.load(f"{D}/labels_bot.pt", weights_only=False)
    ei = torch.load(f"{D}/edge_index.pt", weights_only=False)
    et = torch.load(f"{D}/edge_type.pt", weights_only=False)
    ew = torch.load(f"{D}/edge_weight.pt", weights_only=False)
    N = feat.shape[0]
    print(f"[mgtab] N={N} feat_dim={feat.shape[1]} bot_rate={float(y.float().mean()):.3f} "
          f"relations={dict(zip(*[x.tolist() for x in torch.unique(et, return_counts=True)]))}")

    # stratified 70/10/20 split (MGTAB ships no official split for the base 10199-user graph)
    rng = np.random.default_rng(0)
    yb = y.numpy(); tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(yb == cls)[0]; rng.shuffle(idx)
        n = len(idx); a, b = int(0.7 * n), int(0.8 * n)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    perm = np.array(tr + va + te); inv = np.argsort(perm)
    torch.save(torch.tensor(perm), f"{OUT}/perm_new_to_old.pt")
    torch.save(feat[perm], f"{OUT}/features.pt")
    torch.save(y[perm], f"{OUT}/label.pt")
    # remap edges to new indices; split into per-relation edge_index_dict entries (HGT-style, but
    # single node type "user" repeated 7 times as (user, rel_i, user))
    inv_t = torch.tensor(inv)
    edge_index_dict = {}
    for r in range(7):
        m = et == r
        e = inv_t[ei[:, m]]
        edge_index_dict[("user", REL_NAMES[r], "user")] = e
    torch.save(edge_index_dict, f"{OUT}/edge_index_dict.pt")
    torch.save(ew, f"{OUT}/edge_weight.pt")  # NOTE: not reordered by relation-split; keep original order if needed per-edge
    json.dump({"train_boundary": len(tr), "val_end": len(tr) + len(va), "test_end": N,
               "relations": REL_NAMES, "note": "single homogeneous user node type; no object nodes"},
              open(f"{OUT}/boundaries.json", "w"))
    print(f"[mgtab] split train={len(tr)} val={len(va)} test={len(te)} -> {OUT}")
    print("[mgtab] LIMITATIONS: no raw text (SemVerifier/T2 N/A); T1 degrades to relation-type "
          "co-occurrence cluster (url/hashtag relations, no object identity); T3 features are unnamed.")


if __name__ == "__main__":
    main()
