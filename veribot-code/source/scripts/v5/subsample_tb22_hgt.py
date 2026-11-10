"""HGT-specific subsample of TwiBot-22: full-graph HGTConv attention OOM'd at 700k users even after
raising min_freq (hub hashtags shared by hundreds of thousands of accounts create huge fan-out edges
that frequency-threshold filtering cannot shrink -- the object is ALREADY well above any reasonable
threshold). RelNet trained fine on the full 700k (its social graph is far sparser, ~2M edges total),
so this subsample affects ONLY the HGT track, not RelNet.

Stratified subsample of M users (same train/val/test PROPORTIONS as the full split), sliced from the
already-computed full tensors (no re-encoding), entity graph rebuilt from scratch for just this subset.
"""
import json, pickle, re
import numpy as np, torch
from collections import Counter

FULL = "/path/to/data_root/data/twibot22_full"
D = "/path/to/data_root/data/twibot22_train"
OUT = "/path/to/data_root/data/twibot22_hgt_sub"
M_TRAIN, M_VAL, M_TEST = 35000, 5000, 10000   # ~7% of full 700k, preserving split ratios (0.7/0.1/0.2 approx)
URL_RE = re.compile(r'https?://\S+|t\.co/\S+')
HASHTAG_RE = re.compile(r'#\w+')
MENTION_RE = re.compile(r'@[A-Za-z0-9_]+')

import os
os.makedirs(OUT, exist_ok=True)

bnd = json.load(open(f"{FULL}/boundaries.json"))
tb, ve, te = bnd["train_boundary"], bnd["val_end"], bnd["test_end"]
rng = np.random.default_rng(0)
sub_train = rng.choice(tb, size=M_TRAIN, replace=False)
sub_val = rng.choice(np.arange(tb, ve), size=M_VAL, replace=False)
sub_test = rng.choice(np.arange(ve, te), size=M_TEST, replace=False)
sub_idx = np.concatenate([np.sort(sub_train), np.sort(sub_val), np.sort(sub_test)])  # new-space indices (post stage_features permutation)
print(f"[hgt_sub] sampled {len(sub_idx)} users: train={M_TRAIN} val={M_VAL} test={M_TEST}", flush=True)

# slice already-computed tensors (in "new" index space from stage_features)
for fn in ("des_tensor.pt", "tweets_tensor.pt", "num_properties_tensor.pt", "cat_properties_tensor.pt", "label.pt"):
    t = torch.load(f"{FULL}/{fn}", weights_only=False)
    torch.save(t[sub_idx], f"{OUT}/{fn}")
    print(f"[hgt_sub] sliced {fn} -> {tuple(t[sub_idx].shape)}", flush=True)

json.dump({"train_boundary": M_TRAIN, "val_end": M_TRAIN + M_VAL, "test_end": M_TRAIN + M_VAL + M_TEST},
          open(f"{OUT}/boundaries.json", "w"))

new_of_old = {int(old): new for new, old in enumerate(sub_idx)}  # old "new-space" idx -> subsample idx

# follows edges: keep only edges where BOTH endpoints are in the subsample
ei = torch.load(f"{FULL}/edge_index.pt", weights_only=False).numpy()
sub_set = set(sub_idx.tolist())
mask = np.isin(ei[0], sub_idx) & np.isin(ei[1], sub_idx)
src, dst = ei[0][mask], ei[1][mask]
src2 = np.array([new_of_old[s] for s in src]); dst2 = np.array([new_of_old[d] for d in dst])
torch.save(torch.tensor(np.stack([src2, dst2]), dtype=torch.long), f"{OUT}/edge_index.pt")
torch.save(torch.zeros(len(src2), dtype=torch.long), f"{OUT}/edge_type.pt")
print(f"[hgt_sub] follows edges kept: {len(src2)}/{ei.shape[1]}", flush=True)

# entity graph: rescan tweets for JUST the subsampled users (map new-space idx -> old(raw) idx -> tweets)
perm = torch.load(f"{FULL}/perm_new_to_old.pt", weights_only=False).numpy()   # new-space -> raw twibot22_train idx
old_of_new = {int(n): int(perm[n]) for n in sub_idx}
tw = pickle.load(open(f"{D}/user_tweets.pkl", "rb"))
u2i = torch.load(f"{D}/uid_to_idx.pt", weights_only=False)

user_hash, user_url, user_men, user_twt = {}, {}, {}, {}
hc, uc, tc, mc = Counter(), Counter(), Counter(), Counter()
for new_i in sub_idx.tolist():
    raw_i = old_of_new[new_i]
    tweets = tw.get(raw_i, [])
    if isinstance(tweets, str): tweets = [tweets]
    uh, uu, um, ut = [], [], [], []
    for t in (tweets or []):
        if not isinstance(t, str) or not t.strip(): continue
        t_clean = t.strip(); ut.append(t_clean); tc[t_clean] += 1
        for h in URL_RE.findall(t):
            h = h.strip().strip(".,()[]{}").lower()
            if h: uu.append(h); uc[h] += 1
        for h in HASHTAG_RE.findall(t):
            h = h.strip().strip(".,()[]{}").lower()
            if h: uh.append(h); hc[h] += 1
        for h in MENTION_RE.findall(t):
            h = h.strip().strip(".,()[]{}").lower()
            if h: um.append(h); mc[h] += 1
    user_hash[new_of_old[new_i]] = uh; user_url[new_of_old[new_i]] = uu
    user_men[new_of_old[new_i]] = um; user_twt[new_of_old[new_i]] = ut

min_freq, mention_freq = 2, 5   # subsample is small now, can go back to TB20-faithful thresholds
f_url = {k for k, v in uc.items() if v >= min_freq}
f_hash = {k for k, v in hc.items() if v >= min_freq}
f_twt = {k for k, v in tc.items() if v >= min_freq}
f_men = {k for k, v in mc.items() if v >= mention_freq}
url_to_idx = {k: i for i, k in enumerate(f_url)}
hash_to_idx = {k: i for i, k in enumerate(f_hash)}
twt_to_idx = {k: i for i, k in enumerate(f_twt)}
men_to_idx = {k: i for i, k in enumerate(f_men)}
print(f"[hgt_sub] filtered: url={len(f_url)} hashtag={len(f_hash)} tweet={len(f_twt)} mention={len(f_men)}", flush=True)


def build_edges(user_map, obj_to_idx):
    pairs = set()
    for u, items in user_map.items():
        for it in items:
            if it in obj_to_idx:
                pairs.add((u, obj_to_idx[it]))
    pairs = list(pairs)
    return torch.tensor([[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.long) if pairs else torch.zeros(2, 0, dtype=torch.long)


e_url = build_edges(user_url, url_to_idx)
e_hash = build_edges(user_hash, hash_to_idx)
e_twt = build_edges(user_twt, twt_to_idx)
e_men = build_edges(user_men, men_to_idx)
print(f"[hgt_sub] edges: url={e_url.shape[1]} hashtag={e_hash.shape[1]} tweet={e_twt.shape[1]} mention={e_men.shape[1]}", flush=True)

# degree features from the subsampled follows graph
N = len(sub_idx)
train_idx = torch.arange(0, M_TRAIN)
in_deg = torch.zeros(N); out_deg = torch.zeros(N)
src_t, dst_t = torch.load(f"{OUT}/edge_index.pt", weights_only=False)
E = src_t.shape[0]
in_deg.scatter_add_(0, dst_t, torch.ones(E)); out_deg.scatter_add_(0, src_t, torch.ones(E))
log_in, log_out = torch.log1p(in_deg), torch.log1p(out_deg)
mu_i, sd_i = log_in[train_idx].mean(), log_in[train_idx].std().clamp(min=1e-8)
mu_o, sd_o = log_out[train_idx].mean(), log_out[train_idx].std().clamp(min=1e-8)
deg_feats = torch.stack([(log_in - mu_i) / sd_i, (log_out - mu_o) / sd_o], dim=1)

des = torch.load(f"{OUT}/des_tensor.pt", weights_only=False)
tweet_emb = torch.load(f"{OUT}/tweets_tensor.pt", weights_only=False)
num = torch.load(f"{OUT}/num_properties_tensor.pt", weights_only=False)
cat = torch.load(f"{OUT}/cat_properties_tensor.pt", weights_only=False)
label = torch.load(f"{OUT}/label.pt", weights_only=False)
x_user = torch.cat([des, tweet_emb, num, cat, deg_feats], dim=1)
print(f"[hgt_sub] user.x assembled: {tuple(x_user.shape)}", flush=True)

from torch_geometric.data import HeteroData
hd = HeteroData()
hd["user"].x = x_user
hd["user"].y = label
hd["url"].x = torch.ones(max(len(f_url), 1), 1)
hd["hashtag"].x = torch.ones(max(len(f_hash), 1), 1)
hd["tweet"].x = torch.ones(max(len(f_twt), 1), 1)
hd["mention"].x = torch.ones(max(len(f_men), 1), 1)
ei_full = torch.stack([src_t, dst_t])
hd["user", "follows", "user"].edge_index = ei_full
hd["user", "rev_follows", "user"].edge_index = ei_full.flip(0)
hd["user", "posted_url", "url"].edge_index = e_url
hd["user", "used_hashtag", "hashtag"].edge_index = e_hash
hd["user", "posted_tweet", "tweet"].edge_index = e_twt
hd["user", "posted_mention", "mention"].edge_index = e_men
hd["url", "rev_posted_url", "user"].edge_index = e_url.flip(0)
hd["hashtag", "rev_used_hashtag", "user"].edge_index = e_hash.flip(0)
hd["tweet", "rev_posted_tweet", "user"].edge_index = e_twt.flip(0)
hd["mention", "rev_posted_mention", "user"].edge_index = e_men.flip(0)
torch.save(hd, f"{OUT}/hetero_data.pt")
torch.save(url_to_idx, f"{OUT}/url_to_idx.pt")
torch.save(hash_to_idx, f"{OUT}/hashtag_to_idx.pt")
torch.save(twt_to_idx, f"{OUT}/tweet_to_idx.pt")
torch.save(men_to_idx, f"{OUT}/mention_to_idx.pt")
print(f"[hgt_sub] saved hetero_data.pt -> {OUT}", flush=True)
print("done.", flush=True)
