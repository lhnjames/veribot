"""Option-B preprocessing: build TwiBot-22 into the EXACT TwiBot-20 tensor/graph layout, replicating
verifiable_detection/hetero_dataset.py::build_hetero_graph node-for-node, edge-for-edge, so the SAME
training scripts (verifiable_detection/train_hgt_unified.py, scratch/train_evinet_normalized.py) can
be pointed at TB22's own train/val/test split with zero method changes.

Run stages IN ORDER (each resumable):
  python preprocess_tb22_full.py --stage features   # fast (CPU): num/cat tensors + stratified split
  python preprocess_tb22_full.py --stage edges       # fast (CPU): remap the single-relation social graph
  python preprocess_tb22_full.py --stage text        # slow (GPU): RoBERTa-base encode des + tweets
  python preprocess_tb22_full.py --stage graph        # fast (CPU, run LAST): entity graph + assemble
                                                       # final hetero_data.pt (needs des/tweets from `text`)

Fidelity notes (documented, not hidden):
  * Entity extraction mirrors hetero_dataset.py EXACTLY: url/hashtag regex, mention regex, min_freq=2 for
    url/hashtag/tweet, min_freq=5 for mention. The 'tweet' object node is keyed by EXACT tweet text (not
    per-user) so it captures co-retweet/copy-paste structure across accounts, same semantics as TB20.
  * Object node x = ones(N,1) (TB20 uses a constant placeholder, real signal comes from entity-prior
    features attached at training time by evidence_features.build_entity_prior_features).
  * user.x = cat[des(768), tweet(768), num_prop(5), cat_prop(3), deg_feats(2)] = 1546-d, deg_feats =
    z-normalized log1p(in/out-degree) of the FOLLOWS graph, normalized on TB22's own train indices.
  * TB22 social edge_type is a SINGLE relation (no follower/following distinction like TB20's type0/type1),
    so we use edge_index.pt as-is for 'follows' (no flip-and-merge needed) -- documented degradation vs
    TB20's two-typed social graph, not a bug.
  * Text encoder: TwiBot-20's des/tweet embeddings were precomputed upstream with RoBERTa-base (768-d,
    matching RelNet's des_size=768/tweet_size=768). We use roberta-base directly for internal consistency
    (a retrain only needs INTERNAL consistency, not byte-identical vectors to TwiBot-20's own encoder run).
"""
import argparse, json, os, pickle, re
import numpy as np, torch

D = "/path/to/data_root/data/twibot22_train"
OUT = "/path/to/data_root/data/twibot22_full"
ENCODER = "/path/to/data_root/cache/modelscope/models/AI-ModelScope--roberta-base/snapshots/master"  # local snapshot (HF blocked from js02, downloaded via modelscope mirror)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
URL_RE = re.compile(r'https?://\S+|t\.co/\S+')
HASHTAG_RE = re.compile(r'#\w+')
MENTION_RE = re.compile(r'@[A-Za-z0-9_]+')


def _load_common():
    num5 = np.load(f"{D}/num_properties_raw.npy")
    cat3 = torch.load(f"{D}/cat_properties_tensor.pt", weights_only=False)
    label = torch.load(f"{D}/label.pt", weights_only=False)
    u2i = torch.load(f"{D}/uid_to_idx.pt", weights_only=False)
    return num5, cat3, label, u2i


def stage_features():
    num5, cat3, label, u2i = _load_common()
    N = len(label); y = label.numpy()
    rng = np.random.default_rng(0)
    order = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; rng.shuffle(idx); order.append(idx)
    tr, va, te = [], [], []
    for idx in order:
        n = len(idx); a, b = int(0.7 * n), int(0.8 * n)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    perm = np.array(tr + va + te)
    inv = np.argsort(perm)
    torch.save(torch.tensor(perm), f"{OUT}/perm_new_to_old.pt")
    torch.save(torch.tensor(inv), f"{OUT}/inv_old_to_new.pt")
    num = torch.tensor(num5[perm], dtype=torch.float32)
    num = (num - num[:len(tr)].mean(0)) / (num[:len(tr)].std(0) + 1e-6)   # normalize on TRAIN only
    torch.save(num, f"{OUT}/num_properties_tensor.pt")
    torch.save(cat3[perm].float(), f"{OUT}/cat_properties_tensor.pt")
    torch.save(label[perm], f"{OUT}/label.pt")
    json.dump({"train_boundary": len(tr), "val_end": len(tr) + len(va), "test_end": N},
              open(f"{OUT}/boundaries.json", "w"))
    print(f"[features] N={N} train={len(tr)} val={len(va)} test={len(te)} -> {OUT}")


def stage_edges():
    inv = torch.load(f"{OUT}/inv_old_to_new.pt", weights_only=False)
    ei = torch.load(f"{D}/edge_index.pt", weights_only=False)
    et = torch.load(f"{D}/edge_type.pt", weights_only=False)
    assert torch.unique(et).tolist() == [0], "expected single-relation social graph"
    torch.save(inv[ei], f"{OUT}/edge_index.pt")
    torch.save(et, f"{OUT}/edge_type.pt")
    print(f"[edges] remapped social edges: {ei.shape[1]} (single relation -> used as-is for 'follows')")


def stage_text():
    from transformers import AutoTokenizer, AutoModel
    perm = torch.load(f"{OUT}/perm_new_to_old.pt", weights_only=False).numpy()
    desc = pickle.load(open(f"{D}/description.pkl", "rb"))
    tw = pickle.load(open(f"{D}/user_tweets.pkl", "rb"))
    tok = AutoTokenizer.from_pretrained(ENCODER); model = AutoModel.from_pretrained(ENCODER).to(DEV).eval()

    def _encode(texts, bs=256, max_len=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEV)
            with torch.no_grad():
                h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(vec.cpu())
            if i % 25600 == 0: print(f"    encoded {i}/{len(texts)}", flush=True)
        return torch.cat(out).float()

    desc_texts = [str(desc[o]) if o < len(desc) else "" for o in perm]
    torch.save(_encode(desc_texts), f"{OUT}/des_tensor.pt"); print("[text] des_tensor done", flush=True)

    def cat_tweets(o):
        t = tw.get(o, []) if isinstance(tw, dict) else (tw[o] if o < len(tw) else [])
        if isinstance(t, str): t = [t]
        return " ".join(str(x) for x in (t or [])[:50])
    tweet_texts = [cat_tweets(o) for o in perm]
    torch.save(_encode(tweet_texts), f"{OUT}/tweets_tensor.pt"); print("[text] tweets_tensor done", flush=True)


def stage_graph(min_freq=20, mention_freq=50):
    # TB22 has 700k users (60x TB20's 11.8k) -> TB20's min_freq=2/mention_freq=5 produced edge counts
    # (mention=28.3M, hashtag=15.0M) that OOM'd full-graph HGTConv attention (54GB for one relation
    # alone) on an 80GB GPU. Raised thresholds to keep the graph within full-graph training's memory
    # budget; this is a scale-driven necessity, not a change to entity extraction *semantics* (same
    # regexes, same co-occurrence logic, same object identity keying).
    """Run LAST: entity graph (mirrors hetero_dataset.py) + follows edges + final user.x assembly."""
    from torch_geometric.data import HeteroData
    from collections import Counter
    inv = torch.load(f"{OUT}/inv_old_to_new.pt", weights_only=False).numpy()
    tw = pickle.load(open(f"{D}/user_tweets.pkl", "rb"))
    num = torch.load(f"{OUT}/num_properties_tensor.pt", weights_only=False)
    cat = torch.load(f"{OUT}/cat_properties_tensor.pt", weights_only=False)
    des = torch.load(f"{OUT}/des_tensor.pt", weights_only=False)
    tweet_emb = torch.load(f"{OUT}/tweets_tensor.pt", weights_only=False)
    label = torch.load(f"{OUT}/label.pt", weights_only=False)
    N = num.shape[0]
    bnd = json.load(open(f"{OUT}/boundaries.json"))
    train_idx = torch.arange(0, bnd["train_boundary"])

    print("[graph] scanning tweets for url/hashtag/mention + tweet-text co-occurrence...", flush=True)
    user_hash, user_url, user_men, user_twt = {}, {}, {}, {}
    hc, uc, tc, mc = Counter(), Counter(), Counter(), Counter()
    for old_i, tweets in (tw.items() if isinstance(tw, dict) else enumerate(tw)):
        if old_i >= len(inv): continue
        u = int(inv[old_i])
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
        user_hash[u] = uh; user_url[u] = uu; user_men[u] = um; user_twt[u] = ut

    f_url = {k for k, v in uc.items() if v >= min_freq}
    f_hash = {k for k, v in hc.items() if v >= min_freq}
    f_twt = {k for k, v in tc.items() if v >= min_freq}
    f_men = {k for k, v in mc.items() if v >= mention_freq}
    url_to_idx = {k: i for i, k in enumerate(f_url)}
    hash_to_idx = {k: i for i, k in enumerate(f_hash)}
    twt_to_idx = {k: i for i, k in enumerate(f_twt)}
    men_to_idx = {k: i for i, k in enumerate(f_men)}
    print(f"[graph] filtered: url={len(f_url)} hashtag={len(f_hash)} tweet(shared)={len(f_twt)} mention={len(f_men)}", flush=True)

    def build_edges(user_map, obj_to_idx):
        s, d = set(), set()
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
    print(f"[graph] edges: url={e_url.shape[1]} hashtag={e_hash.shape[1]} tweet={e_twt.shape[1]} mention={e_men.shape[1]}", flush=True)

    # follows graph (single relation, used as-is) -> degree features
    ei = torch.load(f"{OUT}/edge_index.pt", weights_only=False)
    E = ei.shape[1]
    in_deg = torch.zeros(N); out_deg = torch.zeros(N)
    in_deg.scatter_add_(0, ei[1], torch.ones(E)); out_deg.scatter_add_(0, ei[0], torch.ones(E))
    log_in, log_out = torch.log1p(in_deg), torch.log1p(out_deg)
    mu_i, sd_i = log_in[train_idx].mean(), log_in[train_idx].std().clamp(min=1e-8)
    mu_o, sd_o = log_out[train_idx].mean(), log_out[train_idx].std().clamp(min=1e-8)
    deg_feats = torch.stack([(log_in - mu_i) / sd_i, (log_out - mu_o) / sd_o], dim=1)

    x_user = torch.cat([des, tweet_emb, num, cat, deg_feats], dim=1)
    print(f"[graph] user.x assembled: {tuple(x_user.shape)} (expect [N,1546])", flush=True)

    hd = HeteroData()
    hd["user"].x = x_user
    hd["user"].y = label
    hd["url"].x = torch.ones(max(len(f_url), 1), 1)
    hd["hashtag"].x = torch.ones(max(len(f_hash), 1), 1)
    hd["tweet"].x = torch.ones(max(len(f_twt), 1), 1)
    hd["mention"].x = torch.ones(max(len(f_men), 1), 1)
    hd["user", "follows", "user"].edge_index = ei
    hd["user", "rev_follows", "user"].edge_index = ei.flip(0)
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
    u2i = torch.load(f"{D}/uid_to_idx.pt", weights_only=False)
    torch.save({uid: int(inv[old]) for uid, old in u2i.items()}, f"{OUT}/uid_to_idx.pt")
    print(f"[graph] saved hetero_data.pt -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["features", "edges", "text", "graph", "all"])
    a = ap.parse_args()
    if a.stage in ("features", "all"): stage_features()
    if a.stage in ("edges", "all"): stage_edges()
    if a.stage in ("text", "all"): stage_text()
    if a.stage in ("graph", "all"): stage_graph()
    print("done.")
