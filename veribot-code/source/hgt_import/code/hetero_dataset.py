import os
import re
import json
from collections import Counter
import torch
from torch_geometric.data import HeteroData

ROOT = "/path/to/workspace/Verifiable_troll_detection"
DATA_DIR = "/path/to/workspace/Twibot20/processed_data"
JSON_DIR = "/path/to/workspace/Twibot20/Twibot-20"

def build_hetero_graph(min_freq=2, mention_freq=10):
    print("=" * 60)
    # ── 1. Load original user features and graph ──────────────────────────────
    print("[1/5] Loading original tensors...")
    des = torch.load(os.path.join(DATA_DIR, "des_tensor.pt"), map_location="cpu").float()
    tweet = torch.load(os.path.join(DATA_DIR, "tweets_tensor.pt"), map_location="cpu").float()
    num_prop = torch.load(os.path.join(DATA_DIR, "num_properties_tensor.pt"), map_location="cpu").float()
    cat_prop = torch.load(os.path.join(DATA_DIR, "cat_properties_tensor.pt"), map_location="cpu").float()
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), map_location="cpu").long()
    edge_type = torch.load(os.path.join(DATA_DIR, "edge_type.pt"), map_location="cpu").long()
    labels = torch.load(os.path.join(DATA_DIR, "label.pt"), map_location="cpu").long()
    uid_to_idx = torch.load(os.path.join(DATA_DIR, "uid_to_idx.pt"), map_location="cpu")
    
    N = des.shape[0]
    E = edge_index.shape[1]
    
    # Compute degree features (consistent with run_evibot.py)
    in_deg = torch.zeros(N)
    out_deg = torch.zeros(N)
    in_deg.scatter_add_(0, edge_index[1], torch.ones(E))
    out_deg.scatter_add_(0, edge_index[0], torch.ones(E))

    log_in_deg = torch.log1p(in_deg)
    log_out_deg = torch.log1p(out_deg)
    # Z-score normalize on train set (first 8278 nodes)
    train_idx = torch.arange(0, 8278)
    mu_i, std_i = log_in_deg[train_idx].mean(), log_in_deg[train_idx].std().clamp(min=1e-8)
    mu_o, std_o = log_out_deg[train_idx].mean(), log_out_deg[train_idx].std().clamp(min=1e-8)
    log_in_deg = (log_in_deg - mu_i) / std_i
    log_out_deg = (log_out_deg - mu_o) / std_o
    deg_feats = torch.stack([log_in_deg, log_out_deg], dim=1)  # [N, 2]
    
    # Concat features: [N, 1546]
    x_user = torch.cat([des, tweet, num_prop, cat_prop, deg_feats], dim=1)
    
    # Create labels mapping (extend to all nodes, use -1 for unlabeled)
    labels_full = torch.full((N,), -1, dtype=torch.long)
    labels_full[:len(labels)] = labels

    # ── 2. Load JSON files and extract entities ───────────────────────────────
    print("[2/5] Extracting URLs, hashtags, tweets, and mentions from JSONs...")
    json_paths = [
        os.path.join(JSON_DIR, "test.json"),
        os.path.join(JSON_DIR, "dev.json"),
        os.path.join(JSON_DIR, "train.json")
    ]
    
    # Map cleaned ID strings to indices
    clean_uid_to_idx = {}
    for uid, idx in uid_to_idx.items():
        clean_uid = uid[1:] if uid.startswith('u') else uid
        clean_uid_to_idx[clean_uid.strip()] = int(idx)
        
    url_regex = re.compile(r'https?://\S+|t\.co/\S+')
    hashtag_regex = re.compile(r'#\w+')
    mention_regex = re.compile(r'@[A-Za-z0-9_]+')
    
    user_hashtags = {}
    user_urls = {}
    user_tweets_raw = {}
    user_mentions = {}
    
    hashtag_counts = Counter()
    url_counts = Counter()
    tweet_counts = Counter()
    mention_counts = Counter()
    
    for path in json_paths:
        if not os.path.exists(path):
            print(f"  WARNING: file {path} not found.")
            continue
        print(f"  Reading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            users_list = json.load(f)
            for u in users_list:
                uid_raw = str(u.get('ID', '')).strip()
                uid_clean = uid_raw[1:] if uid_raw.startswith('u') else uid_raw
                if uid_clean not in clean_uid_to_idx:
                    continue
                user_idx = clean_uid_to_idx[uid_clean]
                tweets = u.get("tweet", []) or []
                
                u_hash = []
                u_url = []
                u_twt = []
                u_men = []
                for t in tweets:
                    if not t:
                        continue
                    t_clean = t.strip()
                    if t_clean:
                        u_twt.append(t_clean)
                        tweet_counts[t_clean] += 1
                        
                    found_urls = url_regex.findall(t)
                    found_hash = hashtag_regex.findall(t)
                    found_mentions = mention_regex.findall(t)
                    
                    for url in found_urls:
                        url_clean = url.strip().strip(".,()[]{}").lower()
                        if url_clean:
                            u_url.append(url_clean)
                            url_counts[url_clean] += 1
                            
                    for h in found_hash:
                        h_clean = h.strip().strip(".,()[]{}").lower()
                        if h_clean:
                            u_hash.append(h_clean)
                            hashtag_counts[h_clean] += 1
                            
                    for m in found_mentions:
                        m_clean = m.strip().strip(".,()[]{}").lower()
                        if m_clean:
                            u_men.append(m_clean)
                            mention_counts[m_clean] += 1
                            
                user_hashtags[user_idx] = u_hash
                user_urls[user_idx] = u_url
                user_tweets_raw[user_idx] = u_twt
                user_mentions[user_idx] = u_men

    # Filter by minimum frequency (keep graph rich, deduplication will handle memory)
    f_urls = {k: v for k, v in url_counts.items() if v >= min_freq}
    f_hashtags = {k: v for k, v in hashtag_counts.items() if v >= min_freq}
    f_tweets = {k: v for k, v in tweet_counts.items() if v >= min_freq}
    f_mentions = {k: v for k, v in mention_counts.items() if v >= 5}
    
    print(f"  Filtered URLs (freq >= {min_freq}): {len(f_urls):,}")
    print(f"  Filtered Hashtags (freq >= {min_freq}): {len(f_hashtags):,}")
    print(f"  Filtered Shared Tweets (freq >= {min_freq}): {len(f_tweets):,}")
    print(f"  Filtered Coordinated Mentions (freq >= 5): {len(f_mentions):,}")

    # Map entities to indices
    url_to_idx = {url: idx for idx, url in enumerate(f_urls.keys())}
    hashtag_to_idx = {tag: idx for idx, tag in enumerate(f_hashtags.keys())}
    tweet_to_idx = {twt: idx for idx, twt in enumerate(f_tweets.keys())}
    mention_to_idx = {men: idx for idx, men in enumerate(f_mentions.keys())}
    
    # Save lookups for verification/explainability
    torch.save(url_to_idx, os.path.join(DATA_DIR, "url_to_idx.pt"))
    torch.save(hashtag_to_idx, os.path.join(DATA_DIR, "hashtag_to_idx.pt"))
    torch.save(tweet_to_idx, os.path.join(DATA_DIR, "tweet_to_idx.pt"))
    torch.save(mention_to_idx, os.path.join(DATA_DIR, "mention_to_idx.pt"))
    print("  Entity index maps saved.")

    # ── 3. Build edges ────────────────────────────────────────────────────────
    print("[3/5] Constructing unique edges (deduplicated)...")
    edge_user_url_set = set()
    edge_user_hash_set = set()
    edge_user_tweet_set = set()
    edge_user_mention_set = set()
    
    for user_idx, urls_list in user_urls.items():
        for url in urls_list:
            if url in url_to_idx:
                edge_user_url_set.add((user_idx, url_to_idx[url]))
                
    for user_idx, tags_list in user_hashtags.items():
        for tag in tags_list:
            if tag in hashtag_to_idx:
                edge_user_hash_set.add((user_idx, hashtag_to_idx[tag]))
                
    for user_idx, twt_list in user_tweets_raw.items():
        for twt in twt_list:
            if twt in tweet_to_idx:
                edge_user_tweet_set.add((user_idx, tweet_to_idx[twt]))
                
    for user_idx, men_list in user_mentions.items():
        for men in men_list:
            if men in mention_to_idx:
                edge_user_mention_set.add((user_idx, mention_to_idx[men]))
                
    # Convert sets to sorted lists for deterministic construction
    edge_user_url_list = list(edge_user_url_set)
    edge_user_hash_list = list(edge_user_hash_set)
    edge_user_tweet_list = list(edge_user_tweet_set)
    edge_user_mention_list = list(edge_user_mention_set)
    
    # Transpose lists into PyTorch edge tensors
    edge_user_url = torch.tensor([[src for src, dst in edge_user_url_list], [dst for src, dst in edge_user_url_list]], dtype=torch.long)
    edge_user_hash = torch.tensor([[src for src, dst in edge_user_hash_list], [dst for src, dst in edge_user_hash_list]], dtype=torch.long)
    edge_user_tweet = torch.tensor([[src for src, dst in edge_user_tweet_list], [dst for src, dst in edge_user_tweet_list]], dtype=torch.long)
    edge_user_mention = torch.tensor([[src for src, dst in edge_user_mention_list], [dst for src, dst in edge_user_mention_list]], dtype=torch.long)

    print(f"  User -> URL unique edges: {edge_user_url.shape[1]:,}")
    print(f"  User -> Hashtag unique edges: {edge_user_hash.shape[1]:,}")
    print(f"  User -> Tweet unique edges: {edge_user_tweet.shape[1]:,}")
    print(f"  User -> Mention unique edges: {edge_user_mention.shape[1]:,}")

    # ── 4. Construct PyG HeteroData ──────────────────────────────────────────
    print("[4/5] Constructing HeteroData object...")
    data = HeteroData()
    
    # Node features
    data['user'].x = x_user
    data['user'].y = labels_full
    data['url'].x = torch.ones(len(f_urls), 1, dtype=torch.float)
    data['hashtag'].x = torch.ones(len(f_hashtags), 1, dtype=torch.float)
    data['tweet'].x = torch.ones(len(f_tweets), 1, dtype=torch.float)
    data['mention'].x = torch.ones(len(f_mentions), 1, dtype=torch.float)
    
    # Edges
    # Social follows — 修复原来把 edge_index.pt 整体当成单一方向 'follows' 关系、
    # 且完全没有反向边的构造错误。edge_type.pt 的语义（核实自 BotRGCN 官方
    # Dataset.py::Build_Graph）：type=0 是从"following"列表构造，存储顺序
    # (i, target) 本身就是 "i follows target"（方向正确，无需翻转）；type=1 是从
    # "follower"列表构造，存储顺序仍是 (i, target)，但语义是 "target follows i"
    # （方向与存储顺序相反，需要 flip 才能还原真实的关注方向）。两者方向修正后
    # 拼接、去重，才是真正完整、方向正确的 "A follows B" 有向社交图；再镜像出
    # rev_follows，使 HGTConv（消息传递按 edge_index 方向单向进行）能像其余四类
    # 实体关系一样双向传播社交图信号，而不是只沿着一个被污染方向的边集合传播。
    follows_type0 = edge_index[:, edge_type == 0]                    # 已经是 "src follows dst"
    follows_type1 = edge_index[:, edge_type == 1].flip(0)            # 翻转还原 "src follows dst"
    follows_all = torch.cat([follows_type0, follows_type1], dim=1)
    follows_all = torch.unique(follows_all, dim=1)                   # 两个来源可能重叠，去重

    data['user', 'follows', 'user'].edge_index = follows_all
    data['user', 'rev_follows', 'user'].edge_index = follows_all.flip(0)
    print(f"  User -> User follows edges (direction-corrected, deduped): {follows_all.shape[1]:,} "
          f"(raw type0={follows_type0.shape[1]:,}, type1={follows_type1.shape[1]:,})")

    # posted_url, used_hashtag, posted_tweet, posted_mention
    data['user', 'posted_url', 'url'].edge_index = edge_user_url
    data['user', 'used_hashtag', 'hashtag'].edge_index = edge_user_hash
    data['user', 'posted_tweet', 'tweet'].edge_index = edge_user_tweet
    data['user', 'posted_mention', 'mention'].edge_index = edge_user_mention
    
    # Reverse relations (crucial for bidirectional messages)
    data['url', 'rev_posted_url', 'user'].edge_index = torch.stack([edge_user_url[1], edge_user_url[0]], dim=0)
    data['hashtag', 'rev_used_hashtag', 'user'].edge_index = torch.stack([edge_user_hash[1], edge_user_hash[0]], dim=0)
    data['tweet', 'rev_posted_tweet', 'user'].edge_index = torch.stack([edge_user_tweet[1], edge_user_tweet[0]], dim=0)
    data['mention', 'rev_posted_mention', 'user'].edge_index = torch.stack([edge_user_mention[1], edge_user_mention[0]], dim=0)

    # ── 5. Save dataset ───────────────────────────────────────────────────────
    print("[5/5] Saving heterogeneous graph...")
    out_path = os.path.join(DATA_DIR, "hetero_data.pt")
    torch.save(data, out_path)
    print(f"Heterogeneous graph saved successfully → {out_path}")
    print(data)
    print("=" * 60)

if __name__ == "__main__":
    build_hetero_graph()
