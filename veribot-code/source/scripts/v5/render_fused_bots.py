"""Render the 681 fused-classified TB22 accounts into TB20 unit format for SemVerifier citation."""
import json, math, pickle, sys
import numpy as np, torch
sys.path.insert(0, "/path/to/project_root")
from verifiable_bot.v4 import render_v4, thresholds_v4
from verifiable_bot.v4.features_v4 import to_float, to_bool, clean_str, tweet_list, N_FEATURES

D = "/path/to/data_root/data/twibot22_train"
TB20_TRAIN = "/path/to/data_root/data/twibot20_complete/train.json"
SUB = "/path/to/data_root/data/twibot22_hgt_sub"

def extract_features_tb22(record):
    p = record.get("profile") or {}
    followers = to_float(p.get("followers_count")); friends = to_float(p.get("friends_count"))
    statuses = to_float(p.get("statuses_count")); favourites = to_float(p.get("favourites_count"))
    listed = to_float(p.get("listed_count"))
    age = float(record.get("_age_days") or 1.0); age = max(age, 1.0)
    name = clean_str(p.get("name")) or ""; screen_name = clean_str(p.get("screen_name")) or ""
    description = clean_str(p.get("description")) or ""
    sn_digits = sum(c.isdigit() for c in screen_name)
    tweets = tweet_list(record); n = len(tweets)
    rts = sum(1 for t in tweets if isinstance(t, str) and t.startswith("RT @"))
    lengths = [len(t) for t in tweets if isinstance(t, str)]
    distinct = len(set(tweets)) / max(n, 1)
    urls = sum(1 for t in tweets if isinstance(t, str) and "http" in t)
    hashtags = sum(t.count("#") for t in tweets if isinstance(t, str))
    mentions = sum(t.count("@") for t in tweets if isinstance(t, str))
    v = [math.log1p(followers), math.log1p(friends), math.log1p(statuses), math.log1p(favourites),
         math.log1p(listed), friends/max(followers,1), followers/max(friends,1), statuses/age,
         favourites/age, followers/age, friends/age, math.log1p(age), float(len(name)),
         float(len(screen_name)), float(sn_digits), sn_digits/max(len(screen_name),1),
         float(len(description)), 1.0 if description else 0.0, to_bool(p.get("verified")),
         to_bool(p.get("default_profile")), to_bool(p.get("default_profile_image")),
         to_bool(p.get("geo_enabled")), to_bool(p.get("protected")),
         1.0 if clean_str(p.get("url")) else 0.0, 1.0 if clean_str(p.get("location")) else 0.0,
         float(n), rts/max(n,1), float(np.mean(lengths)) if lengths else 0.0,
         float(np.std(lengths)) if lengths else 0.0, distinct, urls/max(n,1),
         hashtags/max(n,1), mentions/max(n,1)]
    assert len(v) == N_FEATURES
    return [0.0 if not math.isfinite(x) else float(x) for x in v]

render_v4.extract_features = extract_features_tb22
print("fitting TB20 thresholds...", flush=True)
TH = thresholds_v4.fit_thresholds(TB20_TRAIN)

bots = json.load(open(f"{SUB}/fused_test_classified_bots.json"))
target_uids = {"u" + b["account_id"] for b in bots}
u2i = torch.load(f"{D}/uid_to_idx.pt", weights_only=False)
target_raw_idx = {u2i[u]: u for u in target_uids if u in u2i}
print(f"targets: {len(bots)} bots, {len(target_raw_idx)} resolved to raw idx", flush=True)

num5 = np.load(f"{D}/num_properties_raw.npy")
cat3 = torch.load(f"{D}/cat_properties_tensor.pt", weights_only=False).numpy()
desc = pickle.load(open(f"{D}/description.pkl", "rb"))
label = torch.load(f"{D}/label.pt", weights_only=False).numpy()
print("loading tweets (8GB)...", flush=True)
tw = pickle.load(open(f"{D}/user_tweets.pkl", "rb"))

out = open("/path/to/project_root/outputs/v5/tb22_fused_bots_render.jsonl", "w")
n_ok = 0
for i, uid in target_raw_idx.items():
    prof = {"followers_count": float(num5[i,0]), "friends_count": float(num5[i,1]),
            "statuses_count": float(num5[i,2]), "favourites_count": float(num5[i,3]),
            "description": desc[i] if i < len(desc) else "",
            "protected": bool(cat3[i,0]), "verified": bool(cat3[i,1]), "default_profile": bool(cat3[i,2])}
    tweets = tw.get(i) or tw.get(uid) or tw.get(uid.lstrip("u")) or []
    if isinstance(tweets, str): tweets = [tweets]
    rec = {"profile": prof, "tweet": tweets, "_age_days": float(num5[i,4])}
    try:
        units = render_v4.build_units_v4(rec, TH)
    except Exception:
        continue
    out.write(json.dumps({"account_id": uid.lstrip("u"), "label": "bot" if label[i]==1 else "human",
                          "units": [{"unit_id": u.unit_id, "text": u.text} for u in units]}, ensure_ascii=False) + "\n")
    n_ok += 1
out.close()
print(f"rendered {n_ok}/{len(target_raw_idx)} -> outputs/v5/tb22_fused_bots_render.jsonl", flush=True)
