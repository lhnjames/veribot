import json, math, numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def s(v):
    return (v or "").strip() if isinstance(v, str) else v
def num(v):
    try: return float(str(v).strip())
    except: return 0.0
def boo(v): return 1.0 if str(v).strip()=="True" else 0.0

def feats(r):
    p = r["profile"]
    fo, fr = num(p.get("followers_count")), num(p.get("friends_count"))
    st, fa, li = num(p.get("statuses_count")), num(p.get("favourites_count")), num(p.get("listed_count"))
    try:
        age = (datetime(2020,9,1) - datetime.strptime(s(p.get("created_at")), "%a %b %d %H:%M:%S %z %Y").replace(tzinfo=None)).days
    except: age = 0
    age = max(age, 1)
    name, sn, des = s(p.get("name")) or "", s(p.get("screen_name")) or "", s(p.get("description")) or ""
    dig_sn = sum(c.isdigit() for c in sn)
    tw = r.get("tweet") or []
    ntw = len(tw)
    rt = sum(1 for t in tw if isinstance(t,str) and t.startswith("RT @"))
    lens = [len(t) for t in tw if isinstance(t,str)]
    uniq = len(set(tw))/max(ntw,1)
    urls = sum(1 for t in tw if isinstance(t,str) and "http" in t)
    hsh = sum(t.count("#") for t in tw if isinstance(t,str))
    men = sum(t.count("@") for t in tw if isinstance(t,str))
    return [
        math.log1p(fo), math.log1p(fr), math.log1p(st), math.log1p(fa), math.log1p(li),
        fr/max(fo,1), fo/max(fr,1), st/age, fa/age, fo/age, fr/age,
        math.log1p(age), len(name), len(sn), dig_sn, dig_sn/max(len(sn),1), len(des),
        1.0 if des else 0.0, boo(p.get("verified")), boo(p.get("default_profile")),
        boo(p.get("default_profile_image")), boo(p.get("geo_enabled")), boo(p.get("protected")),
        1.0 if s(p.get("url")) else 0.0, 1.0 if s(p.get("location")) else 0.0,
        ntw, rt/max(ntw,1), np.mean(lens) if lens else 0, np.std(lens) if lens else 0,
        uniq, urls/max(ntw,1), hsh/max(ntw,1), men/max(ntw,1),
    ]

NAMES = ["log_followers","log_friends","log_statuses","log_favourites","log_listed",
    "friends/followers","followers/friends","statuses/age","fav/age","followers/age","friends/age",
    "log_age","len_name","len_screenname","digits_in_sn","digit_ratio_sn","len_desc",
    "has_desc","verified","default_profile","default_prof_img","geo_enabled","protected",
    "has_url","has_location","n_tweets","retweet_ratio","mean_tweet_len","std_tweet_len",
    "uniq_tweet_ratio","url_per_tweet","hashtag_per_tweet","mention_per_tweet"]

def load(f):
    d = json.load(open(f))
    X = np.array([feats(r) for r in d], dtype=np.float64)
    y = np.array([int(r["label"]) for r in d])
    return np.nan_to_num(X, posinf=0, neginf=0), y, [r["ID"] for r in d]

Xtr, ytr, _ = load("data/twibot20_complete/train.json")
Xte, yte, ids = load("data/twibot20_complete/test.json")
print("train", Xtr.shape, "test", Xte.shape, "bot rate", round(ytr.mean(),3), round(yte.mean(),3))

# A: profile-only (前 25 维)  B: profile + tweet 统计 (全部)
for tag, sl in [("profile-only", slice(0,25)), ("profile+tweetstats", slice(0,33))]:
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    clf.fit(Xtr[:,sl], ytr)
    pr = clf.predict_proba(Xte[:,sl])[:,1]
    pred = (pr>=0.5).astype(int)
    print(f"RF {tag:20s} ACC={accuracy_score(yte,pred):.4f}  AUC={roc_auc_score(yte,pr):.4f}  F1={f1_score(yte,pred):.4f}")
    if sl.stop==33:
        imp = sorted(zip(clf.feature_importances_, NAMES), reverse=True)[:12]
        print("  top features:", [(n, round(v,3)) for v,n in imp])
        np.save("./rf_test_prob.npy", pr)
        json.dump(ids, open("./rf_test_ids.json","w"))
