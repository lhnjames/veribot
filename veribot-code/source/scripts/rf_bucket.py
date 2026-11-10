import json, math, numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
exec(open("./rf_baseline.py").read().split("NAMES =")[0].split("import json")[1].replace("from datetime","",1).join(["",""]) ) if False else None

def s(v): return (v or "").strip() if isinstance(v, str) else v
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
    ntw = len(tw); rt = sum(1 for t in tw if isinstance(t,str) and t.startswith("RT @"))
    lens = [len(t) for t in tw if isinstance(t,str)]
    uniq = len(set(tw))/max(ntw,1)
    urls = sum(1 for t in tw if isinstance(t,str) and "http" in t)
    hsh = sum(t.count("#") for t in tw if isinstance(t,str))
    men = sum(t.count("@") for t in tw if isinstance(t,str))
    return [math.log1p(fo), math.log1p(fr), math.log1p(st), math.log1p(fa), math.log1p(li),
        fr/max(fo,1), fo/max(fr,1), st/age, fa/age, fo/age, fr/age,
        math.log1p(age), len(name), len(sn), dig_sn, dig_sn/max(len(sn),1), len(des),
        1.0 if des else 0.0, boo(p.get("verified")), boo(p.get("default_profile")),
        boo(p.get("default_profile_image")), boo(p.get("geo_enabled")), boo(p.get("protected")),
        1.0 if s(p.get("url")) else 0.0, 1.0 if s(p.get("location")) else 0.0,
        ntw, rt/max(ntw,1), np.mean(lens) if lens else 0, np.std(lens) if lens else 0,
        uniq, urls/max(ntw,1), hsh/max(ntw,1), men/max(ntw,1)]

def load(f):
    d = json.load(open(f))
    X = np.array([feats(r) for r in d], dtype=np.float64)
    y = np.array([int(r["label"]) for r in d])
    return np.nan_to_num(X, posinf=0, neginf=0), y

Xtr, ytr = load("data/twibot20_complete/train.json")
Xte, yte = load("data/twibot20_complete/test.json")

CONT = [i for i in range(33) if i not in (17,18,19,20,21,22,23,24)]  # 连续维（其余是本来就二元的）

def run(tag, Xa, Xb):
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    clf.fit(Xa, ytr); pr = clf.predict_proba(Xb)[:,1]; pred=(pr>=0.5).astype(int)
    print(f"{tag:34s} ACC={accuracy_score(yte,pred):.4f}  AUC={roc_auc_score(yte,pr):.4f}  F1={f1_score(yte,pred):.4f}")

run("原始数值 (raw)", Xtr, Xte)

for nb, name in [(5,"5档"),(10,"10档")]:
    qs = np.linspace(0,100,nb+1)[1:-1]
    Btr, Bte = Xtr.copy(), Xte.copy()
    for i in CONT:
        cut = np.percentile(Xtr[:,i], qs)
        Btr[:,i] = np.digitize(Xtr[:,i], cut)
        Bte[:,i] = np.digitize(Xte[:,i], cut)
    run(f"分桶去数字化 ({name}) — v3 做法", Btr, Bte)

# 只保留 v3 实际渲染的那几个字段: followers, friends, statuses, listed, age, verified, has_desc
V3 = [0,1,2,4,11,18,17]
run("v3 实际可见字段 (raw)", Xtr[:,V3], Xte[:,V3])
qs = np.linspace(0,100,6)[1:-1]
Btr, Bte = Xtr[:,V3].copy(), Xte[:,V3].copy()
for j,i in enumerate(V3):
    if i in (17,18): continue
    cut = np.percentile(Xtr[:,i], qs)
    Btr[:,j] = np.digitize(Xtr[:,i], cut); Bte[:,j] = np.digitize(Xte[:,i], cut)
run("v3 实际可见字段 + 5档分桶", Btr, Bte)
