import json, math, numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

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

def text_of(r):
    p = r["profile"]
    tw = [t for t in (r.get("tweet") or []) if isinstance(t,str)][:60]
    return ((s(p.get("description")) or "") + " || " + (s(p.get("name")) or "") + " " +
            (s(p.get("screen_name")) or "") + " || " + " ".join(tw))

def load(f):
    d = json.load(open(f))
    X = np.nan_to_num(np.array([feats(r) for r in d], dtype=np.float64), posinf=0, neginf=0)
    T = [text_of(r) for r in d]
    y = np.array([int(r["label"]) for r in d])
    return X, T, y

Xtr, Ttr, ytr = load("data/twibot20_complete/train.json")
Xte, Tte, yte = load("data/twibot20_complete/test.json")

def rep(tag, pr):
    pred=(pr>=0.5).astype(int)
    print(f"{tag:38s} ACC={accuracy_score(yte,pred):.4f}  AUC={roc_auc_score(yte,pr):.4f}  F1={f1_score(yte,pred):.4f}")
    return pr

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xtr,ytr)
p_rf = rep("A 结构化特征 RF (33维)", rf.predict_proba(Xte)[:,1])

vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
Vtr = vec.fit_transform(Ttr); Vte = vec.transform(Tte)
lr = LogisticRegression(max_iter=2000, C=1.0).fit(Vtr, ytr)
p_tx = rep("B 纯文本 TF-IDF+LR (简介+推文)", lr.predict_proba(Vte)[:,1])

p_ens = rep("C  A+B logit平均", 1/(1+np.exp(-((np.log(p_rf/(1-p_rf+1e-9)+1e-9)+np.log(p_tx/(1-p_tx+1e-9)+1e-9))/2))))

# 有监督堆叠（用 train 的 5 折 OOF 更严谨，这里给一个乐观上界参考）
from sklearn.model_selection import cross_val_predict
oof_rf = cross_val_predict(RandomForestClassifier(n_estimators=300,min_samples_leaf=2,n_jobs=-1,random_state=0), Xtr, ytr, cv=5, method="predict_proba")[:,1]
oof_tx = cross_val_predict(LogisticRegression(max_iter=2000), Vtr, ytr, cv=5, method="predict_proba")[:,1]
st = LogisticRegression(max_iter=1000).fit(np.c_[oof_rf,oof_tx], ytr)
p_st = rep("D  A+B 有监督堆叠(OOF)", st.predict_proba(np.c_[p_rf,p_tx])[:,1])

print()
print("oracle 上界 P(A对 或 B对) =", round(np.mean(((p_rf>=.5).astype(int)==yte)|((p_tx>=.5).astype(int)==yte)),4))
print("互补率 P(A错 且 B对)      =", round(np.mean(((p_rf>=.5).astype(int)!=yte)&((p_tx>=.5).astype(int)==yte)),4))
