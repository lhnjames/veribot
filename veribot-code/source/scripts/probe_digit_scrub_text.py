"""Does scrubbing digits out of free text cost the semantic channel anything?

The design doc's "de-numericalisation is free" evidence (83.94% bucketed vs 83.01% raw) is about
*structured features*. render_v4 additionally scrubs digits out of verbatim tweets and profile
descriptions, which is a different intervention with no evidence behind it. This measures it on the
same TF-IDF channel the doc quotes at 72.53%.
"""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DIGIT_RUN = re.compile(r"\d+(?:[.,:/\-]\d+)*")


def clean(v):
    return (v or "").strip() if isinstance(v, str) else v


def text_of(r):
    p = r["profile"]
    tw = [t for t in (r.get("tweet") or []) if isinstance(t, str)][:60]
    return ((clean(p.get("description")) or "") + " || " + (clean(p.get("name")) or "") + " "
            + (clean(p.get("screen_name")) or "") + " || " + " ".join(tw))


def load(f):
    d = json.load(open(f))
    return [text_of(r) for r in d], np.array([int(r["label"]) for r in d])


Ttr, ytr = load("data/twibot20_complete/train.json")
Tte, yte = load("data/twibot20_complete/test.json")

for tag, fn in (("原文（含数字）", lambda s: s), ("去数字化（render_v4 做法）", lambda s: DIGIT_RUN.sub("number", s))):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Vtr = vec.fit_transform([fn(t) for t in Ttr])
    Vte = vec.transform([fn(t) for t in Tte])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Vtr, ytr)
    pr = lr.predict_proba(Vte)[:, 1]
    pred = (pr >= 0.5).astype(int)
    print(f"{tag:26s} ACC={accuracy_score(yte, pred):.4f}  AUC={roc_auc_score(yte, pr):.4f}  "
          f"F1={f1_score(yte, pred):.4f}  vocab={len(vec.vocabulary_)}")
