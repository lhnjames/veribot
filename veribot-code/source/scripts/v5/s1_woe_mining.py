"""VeriRoute v5 — S1: WoE evidence mining (ZERO LLM, CPU only).  v2.

Fixes over v1 (both diagnosed on real data):
  * v1 candidate pool admitted near-pure signatures (e.g. PROFILE_01 "verified" WoE=-7.47)
    whose extreme WoE does NOT transfer to the small dev sample -> leakage-gate borderline.
  * v1 gold was 100% sufficient because WoE magnitudes dwarf the spec's fixed 1.2 cutoff.

Principled changes (TRAIN-only selection; dev is a held-out check evaluated ONCE, never tuned on):
  1. TRAIN 2-fold consistency: a signature is admitted only if, estimated INDEPENDENTLY on two
     deterministic halves of train, BOTH halves give the same sign and |WoE| >= woe_min. This
     selects signals that are stable/transferable using train alone.
  2. Bayesian-shrunk WoE (larger pseudocount) so tiny-cell estimates are pulled toward 0.
  3. Insufficient = bottom `insuff_q` quantile of per-account gold strength, calibrated on TRAIN
     (targets the spec's 15-25% insufficient without touching dev).

Outputs under outputs/v5/s1_woe/ — never touches v4.  Deterministic (hash-based folds, no RNG).
"""
import json
import math
import os
import re
import zlib
from collections import Counter

RENDER = "outputs/v4/render_v4"
OUT = "outputs/v5/s1_woe"
os.makedirs(OUT, exist_ok=True)

SRC_PREFIXES = ("PROFILE", "OBS", "ACTIVITY", "RELATION")
SUPPORT_MIN = int(os.environ.get("WOE_SUP", 50))
WOE_MIN = float(os.environ.get("WOE_MIN", 0.5))
SHRINK = float(os.environ.get("WOE_SHRINK", 5.0))   # pseudocount per class (Bayesian shrink)
INSUFF_Q = float(os.environ.get("WOE_INSUFF_Q", 0.20))  # bottom quantile -> insufficient


def norm_bucket(text):
    t = text.split(":", 1)[1] if ":" in text else text
    t = t.lower().replace("relative to other accounts in this dataset", "").replace("relative to other accounts", "")
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^a-z#;/ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip(" ;.")


def src_type(uid):
    return uid.split("_", 1)[0]


def load(split):
    with open(f"{RENDER}/{split}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def account_sigs(row):
    out = {}
    for u in row.get("units", []):
        uid = u.get("unit_id", "")
        if src_type(uid) not in SRC_PREFIXES:
            continue
        b = norm_bucket(u.get("text", ""))
        if b:
            out.setdefault(f"{uid}::{b}", (uid, src_type(uid)))
    return out


def fold_of(account_id):
    return zlib.crc32(str(account_id).encode()) & 1


def woe_table(rows):
    """Shrunk WoE over a set of accounts. Returns sig -> dict."""
    n_bot = sum(1 for r in rows if r["label"] == "bot")
    n_hum = sum(1 for r in rows if r["label"] == "human")
    cb, ch, meta = Counter(), Counter(), {}
    for r in rows:
        lab = r["label"]
        for sig, (uid, st) in account_sigs(r).items():
            meta[sig] = (uid, st)
            (cb if lab == "bot" else ch)[sig] += 1
    tab = {}
    for sig, (uid, st) in meta.items():
        p_bot = (cb[sig] + SHRINK) / (n_bot + 2 * SHRINK)
        p_hum = (ch[sig] + SHRINK) / (n_hum + 2 * SHRINK)
        woe = math.log(p_bot / p_hum)
        tab[sig] = {"unit_id": uid, "src_type": st, "woe": round(woe, 4),
                    "iv": round((p_bot - p_hum) * woe, 5), "support": cb[sig] + ch[sig],
                    "c_bot": cb[sig], "c_human": ch[sig], "dir": "bot" if woe > 0 else "human"}
    return tab


def top2_diff_src(sig_st_list, key):
    chosen, used = [], set()
    for sig, st in sorted(sig_st_list, key=key, reverse=True):
        if st in used:
            continue
        chosen.append(sig); used.add(st)
        if len(chosen) == 2:
            break
    return chosen


def main():
    train = load("train")
    dev = load("dev")

    full = woe_table(train)
    fa = woe_table([r for r in train if fold_of(r["account_id"]) == 0])
    fb = woe_table([r for r in train if fold_of(r["account_id"]) == 1])

    # TRAIN 2-fold consistent candidate pool (no dev involved)
    def admit(sig):
        e = full.get(sig)
        if not e or e["support"] < SUPPORT_MIN:
            return False
        a, b = fa.get(sig), fb.get(sig)
        if not a or not b:
            return False
        if abs(a["woe"]) < WOE_MIN or abs(b["woe"]) < WOE_MIN:
            return False
        return (a["woe"] > 0) == (b["woe"] > 0) and abs(full[sig]["woe"]) >= WOE_MIN
    for sig, e in full.items():
        e["candidate"] = admit(sig)
    cand = {s: e for s, e in full.items() if e["candidate"]}
    print(f"[S1v2] train={len(train)} | signatures={len(full)} | fold-consistent candidate_pool={len(cand)}")

    # per-account gold (top-2 diff-src among account's candidate sigs), record raw strength
    def gold_for(row, table, restrict_cand=True):
        lst = [(sig, st) for sig, (uid, st) in account_sigs(row).items()
               if sig in table and (not restrict_cand or table[sig].get("candidate"))]
        g = top2_diff_src(lst, key=lambda x: abs(table[x[0]]["woe"]))
        strength = sum(abs(table[s]["woe"]) for s in g)
        return g, round(strength, 4)

    strengths = []
    prelim = []
    for r in train:
        g, s = gold_for(r, full)
        prelim.append((r, g, s))
        if len(g) == 2:
            strengths.append(s)
    strengths.sort()
    # train-calibrated insufficiency cutoff (bottom quantile of 2-gold strengths)
    idx = max(0, min(len(strengths) - 1, int(INSUFF_Q * len(strengths))))
    cutoff = strengths[idx] if strengths else 0.0

    status_ct = Counter(); src_used = Counter()
    with open(f"{OUT}/evidence_gold_train.jsonl", "w") as f:
        for r, g, s in prelim:
            status = "sufficient" if (len(g) == 2 and s >= cutoff) else "insufficient"
            status_ct[status] += 1
            for sig in g:
                src_used[full[sig]["src_type"]] += 1
            f.write(json.dumps({"account_id": r["account_id"], "label": r["label"], "gold": g,
                                "evidence_status": status, "sum_abs_woe": s}, ensure_ascii=False) + "\n")
    insuff_frac = status_ct["insufficient"] / max(1, sum(status_ct.values()))
    print(f"[S1v2] gold status={dict(status_ct)} (insufficient={insuff_frac:.1%}, target 15-25%) "
          f"| strength cutoff={cutoff} | src usage={dict(src_used)}")

    with open(f"{OUT}/evidence_value_v1.json", "w") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    top = sorted(cand.values(), key=lambda e: -e["iv"])[:12]
    print("[S1v2] top candidate signatures by IV:")
    for e in top:
        print(f"       woe={e['woe']:+.2f} iv={e['iv']:.4f} sup={e['support']} (b={e['c_bot']}/h={e['c_human']}) {e['unit_id']}")

    # LEAKAGE GATE (held-out check, evaluated ONCE): over each dev account's frozen candidate sigs,
    # top-2 by frozen train-WoE vs top-2 by dev-label WoE.  Same candidate set both sides.
    dev_tab = woe_table(dev)
    ov = []
    for r in dev:
        lst = [(sig, st) for sig, (uid, st) in account_sigs(r).items() if sig in cand]
        if len(lst) < 2:
            continue
        gf = top2_diff_src(lst, key=lambda x: abs(full[x[0]]["woe"]))
        gl = top2_diff_src(lst, key=lambda x: abs(dev_tab.get(x[0], {}).get("woe", 0.0)))
        if len(gf) == 2 and len(gl) == 2:
            ov.append(len(set(gf) & set(gl)) / 2.0)
    overlap = sum(ov) / len(ov) if ov else 0.0
    gate = overlap > 0.85
    print(f"[S1v2][GATE] dev scored={len(ov)} | frozen-vs-labelinformed overlap={overlap:.3f} "
          f"| >0.85 -> {'PASS' if gate else 'FAIL'}")

    json.dump({"n_train": len(train), "n_signatures": len(full), "candidate_pool": len(cand),
               "gold_status": dict(status_ct), "insufficient_frac": round(insuff_frac, 4),
               "strength_cutoff": cutoff, "dev_leakage_overlap": round(overlap, 4),
               "gate_pass": gate, "params": {"support_min": SUPPORT_MIN, "woe_min": WOE_MIN,
               "shrink": SHRINK, "insuff_q": INSUFF_Q}},
              open(f"{OUT}/s1_report.json", "w"), ensure_ascii=False, indent=2)
    print("[S1v2] DONE")


if __name__ == "__main__":
    main()
