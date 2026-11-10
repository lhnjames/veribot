"""Design doc 1.4 item 1, degree-stratified supplement to the random-20 sample: the random sample
skewed heavily toward accounts with exactly 1 neighbor (attention trivially = 1.0), which validates
the aggregation arithmetic but not the RANKING behavior (nothing to rank among a single candidate).
This script selects >=10 accounts with >=5 unique ranked neighbors (top_k_size==5, since TOP_K=5
caps the list -- any account showing size 5 in the sidecar has at least 5 unique neighbors by
construction), stratified to also cover: both top1_predicted_label values (bot and human), and all
three of this account's own split memberships (hgt_train / hgt_val / project_holdout).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HGT_VAL_SALT = "hgt_val_v1"


def hgt_val_split(uid: str) -> str:
    key = f"{HGT_VAL_SALT}:{uid}"
    return "hgt_val" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0 else "hgt_train"


def project_holdout_split(uid: str) -> str:
    return "project_holdout" if int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 == 0 else "not_holdout"


def target_split_label(uid: str) -> str:
    if project_holdout_split(uid) == "project_holdout":
        return "project_holdout"
    return hgt_val_split(uid)  # "hgt_train" | "hgt_val"


with open(HERE / "data" / "graph_facts_sidecar.json") as f:
    sidecar = json.load(f)

eligible = {uid: v for uid, v in sidecar.items() if v["top_k_size"] == 5}
print(f"accounts with >=5 unique ranked neighbors (top_k_size==5): {len(eligible)}/{len(sidecar)}")

rng = random.Random(1)
ids = list(eligible.keys())
rng.shuffle(ids)

selected: list[str] = []
seen_pred = set()
seen_split = set()
# first pass: greedily grab accounts that add split/prediction diversity
for uid in ids:
    v = eligible[uid]
    pred = v["top1_predicted_label"]
    split = target_split_label(uid)
    if len(selected) < 10 and (pred not in seen_pred or split not in seen_split):
        selected.append(uid)
        seen_pred.add(pred)
        seen_split.add(split)
# fill up to at least 10 with more random picks if diversity criteria satisfied early
for uid in ids:
    if len(selected) >= 10:
        break
    if uid not in selected:
        selected.append(uid)

print(f"selected {len(selected)} accounts; "
      f"top1_predicted_label values covered: {sorted({eligible[u]['top1_predicted_label'] for u in selected})}; "
      f"target-split values covered: {sorted({target_split_label(u) for u in selected})}")

digit_violations = []
lines = []
for uid in selected:
    v = eligible[uid]
    split = target_split_label(uid)
    lines.append("=" * 78)
    lines.append(f"target account_id: {uid}  (this account's own split: {split})")
    lines.append(f"GRAPH_01: {v['graph_01_text']}")
    lines.append(f"GRAPH_02: {v['graph_02_text']}")
    for field in ("graph_01_text", "graph_02_text"):
        if re.search(r"\d", v[field]):
            digit_violations.append((uid, field, v[field]))
    lines.append(f"raw top-{v['top_k_size']} neighbors, ranked by max(head-mean attention across directions):")
    for nb in v["top_k_neighbors"]:
        dirs_str = ", ".join(f"{d}={a:.4f}" for d, a in nb["per_direction_attention"].items())
        lines.append(f"  neighbor_id={nb['neighbor_account_id']:22s} direction={nb['direction']:9s} "
                     f"attn[{dirs_str}] ranking_attn={nb['ranking_attention']:.4f} "
                     f"model_predicted_label={nb['predicted_label']}")
    recomputed_bot_count = sum(1 for nb in v["top_k_neighbors"] if nb["predicted_label"] == "bot")
    recomputed_frac = recomputed_bot_count / v["top_k_size"]
    if recomputed_frac > 0.6:
        recomputed_tier = "majority"
    elif recomputed_frac < 0.4:
        recomputed_tier = "a small minority"
    else:
        recomputed_tier = "about half"
    consistent = recomputed_tier == v["bot_fraction_tier"]
    lines.append(f"recomputed bot_count={recomputed_bot_count}/{v['top_k_size']} "
                 f"frac={recomputed_frac:.2f} -> recomputed_tier={recomputed_tier} "
                 f"(stored tier={v['bot_fraction_tier']}) "
                 f"{'CONSISTENT' if consistent else 'MISMATCH !!!'}")
    if not consistent:
        raise AssertionError(f"{uid}: tier mismatch on recompute")

report = "\n".join(lines)
print(report)

print("\n" + "=" * 78)
if digit_violations:
    print(f"DIGIT-FREE ASSERTION FAILED: {digit_violations}")
    sys.exit(1)
print(f"DIGIT-FREE ASSERTION PASSED: all {len(selected)*2} text fields, zero digits.")
print(f"TIER RECOMPUTE: all {len(selected)} accounts consistent (independently recomputed from "
      f"per-neighbor predicted labels, matching stored tier exactly).")

with open(HERE / "data" / "audit_degree_stratified_report.txt", "w") as f:
    f.write(report)
