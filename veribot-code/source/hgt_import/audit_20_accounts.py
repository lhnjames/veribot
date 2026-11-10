"""Design doc 1.4 item 1: 20-account human-auditable sample. For each: raw per-direction attention
(head-mean, both directions when mutual), the relation each neighbor edge belongs to, the neighbor's
own model-predicted label (from THIS clean HGT + its frozen threshold, not ground truth), and the
resulting GRAPH_xx text -- side by side, so a human can check the aggregation is honest. Also
programmatically asserts the model-visible text contains ZERO Arabic numerals (design doc's explicit
requirement), not just eyeballed.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

with open(HERE / "data" / "graph_facts_sidecar.json") as f:
    sidecar = json.load(f)

rng = random.Random(0)
sample_ids = rng.sample(list(sidecar.keys()), 20)

digit_violations = []
lines = []
for uid in sample_ids:
    v = sidecar[uid]
    lines.append("=" * 78)
    lines.append(f"target account_id: {uid}")
    lines.append(f"GRAPH_01: {v['graph_01_text']}")
    lines.append(f"GRAPH_02: {v['graph_02_text']}")
    for text_field in ("graph_01_text", "graph_02_text"):
        text = v[text_field]
        if re.search(r"\d", text):
            digit_violations.append((uid, text_field, text))
    lines.append(f"raw top-{v['top_k_size']} neighbors, ranked by max(head-mean attention "
                 f"across directions):")
    for nb in v["top_k_neighbors"]:
        dirs_str = ", ".join(f"{d}={a:.4f}" for d, a in nb["per_direction_attention"].items())
        lines.append(f"  neighbor_id={nb['neighbor_account_id']:22s} "
                     f"direction={nb['direction']:9s} attn[{dirs_str}] "
                     f"ranking_attn={nb['ranking_attention']:.4f} "
                     f"model_predicted_label={nb['predicted_label']}")
    lines.append(f"bot_count_among_top_k = {v['bot_count_among_top_k']}/{v['top_k_size']} "
                 f"-> tier = {v['bot_fraction_tier']}")
    lines.append(f"top1: direction={v['top1_direction']} predicted={v['top1_predicted_label']}")

report = "\n".join(lines)
print(report)

print("\n" + "=" * 78)
print("DIGIT-FREE ASSERTION (design doc: model-visible GRAPH_xx text must contain zero Arabic "
      "numerals):")
if digit_violations:
    print(f"  FAILED: {len(digit_violations)} violation(s):")
    for uid, field, text in digit_violations:
        print(f"    {uid} {field}: {text!r}")
    sys.exit(1)
else:
    print(f"  PASSED: all {len(sample_ids)*2} GRAPH_xx text fields across {len(sample_ids)} "
          f"accounts contain zero digits.")

with open(HERE / "data" / "audit_20_accounts_report.txt", "w") as f:
    f.write(report)
print(f"\nfull report written to hgt_import/data/audit_20_accounts_report.txt")
