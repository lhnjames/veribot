#!/usr/bin/env python3
"""Create safe smoke SFT targets without inventing evidence or calling a teacher."""
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--limit",type=int,default=32); a=ap.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
with open(a.input,encoding="utf-8") as src, open(a.output,"w",encoding="utf-8") as dst:
    for i,line in enumerate(src):
        if i>=a.limit: break
        record=json.loads(line)
        target={"label":record["label"],"confidence":0.5,"evidence_status":"insufficient",
                "core_claim":"","evidence_kind":"","anchors":[],"alternative_explanation":"Insufficient visible evidence."}
        dst.write(json.dumps({"record":record,"target":target},ensure_ascii=False)+"\n")
