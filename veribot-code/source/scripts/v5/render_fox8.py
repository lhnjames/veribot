"""Render Fox8-23 into the EXACT TwiBot-20 SFT unit format, for SemVerifier zero-shot.
Unlike TwiBot-22, Fox8 ships FULL native Twitter profile dicts (screen_name, listed_count, url,
location, created_at, ...) identical in shape to TwiBot-20's raw record -- so we use the REAL,
unmodified extract_features (no monkeypatch needed). This is the closest-to-training-distribution
zero-shot test we can run.
"""
import json, sys
sys.path.insert(0, "/path/to/project_root")
from verifiable_bot.v4 import render_v4, thresholds_v4

D = "/path/to/data_root/data/fox8/fox8_23_dataset.ndjson"
TB20_TRAIN = "/path/to/data_root/data/twibot20_complete/train.json"
OUT = "/path/to/project_root/outputs/v5/render_fox8_test.jsonl"

print("fitting TB20 thresholds...", flush=True)
TH = thresholds_v4.fit_thresholds(TB20_TRAIN)

n_ok = 0
with open(OUT, "w") as out:
    for line in open(D):
        r = json.loads(line)
        tweets_raw = r.get("user_tweets") or []
        if not tweets_raw:
            continue
        profile = tweets_raw[0].get("user") or {}
        texts = [t.get("text", "") for t in tweets_raw if isinstance(t, dict)]
        rec = {"profile": profile, "tweet": texts}
        try:
            units = render_v4.build_units_v4(rec, TH)
        except Exception:
            continue
        out.write(json.dumps({"account_id": str(r["user_id"]), "label": r["label"],
                              "source_subset": r.get("dataset"),
                              "units": [{"unit_id": u.unit_id, "text": u.text} for u in units]},
                             ensure_ascii=False) + "\n")
        n_ok += 1
print(f"rendered {n_ok} -> {OUT}", flush=True)
