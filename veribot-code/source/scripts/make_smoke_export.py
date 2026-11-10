#!/usr/bin/env python3
"""Build a tiny official-train-only two-dataset export for remote pipeline smoke tests."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--twibot20-train", type=Path, required=True)
ap.add_argument("--twibot22-train", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--accounts", type=int, default=128)
a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)

users = json.loads(a.twibot20_train.read_text())[:a.accounts]
(a.output / "twibot20_train.json").write_text(json.dumps(users, ensure_ascii=False))

with open(a.twibot22_train / "user_tweets.pkl", "rb") as f: tweets = pickle.load(f)
chosen = sorted(tweets)[:a.accounts]
old_to_new = {old: new for new, old in enumerate(chosen)}
with open(a.output / "user_tweets.pkl", "wb") as f:
    pickle.dump({old_to_new[k]: tweets[k] for k in chosen}, f)
with open(a.twibot22_train / "description.pkl", "rb") as f: desc = pickle.load(f)
with open(a.output / "description.pkl", "wb") as f: pickle.dump([desc[k] for k in chosen], f)
num = np.load(a.twibot22_train / "num_properties_raw.npy", mmap_mode="r")
np.save(a.output / "num_properties_raw.npy", np.asarray(num[chosen]))
cat = torch.load(a.twibot22_train / "cat_properties_tensor.pt", map_location="cpu")
labels = torch.load(a.twibot22_train / "label.pt", map_location="cpu")
torch.save(cat[chosen], a.output / "cat_properties_tensor.pt")
torch.save(labels[chosen], a.output / "label.pt")
uids = torch.load(a.twibot22_train / "uid_to_idx.pt", map_location="cpu")
idx_to_uid = {int(v): k for k, v in uids.items()}
torch.save({idx_to_uid[k]: old_to_new[k] for k in chosen}, a.output / "uid_to_idx.pt")
edge = torch.load(a.twibot22_train / "edge_index.pt", map_location="cpu").long()
etype = torch.load(a.twibot22_train / "edge_type.pt", map_location="cpu").long()
keep = torch.tensor([(int(s) in old_to_new and int(d) in old_to_new) for s, d in edge.t()])
kept = edge[:, keep]
if kept.numel(): kept = torch.tensor([[old_to_new[int(s)], old_to_new[int(d)]] for s, d in kept.t()]).t()
torch.save(kept, a.output / "edge_index.pt"); torch.save(etype[keep], a.output / "edge_type.pt")
n=len(chosen); (a.output / "boundaries.json").write_text(json.dumps({"train_boundary":n,"val_end":n,"test_end":n}))
(a.output / "EXPORT_MANIFEST.json").write_text(json.dumps({"official_split":"train_only","accounts":n,
 "contains_validation":False,"contains_test":False,"purpose":"remote_smoke_only"},indent=2))
