#!/usr/bin/env python3
"""Remote acceptance check proving that an export contains only official training rows."""
import argparse, json, pickle
from pathlib import Path
import torch

ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path); args = ap.parse_args()
m = json.loads((args.root / "EXPORT_MANIFEST.json").read_text())
assert m["official_split"] == "train_only" and not m["contains_validation"] and not m["contains_test"]
n = m["accounts"]
labels = torch.load(args.root / "label.pt", map_location="cpu")
uids = torch.load(args.root / "uid_to_idx.pt", map_location="cpu")
edges = torch.load(args.root / "edge_index.pt", map_location="cpu")
with open(args.root / "description.pkl", "rb") as f: descriptions = pickle.load(f)
assert len(labels) == len(uids) == len(descriptions) == n
assert edges.numel() == 0 or int(edges.max()) < n
print(json.dumps({"train_only": True, "accounts": n, "edges": edges.shape[1]}))
