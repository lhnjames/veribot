#!/usr/bin/env python3
"""Create a train-only TwiBot-22 export. Official validation/test rows never leave source."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    boundaries = json.loads((args.source / "boundaries.json").read_text())
    n = int(boundaries["train_boundary"])

    tweet_output = args.output / "user_tweets.pkl"
    if not tweet_output.exists():
        with open(args.source / "user_tweets.pkl", "rb") as f:
            tweets = pickle.load(f)
        with open(tweet_output, "wb") as f:
            pickle.dump({int(k): v for k, v in tweets.items() if int(k) < n}, f, protocol=pickle.HIGHEST_PROTOCOL)
        del tweets

    with open(args.source / "description.pkl", "rb") as f:
        descriptions = pickle.load(f)
    with open(args.output / "description.pkl", "wb") as f:
        pickle.dump(descriptions[:n], f, protocol=pickle.HIGHEST_PROTOCOL)

    num = np.load(args.source / "num_properties_raw.npy", mmap_mode="r")
    np.save(args.output / "num_properties_raw.npy", np.asarray(num[:n]))
    cat = torch.load(args.source / "cat_properties_tensor.pt", map_location="cpu")
    torch.save(cat[:n], args.output / "cat_properties_tensor.pt")
    labels = torch.load(args.source / "label.pt", map_location="cpu")
    torch.save(labels[:n], args.output / "label.pt")

    uid_to_idx = torch.load(args.source / "uid_to_idx.pt", map_location="cpu")
    torch.save({uid: int(idx) for uid, idx in uid_to_idx.items() if int(idx) < n}, args.output / "uid_to_idx.pt")

    edge_index = torch.load(args.source / "edge_index.pt", map_location="cpu").long()
    edge_type = torch.load(args.source / "edge_type.pt", map_location="cpu").long()
    keep = (edge_index[0] < n) & (edge_index[1] < n)
    torch.save(edge_index[:, keep], args.output / "edge_index.pt")
    torch.save(edge_type[keep], args.output / "edge_type.pt")
    (args.output / "boundaries.json").write_text(json.dumps({"train_boundary": n, "val_end": n, "test_end": n}))
    (args.output / "EXPORT_MANIFEST.json").write_text(json.dumps({
        "source": "TwiBot-22", "official_split": "train_only", "accounts": n,
        "contains_validation": False, "contains_test": False,
        "tweet_content": "unaltered raw text, capped at 200 posts/account by source preprocessing"
    }, indent=2))


if __name__ == "__main__":
    main()
