"""Fox8 preprocessing: build the standard gated-fusion bundle (num_properties_tensor.pt, text_tensor.pt,
label.pt, boundaries.json) from the pre-extracted feature bundle in /path/to/SAHG/data/fox8_23/ (features.csv:
31 numeric behavioral columns; roberta_features.pt: 768-d RoBERTa text embeddings, already computed;
labels.csv: balanced 1140/1140 human/bot). The dataset ships with NO follower graph -- Fox8's Twitter API
user objects expose follower/friend COUNTS but not the neighbor list -- so only the graph-less gated path
(text+behavioral fusion) applies here, same structural limitation as BotSim-24 (Reddit, no native graph).
"""
import csv, json, os
import numpy as np, torch

D = "/path/to/SAHG/data/fox8_23"
OUT = "/path/to/data_root/data/fox8_processed"
os.makedirs(OUT, exist_ok=True)


def main():
    rows = list(csv.DictReader(open(f"{D}/features.csv")))
    labels = [int(r["label"]) for r in csv.DictReader(open(f"{D}/labels.csv"))]
    assert len(rows) == len(labels), f"{len(rows)} feature rows vs {len(labels)} labels"
    y = torch.tensor(labels, dtype=torch.long)

    feat_cols = [c for c in rows[0].keys() if c != "user_id"]
    num = torch.tensor([[float(r[c]) for c in feat_cols] for r in rows], dtype=torch.float32)
    text = torch.load(f"{D}/roberta_features.pt", map_location="cpu", weights_only=False).float()
    assert text.shape[0] == len(rows), f"text {text.shape[0]} vs rows {len(rows)}"

    print(f"[fox8] N={len(rows)} bot_rate={float(y.float().mean()):.3f} num_feats={len(feat_cols)} "
          f"text_dim={text.shape[1]} (no follower graph -- Twitter API exposes counts, not neighbor list)")

    rng = np.random.default_rng(0); yb = y.numpy(); tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(yb == cls)[0]; rng.shuffle(idx)
        n = len(idx); a, b = int(0.7 * n), int(0.8 * n)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    perm = np.array(tr + va + te)

    num_p = num[perm]
    mu, std = num_p[:len(tr)].mean(0), num_p[:len(tr)].std(0).clamp(min=1e-6)
    num_p = (num_p - mu) / std

    torch.save(num_p, f"{OUT}/num_properties_tensor.pt")
    torch.save(text[perm], f"{OUT}/text_tensor.pt")
    torch.save(y[perm], f"{OUT}/label.pt")
    json.dump({"train_boundary": len(tr), "val_end": len(tr) + len(va), "test_end": len(rows),
               "note": f"num={len(feat_cols)}-d behavioral (verified/profile flags + engineered ratios), "
                       "text=768-d RoBERTa-base (precomputed), NO graph (Fox8 has no follower list)."},
              open(f"{OUT}/boundaries.json", "w"))
    print(f"[fox8] split train={len(tr)} val={len(va)} test={len(te)} -> {OUT}")


if __name__ == "__main__":
    main()
