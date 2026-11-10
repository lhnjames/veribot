"""Fox8 has NO real relational field anywhere in the available data (Twitter API only exposes
follower/friend COUNTS, never the neighbor list; no hashtag/mention lists survive in the SAHG bundle
either). Per user request, we CONSTRUCT a graph as additional input: a kNN similarity graph (k=10,
cosine similarity over z-normalized [behavioral(31d) || text(768d)] features). This is explicitly a
DIFFERENT KIND of graph from TB20/22/BotSim's real social/entity structures -- a derived similarity
network, not an observed relationship. Documented as such, not presented as a real social graph.

Reuses the existing fox8_processed/{num_properties_tensor,text_tensor,label}.pt + boundaries.json
(same split order) so results are directly comparable to the 'gated' (graph-less) baseline.

Outputs (data_dir = /path/to/data_root/data/fox8_graph):
  features.pt, edge_index_dict.pt, label.pt, boundaries.json   -- for train_hgt_homogeneous
  des_tensor.pt, tweets_tensor.pt, num_properties_tensor.pt, cat_properties_tensor.pt,
  edge_index.pt, edge_type.pt                                  -- for train_relnet
"""
import json, os, shutil
import numpy as np, torch

SRC = "/path/to/data_root/data/fox8_processed"
OUT = "/path/to/data_root/data/fox8_graph"
os.makedirs(OUT, exist_ok=True)
K = 10


def main():
    num = torch.load(f"{SRC}/num_properties_tensor.pt", map_location="cpu").float()
    text = torch.load(f"{SRC}/text_tensor.pt", map_location="cpu").float()
    label = torch.load(f"{SRC}/label.pt", map_location="cpu").long()
    boundaries = json.load(open(f"{SRC}/boundaries.json"))
    N = num.shape[0]

    text_n = text / text.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    num_n = num / num.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    combo = torch.cat([num_n, text_n], dim=-1)
    combo = combo / combo.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    sim = combo @ combo.T
    sim.fill_diagonal_(-2.0)
    topk = sim.topk(K, dim=-1).indices  # (N, K) nearest neighbor indices per node
    src = torch.arange(N).unsqueeze(1).expand(-1, K).reshape(-1)
    dst = topk.reshape(-1)
    ei = torch.stack([src, dst])
    ei = torch.cat([ei, ei.flip(0)], dim=1)  # symmetrize
    ei = torch.unique(ei, dim=1)
    print(f"[fox8-knn] N={N} k={K} symmetrized unique edges={ei.shape[1]}")

    json.dump({**boundaries, "note": boundaries.get("note", "") +
               f" CONSTRUCTED kNN(k={K}) cosine-similarity graph over [behavioral||text] -- "
               "NOT a real social/entity graph, no follower data exists for Fox8."},
              open(f"{OUT}/boundaries.json", "w"))
    torch.save(label, f"{OUT}/label.pt")

    # ---- HGT-homogeneous tensors ----
    feat = torch.cat([num, text], dim=-1)
    torch.save(feat, f"{OUT}/features.pt")
    edge_index_dict = {("user", "knn_sim", "user"): ei}
    torch.save(edge_index_dict, f"{OUT}/edge_index_dict.pt")

    # ---- RelNet tensors ----
    torch.save(text, f"{OUT}/des_tensor.pt")
    torch.save(text, f"{OUT}/tweets_tensor.pt")   # no separate tweet-only embedding available
    torch.save(num, f"{OUT}/num_properties_tensor.pt")
    torch.save(torch.zeros(N, 1), f"{OUT}/cat_properties_tensor.pt")
    torch.save(ei, f"{OUT}/edge_index.pt")
    torch.save(torch.zeros(ei.shape[1], dtype=torch.long), f"{OUT}/edge_type.pt")
    print(f"[fox8-knn] saved -> {OUT}")


if __name__ == "__main__":
    main()
