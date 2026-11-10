import torch

from hgt_import.build_hgt_graph_sidecar import deduplicated_topk


def test_topk_uses_attention_and_deduplicates_mutual_neighbor():
    # Canonical follows: zero follows one, one follows zero (mutual), two follows zero.
    follower_src = torch.tensor([0, 1, 2])
    follower_dst = torch.tensor([1, 0, 0])
    follower_alpha = torch.tensor([0.2, 0.4, 0.9])
    # Reverse relation presents accounts the target follows as incoming messages.
    following_src = torch.tensor([1, 0, 0])
    following_dst = torch.tensor([0, 1, 2])
    following_alpha = torch.tensor([0.7, 0.3, 0.1])
    top = deduplicated_topk(
        3, follower_src, follower_dst, follower_alpha,
        following_src, following_dst, following_alpha,
    )
    target_zero = top["target"] == 0
    assert top["neighbor"][target_zero].tolist() == [2, 1]
    assert torch.allclose(top["attention"][target_zero], torch.tensor([0.9, 0.7]))
    # Neighbor one appears once and is tagged with both direction bits.
    assert top["direction_bits"][target_zero].tolist() == [1, 3]
