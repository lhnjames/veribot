"""Verify matrix aggregation is the same gated relation mean operation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from relnet_reimplementation import GatedRelNet
from verifiable_bot.aaai_dense_relnet import DenseGatedRelNet, normalized_adjacency


class DenseRelationEquivalenceTests(unittest.TestCase):
    def test_source_edge_gates_match_rebuilt_selected_graph(self):
        torch.manual_seed(17)
        full = {"reply": torch.tensor([[0, 1, 2], [2, 2, 0]], dtype=torch.long),
                "rev_reply": torch.tensor([[2, 2, 0], [0, 1, 2]], dtype=torch.long)}
        chosen = {"reply": full["reply"][:, [0, 2]],
                  "rev_reply": full["rev_reply"][:, [0, 2]]}
        weights = {name: torch.tensor([1.0, 0.0, 1.0]) for name in full}
        model = GatedRelNet(4, hidden_channels=8,
                            relation_names=tuple(full), num_layers=2,
                            dropout=0.0, out_channels=2).eval()
        x = torch.randn(3, 4)
        gated = model(x, full, weights)
        rebuilt = model(x, chosen)
        self.assertEqual(gated.shape, (3, 2))
        torch.testing.assert_close(gated, rebuilt, atol=1e-6, rtol=1e-6)

    def test_forward_and_input_gradient_match_edge_mean(self):
        torch.manual_seed(41)
        relation_names = ("shared_subreddit", "reply")
        edges = {
            "shared_subreddit": torch.tensor([[0, 1, 1, 2, 4],
                                                 [1, 0, 0, 1, 3]], dtype=torch.long),
            "reply": torch.tensor([[0, 3, 4], [2, 2, 1]], dtype=torch.long),
        }
        adjacency = {name: normalized_adjacency(edge, 5)
                     for name, edge in edges.items()}
        direct = GatedRelNet(3, hidden_channels=8,
                             relation_names=relation_names,
                             num_layers=2, dropout=0.0).eval()
        matrix = DenseGatedRelNet(3, hidden_channels=8,
                                  relation_names=relation_names,
                                  num_layers=2, dropout=0.0).eval()
        matrix.load_state_dict(direct.state_dict(), strict=True)
        x_a = torch.randn((5, 3), requires_grad=True)
        x_b = x_a.detach().clone().requires_grad_(True)
        output_a = direct(x_a, edges)
        output_b = matrix(x_b, adjacency)
        torch.testing.assert_close(output_a, output_b, atol=1e-6, rtol=1e-6)
        output_a.square().sum().backward()
        output_b.square().sum().backward()
        torch.testing.assert_close(x_a.grad, x_b.grad, atol=1e-6, rtol=1e-6)
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
                direct.named_parameters(), matrix.named_parameters()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad,
                                       atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
