"""Exercise the submitted HGT/RelNet classes through the new shared contract."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))
from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from verifiable_bot.aaai_veribot_core import (
    VeriBotDecision, VeriBotGraphInput, certify_aaai_source_response,
)
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord


class OriginalArchitecturePathTests(unittest.TestCase):
    def test_source_edge_reaches_both_original_branches_and_fusion(self):
        torch.manual_seed(23)
        ids = ("u0", "u1", "u2")
        x = {
            "user": torch.tensor([[1.0, 0.3], [0.2, 1.1], [0.7, -0.2]]),
            "hashtag": torch.ones((2, 1)),
        }
        edge_types = (
            ("user", "follows", "user"),
            ("user", "rev_follows", "user"),
            ("user", "used_hashtag", "hashtag"),
            ("hashtag", "rev_used_hashtag", "user"),
        )
        hgt = VerifiableHGT(
            metadata=(list(x), list(edge_types)),
            in_channels={name: value.shape[1] for name, value in x.items()},
            hidden_channels=8, out_channels=1, num_layers=2,
            heads=2, dropout=0.0)
        relnet = GatedRelNet(2, hidden_channels=8,
                             relation_names=("follows", "rev_follows"),
                             num_layers=2, dropout=0.0)
        decision = VeriBotDecision(hgt, relnet,
                                   hgt_sha256="toy-original-hgt-weights",
                                   relnet_sha256="toy-original-relnet-weights")
        sources = {
            "edge-a": SourceRecord("edge-a", "u0", "follow_pointer",
                                   {"target_user_id": "u1"}, "raw#edge-a"),
            "edge-b": SourceRecord("edge-b", "u2", "follow_pointer",
                                   {"target_user_id": "u1"}, "raw#edge-b"),
        }

        def edges(pairs):
            return (torch.tensor(pairs, dtype=torch.long).T.contiguous()
                    if pairs else torch.empty((2, 0), dtype=torch.long))

        def build(remaining):
            follows = [(ids.index(row.account_id), ids.index(row.payload["target_user_id"]))
                       for row in remaining.values()]
            forward = edges(follows)
            reverse = forward.flip(0)
            hetero = {
                edge_types[0]: forward,
                edge_types[1]: reverse,
                edge_types[2]: edges([(0, 0), (1, 1), (2, 0)]),
                edge_types[3]: edges([(0, 0), (1, 1), (0, 2)]),
            }
            return VeriBotGraphInput(
                x_dict=x, edge_index_dict=hetero,
                relnet_edges={"follows": forward, "rev_follows": reverse},
                account_ids=ids, user_fact_names=("fact_1", "fact_2"),
                source_dependencies={sid: ("HGT:follows", "RelNet:follows")
                                     for sid in remaining},
                graph_rule_sha256="toy-follow-rule")

        certificate = certify_aaai_source_response(
            account_id="u1",
            claim=AtomicClaim(("edge-a",), "field", "u1", field="target_user_id"),
            sources=sources, build_input=build, decision=decision,
            control_source_groups=(("edge-b",),))
        before, after = certificate["before"], certificate["after"]
        self.assertNotEqual(before["model_input_sha256"], after["model_input_sha256"])
        self.assertAlmostEqual(before["fused_logit"],
                               (before["hgt_logit"] + before["relnet_logit"]) / 2,
                               places=6)
        self.assertAlmostEqual(after["fused_logit"],
                               (after["hgt_logit"] + after["relnet_logit"]) / 2,
                               places=6)
        self.assertEqual(len(certificate["controls"]), 1)
        self.assertGreater(abs(certificate["hgt_delta_logit"]), 1e-7)
        self.assertGreater(abs(certificate["relnet_delta_logit"]), 1e-7)
        self.assertGreater(certificate["hgt_user_embedding_l2_change"], 1e-7)
        self.assertGreater(certificate["relnet_stage_l2_change"]["relation_layer_2"], 1e-7)
        self.assertLessEqual(before["relnet_forward_replay_max_error"], 1e-6)
        self.assertLessEqual(after["relnet_forward_replay_max_error"], 1e-6)
        # The submitted auxiliary HGT evidence head does not feed bot_logit.
        # The source-edge response above instead travels through real HGT
        # message passing and RelNet relation aggregation.
        built = build(sources)
        with torch.inference_mode():
            original = hgt(built.x_dict, built.edge_index_dict)
            original_bot = original["bot_logit"].clone()
            original_evidence = original["evidence_logit"].clone()
            hgt.evidence_head.weight.fill_(100.0)
            hgt.evidence_head.bias.fill_(-100.0)
            modified = hgt(built.x_dict, built.edge_index_dict)
        self.assertTrue(torch.equal(original_bot, modified["bot_logit"]))
        self.assertFalse(torch.equal(original_evidence, modified["evidence_logit"]))


if __name__ == "__main__":
    unittest.main()
