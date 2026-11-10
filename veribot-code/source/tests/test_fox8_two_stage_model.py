import unittest

import torch

from source.hgt_import.code.relnet_reimplementation import GatedRelNet
from source.verifiable_bot.fox8_atomic_sources import AccountAtoms, RelationAtom, TweetAtom
from source.verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs
from source.verifiable_bot.fox8_two_stage_certificate import relnet_source_closure
from source.verifiable_bot.fox8_two_stage_intervention import TwoStageIntervention
from source.verifiable_bot.fox8_two_stage_model import HGTSourceSelector, TwoStageFox8


class TwoStageModelTests(unittest.TestCase):
    def test_full_path_intervention_rebuilds_hgt_and_relnet_inputs(self):
        class TinyHGT(torch.nn.Module):
            def forward(self, x_dict, edge_index_dict):
                return {"user_emb": x_dict["user"][:, :8]}

        def account(uid, line):
            tweet = TweetAtom(0, line, uid, 0, 0, 3, 0, 0, "en",
                              1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, line, {"id_str": uid, "followers_count": 1},
                                (tweet,))

        ids = ["a", "b"]
        relation = RelationAtom("r", "a", "b", "reply", ((1, 0, "reply"),))
        bank = Fox8TwoStageInputs.build(
            {uid: account(uid, i + 1) for i, uid in enumerate(ids)}, (relation,), ids)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(bank.full_graph),
                             num_layers=2, dropout=0.0, out_channels=2)
        model = TwoStageFox8(HGTSourceSelector(context_dim=8, hidden=12), relnet,
                             torch.zeros(len(ALL_FACTS)), torch.ones(len(ALL_FACTS)))
        oracle = TwoStageIntervention(hgt=TinyHGT(), model=model, batch=bank,
                                      base_mean=torch.zeros(31).numpy(),
                                      base_scale=torch.ones(31).numpy())
        source = bank.accounts[0].source_id("profile", "followers_count")
        self.assertIn(source, oracle.target_source_ids("a"))
        baseline = oracle.predict("a")
        changed = oracle.predict("a", frozenset({source}))
        self.assertTrue(torch.isfinite(torch.tensor(baseline["margin"])))
        self.assertTrue(torch.isfinite(torch.tensor(changed["margin"])))
        with self.assertRaisesRegex(ValueError, "absent"):
            oracle.predict("a", frozenset({"missing"}))

    def test_certificate_includes_two_hops_and_excludes_third_hop(self):
        def account(uid, line):
            tweet = TweetAtom(0, line, uid, 0, 0, 3, 0, 0, "en",
                              1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, line, {"id_str": uid, "followers_count": 1},
                                (tweet,))

        ids = ["a", "b", "c", "d"]
        relations = tuple(RelationAtom(f"r{i}", src, dst, "reply",
                                       ((i + 1, 0, "reply"),))
                          for i, (src, dst) in enumerate(zip(ids[:-1], ids[1:])))
        bank = Fox8TwoStageInputs.build(
            {uid: account(uid, i + 1) for i, uid in enumerate(ids)}, relations, ids)
        masks = {"profile": torch.ones((len(bank.profile_account), 2)),
                 "tweet": torch.ones((len(bank.tweet_account), 2)),
                 "relation": torch.ones((len(bank.relations), 2))}
        result = relnet_source_closure(bank, "d", masks, 1)
        self.assertEqual(set(result["receptive_field_accounts"]), {"b", "c", "d"})
        self.assertEqual(result["source_counts"]["native_relation_event"], 2)
        self.assertTrue(all(item["source_id"] for item in result["source_records"]))

    def test_both_class_views_use_selected_sources_and_keep_gradients(self):
        def account(uid, source):
            atom = TweetAtom(0, source, uid, 0, 0, 5, 1, 0, "en",
                             1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, source,
                                {"id_str": uid, "followers_count": 5}, (atom,))

        ids = ["a", "b"]
        bank = {"a": account("a", 1), "b": account("b", 2)}
        relation = RelationAtom("r", "a", "b", "reply", ((1, 0, "reply"),))
        batch = Fox8TwoStageInputs.build(bank, (relation,), ids)
        selector = HGTSourceSelector(context_dim=8, hidden=12)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(batch.full_graph),
                             num_layers=2, dropout=0.0, out_channels=2)
        model = TwoStageFox8(selector, relnet,
                             torch.zeros(len(ALL_FACTS)), torch.ones(len(ALL_FACTS)))
        context = torch.randn(2, 8)
        model.train()
        output = model(context, batch, compute_complement=True)
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertEqual(tuple(output["complement_logits"].shape), (2, 2))
        (output["logits"].square().mean() +
         output["complement_logits"].square().mean()).backward()
        self.assertIsNotNone(selector.profile_head[-1].weight.grad)
        self.assertIsNotNone(selector.tweet_head[-1].weight.grad)
        self.assertIsNotNone(selector.relation_head[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
